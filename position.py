"""The board, compiled: bitboards, move generation, make and unmake, hashing and exchanges.

Everything here runs inside numba, so it uses only integers and NumPy arrays. python-chess still
reads the FEN, plays the book and checks the final answer in agent.py, but the search never
touches it: one python-chess push costs about 13 us, the whole node here costs well under one.

Squares follow python-chess, a1 = 0 to h8 = 63, so bitboards pass straight to the network.
Pieces are numbered colour * 6 + type, White pawn 0 to Black king 11, and EMPTY is -1.

A move is one int32:

    bits 0-5    from square
    bits 6-11   to square
    bits 12-14  promotion piece type (KNIGHT to QUEEN), 0 for none
    bits 15-16  NORMAL, DOUBLE_PUSH, CASTLING or EN_PASSANT

Moves are generated pseudo-legally. make_move plays one, and takes it back and returns False if
it leaves the mover's king attacked, so a caller only ever searches legal moves.

numba follows NumPy's typing: an unsigned 64-bit bitboard combined with a plain Python int turns
into a float. Every bitboard constant is therefore a np.uint64, and single squares come from
SQUARE_BB rather than `1 << square`.

Every compiled function has its argument types written out. Without them numba compiles a
separate copy for each constant a caller passes: 88 copies of _ray from move generation alone,
and a second whole search for `ply=1`, which took the start-up compile past the 90 s budget.
With them each function is compiled once, when this module is imported.
"""

import os

import chess
import numba
import numpy as np
from numba import boolean, float32, int8, int16, int32, int64, njit, types, uint64
from numba.experimental import structref

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
EMPTY = -1

NORMAL, DOUBLE_PUSH, CASTLING, EN_PASSANT = 0, 1, 2, 3
NULL_MOVE = 0  # a1a1, never a real move
MAX_MOVES = 256
# The search stops at ply 96; quiescence and extensions can go a little past the node that stops.
MAX_PLY = 128
# The network's accumulator width, and the eight output buckets its PSQT term is kept for.
ACCUMULATOR_WIDTH = 1024
PSQT_BUCKETS = 8

WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE = 1, 2, 4, 8

# Local testing only. The referee starts a fresh process for every game, so each game compiles
# the engine again. With NNUE_CHESS_CACHE_DIR set, agent.py points numba's cache at that
# directory: compiled functions are written there once and loaded in every later game. The
# platform never sets it, and wipes its scratch space between games anyway.
CACHE = bool(os.environ.get("NNUE_CHESS_CACHE_DIR"))

U0 = np.uint64(0)
U1 = np.uint64(1)

# --- tables, built once at import ----------------------------------------------------------

SQUARE_BB = np.array([1 << square for square in range(64)], dtype=np.uint64)
KNIGHT_ATTACKS = np.array(chess.BB_KNIGHT_ATTACKS, dtype=np.uint64)
KING_ATTACKS = np.array(chess.BB_KING_ATTACKS, dtype=np.uint64)
# PAWN_ATTACKS[colour, square]: the squares a pawn of that colour on that square attacks.
PAWN_ATTACKS = np.array(
    [chess.BB_PAWN_ATTACKS[chess.WHITE], chess.BB_PAWN_ATTACKS[chess.BLACK]], dtype=np.uint64
)
RANK_1, RANK_2, RANK_7, RANK_8 = (np.uint64(chess.BB_RANKS[rank]) for rank in (0, 1, 6, 7))
DARK_SQUARES = np.uint64(chess.BB_DARK_SQUARES)
LIGHT_SQUARES = np.uint64(chess.BB_LIGHT_SQUARES)

# Rays run from a square to the edge of the board, not including the square itself. The first
# four directions walk towards higher square numbers, where the nearest blocker is the lowest
# set bit; the last four walk down, where it is the highest.
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (-1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1))
EAST, NORTH, NORTH_EAST, NORTH_WEST, WEST, SOUTH, SOUTH_WEST, SOUTH_EAST = range(8)


def _rays() -> np.ndarray:
    rays = np.zeros((8, 64), dtype=np.uint64)
    for direction, (file_step, rank_step) in enumerate(DIRECTIONS):
        for square in range(64):
            file, rank = (square & 7) + file_step, (square >> 3) + rank_step
            ray = 0
            while 0 <= file < 8 and 0 <= rank < 8:
                ray |= 1 << (rank * 8 + file)
                file, rank = file + file_step, rank + rank_step
            rays[direction, square] = ray
    return rays


RAYS = _rays()

# Castling rights a move keeps when it starts or ends on each square: moving a king or a rook, or
# capturing a rook in its corner, takes the matching rights away.
CASTLING_KEEP = np.full(64, 15, dtype=np.int64)
CASTLING_KEEP[chess.E1] = 15 & ~(WHITE_KINGSIDE | WHITE_QUEENSIDE)
CASTLING_KEEP[chess.H1] = 15 & ~WHITE_KINGSIDE
CASTLING_KEEP[chess.A1] = 15 & ~WHITE_QUEENSIDE
CASTLING_KEEP[chess.E8] = 15 & ~(BLACK_KINGSIDE | BLACK_QUEENSIDE)
CASTLING_KEEP[chess.H8] = 15 & ~BLACK_KINGSIDE
CASTLING_KEEP[chess.A8] = 15 & ~BLACK_QUEENSIDE

# Zobrist keys: a position's key is the XOR of one random number per feature it has, so a move
# updates it by XOR-ing out what changed. A fixed seed keeps keys the same from run to run.
_random = np.random.default_rng(20260917)
ZOBRIST_PIECES = _random.integers(0, 2**64, size=(12, 64), dtype=np.uint64, endpoint=False)
ZOBRIST_CASTLING = _random.integers(0, 2**64, size=16, dtype=np.uint64, endpoint=False)
ZOBRIST_EN_PASSANT = _random.integers(0, 2**64, size=8, dtype=np.uint64, endpoint=False)
ZOBRIST_SIDE = np.uint64(_random.integers(0, 2**64, dtype=np.uint64, endpoint=False))

# Lowest set bit to square number by de Bruijn multiplication: numba has no bit-scan builtin.
DEBRUIJN = np.uint64(0x03F7_9D71_B4CB_0A89)
SHIFT_58 = np.uint64(58)
_bit_to_square = np.zeros(64, dtype=np.int64)
for _square in range(64):
    _bit_to_square[(((1 << _square) * 0x03F7_9D71_B4CB_0A89) & 0xFFFF_FFFF_FFFF_FFFF) >> 58] = (
        _square
    )
BIT_TO_SQUARE = _bit_to_square

M1 = np.uint64(0x5555_5555_5555_5555)
M2 = np.uint64(0x3333_3333_3333_3333)
M4 = np.uint64(0x0F0F_0F0F_0F0F_0F0F)
H01 = np.uint64(0x0101_0101_0101_0101)
S1, S2, S4, S8, S16, S32, S56 = (np.uint64(shift) for shift in (1, 2, 4, 8, 16, 32, 56))


# --- bits ----------------------------------------------------------------------------------


@njit(int64(uint64), cache=CACHE)
def lsb(bb):  # type: ignore[no-untyped-def]
    """Square of the lowest set bit. bb must not be empty."""
    return BIT_TO_SQUARE[((bb & (~bb + U1)) * DEBRUIJN) >> SHIFT_58]


@njit(int64(uint64), cache=CACHE)
def msb(bb):  # type: ignore[no-untyped-def]
    """Square of the highest set bit: smear it downwards, then keep only that bit."""
    bb |= bb >> S1
    bb |= bb >> S2
    bb |= bb >> S4
    bb |= bb >> S8
    bb |= bb >> S16
    bb |= bb >> S32
    return BIT_TO_SQUARE[((bb ^ (bb >> S1)) * DEBRUIJN) >> SHIFT_58]


@njit(int64(uint64), cache=CACHE)
def popcount(bb):  # type: ignore[no-untyped-def]
    bb = bb - ((bb >> S1) & M1)
    bb = (bb & M2) + ((bb >> S2) & M2)
    bb = (bb + (bb >> S4)) & M4
    return np.int64((bb * H01) >> S56)


# --- the position --------------------------------------------------------------------------


class PositionType(types.StructRef):
    def preprocess_fields(self, fields):  # type: ignore[no-untyped-def]
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class Position(structref.StructRefProxy):
    """A position and the undo stack of the moves made from it.

    A numba structref: a mutable struct that compiled functions share by pointer, so a call
    costs a few nanoseconds however many fields it has. A named tuple of arrays was copied
    field by field into every call, 56 ns each, and made the search 28% slower. A numba class
    is shared by pointer too, but numba cannot cache functions that take one (see CACHE).
    """

    @property
    def key(self) -> int:
        """Fields are read in compiled code; this one is also wanted from Python."""
        return int(_key_of(self))


POSITION_FIELDS = (
    ("pieces", uint64[::1]),  # 12 bitboards, one per piece
    ("colours", uint64[::1]),  # all White pieces, all Black pieces
    ("mailbox", int8[::1]),  # the piece on each square, or EMPTY
    ("side", int64),
    ("castling", int64),
    ("ep", int64),  # en passant target square, or -1
    ("halfmove", int64),
    ("key", uint64),
    ("ep_key", uint64),  # the en passant part of key, so a move can take it out again
    ("ply", int64),  # moves made since the search started, and the next free undo slot
    ("undo_moves", int32[::1]),
    ("undo_captured", int8[::1]),
    ("undo_castling", int8[::1]),
    ("undo_ep", int8[::1]),
    ("undo_halfmove", int16[::1]),
    ("undo_keys", uint64[::1]),
    ("undo_ep_keys", uint64[::1]),
    ("undo_piece", int8[::1]),  # the piece each move moved, for updating the accumulators
    # The network's accumulators for the position at each ply, White's view then Black's, and
    # whether each is up to date. evaluation.py fills them in only when a position is evaluated.
    ("accumulators", float32[:, :, ::1]),
    ("psqt", float32[:, :, ::1]),
    ("accumulator_ok", int8[:, ::1]),
)
structref.register(PositionType)
structref.define_proxy(Position, PositionType, [name for name, _ in POSITION_FIELDS])
POSITION = PositionType(POSITION_FIELDS)
MOVES = int32[::1]


def new_position() -> Position:
    values = {
        "pieces": np.zeros(12, dtype=np.uint64),
        "colours": np.zeros(2, dtype=np.uint64),
        "mailbox": np.full(64, EMPTY, dtype=np.int8),
        "side": np.int64(WHITE),
        "castling": np.int64(0),
        "ep": np.int64(-1),
        "halfmove": np.int64(0),
        "key": U0,
        "ep_key": U0,
        "ply": np.int64(0),
        "undo_moves": np.zeros(MAX_PLY, dtype=np.int32),
        "undo_captured": np.zeros(MAX_PLY, dtype=np.int8),
        "undo_castling": np.zeros(MAX_PLY, dtype=np.int8),
        "undo_ep": np.zeros(MAX_PLY, dtype=np.int8),
        "undo_halfmove": np.zeros(MAX_PLY, dtype=np.int16),
        "undo_keys": np.zeros(MAX_PLY, dtype=np.uint64),
        "undo_ep_keys": np.zeros(MAX_PLY, dtype=np.uint64),
        "undo_piece": np.zeros(MAX_PLY, dtype=np.int8),
        "accumulators": np.zeros((MAX_PLY + 1, 2, ACCUMULATOR_WIDTH), dtype=np.float32),
        "psqt": np.zeros((MAX_PLY + 1, 2, PSQT_BUCKETS), dtype=np.float32),
        "accumulator_ok": np.zeros((MAX_PLY + 1, 2), dtype=np.int8),
    }
    return Position(*(values[name] for name, _ in POSITION_FIELDS))


@njit(uint64(POSITION), cache=CACHE)
def _key_of(pos):  # type: ignore[no-untyped-def]
    return pos.key


# Every compiled function is pinned to POSITION. A Position built with a field of another type
# would quietly compile a second copy of everything, so check it once here.
assert numba.typeof(new_position()) == POSITION


@njit(types.void(POSITION, int64, int64), cache=CACHE)
def _place(pos, square, piece):  # type: ignore[no-untyped-def]
    bit = SQUARE_BB[square]
    pos.pieces[piece] |= bit
    pos.colours[piece // 6] |= bit
    pos.mailbox[square] = piece
    pos.key ^= ZOBRIST_PIECES[piece, square]


@njit(types.void(POSITION, int64, int64), cache=CACHE)
def _remove(pos, square, piece):  # type: ignore[no-untyped-def]
    bit = SQUARE_BB[square]
    pos.pieces[piece] ^= bit
    pos.colours[piece // 6] ^= bit
    pos.mailbox[square] = EMPTY
    pos.key ^= ZOBRIST_PIECES[piece, square]


@njit(uint64(POSITION), cache=CACHE)
def _en_passant_key(pos):  # type: ignore[no-untyped-def]
    """The en passant square counts towards the key only if a pawn could capture onto it."""
    if pos.ep < 0:
        return U0
    capturers = PAWN_ATTACKS[1 - pos.side, pos.ep] & pos.pieces[pos.side * 6 + PAWN]
    return ZOBRIST_EN_PASSANT[pos.ep & 7] if capturers else U0


@njit(types.void(POSITION, int8[::1], int64, int64, int64, int64), cache=CACHE)
def set_position(pos, mailbox, side, castling, ep, halfmove):  # type: ignore[no-untyped-def]
    """Load a position from a 64-square mailbox, and start a new search from it."""
    pos.pieces[:] = U0
    pos.colours[:] = U0
    pos.mailbox[:] = EMPTY
    pos.key = U0
    for square in range(64):
        if mailbox[square] != EMPTY:
            _place(pos, square, mailbox[square])
    pos.side = side
    pos.castling = castling
    pos.ep = ep
    pos.halfmove = halfmove
    pos.ply = 0
    pos.accumulator_ok[0, :] = 0
    pos.key ^= ZOBRIST_CASTLING[castling]
    if side == BLACK:
        pos.key ^= ZOBRIST_SIDE
    pos.ep_key = _en_passant_key(pos)
    pos.key ^= pos.ep_key


def load(pos: Position, board: chess.Board) -> None:
    """Copy a python-chess board into pos."""
    mailbox = np.full(64, EMPTY, dtype=np.int8)
    for square, piece in board.piece_map().items():
        colour = WHITE if piece.color == chess.WHITE else BLACK
        mailbox[square] = colour * 6 + piece.piece_type - 1
    rights = board.clean_castling_rights()
    castling = (
        (WHITE_KINGSIDE if rights & chess.BB_H1 else 0)
        | (WHITE_QUEENSIDE if rights & chess.BB_A1 else 0)
        | (BLACK_KINGSIDE if rights & chess.BB_H8 else 0)
        | (BLACK_QUEENSIDE if rights & chess.BB_A8 else 0)
    )
    side = WHITE if board.turn == chess.WHITE else BLACK
    ep = -1 if board.ep_square is None else board.ep_square
    set_position(pos, mailbox, side, castling, ep, min(board.halfmove_clock, 30_000))


@njit(uint64(POSITION), cache=CACHE)
def compute_key(pos):  # type: ignore[no-untyped-def]
    """The key from scratch, to check the incremental one against."""
    key = U0
    for square in range(64):
        if pos.mailbox[square] != EMPTY:
            key ^= ZOBRIST_PIECES[pos.mailbox[square], square]
    key ^= ZOBRIST_CASTLING[pos.castling]
    if pos.side == BLACK:
        key ^= ZOBRIST_SIDE
    return key ^ _en_passant_key(pos)


# --- moves ---------------------------------------------------------------------------------


@njit(int32(int64, int64, int64, int64), cache=CACHE)
def encode(from_square, to_square, promotion, flag):  # type: ignore[no-untyped-def]
    return np.int32(from_square | (to_square << 6) | (promotion << 12) | (flag << 15))


@njit(boolean(POSITION, int64), cache=CACHE)
def is_capture(pos, move):  # type: ignore[no-untyped-def]
    return pos.mailbox[(move >> 6) & 63] != EMPTY or ((move >> 15) & 3) == EN_PASSANT


def to_uci(move: int) -> str:
    move = int(move)
    uci = chess.SQUARE_NAMES[move & 63] + chess.SQUARE_NAMES[(move >> 6) & 63]
    promotion = (move >> 12) & 7
    return uci + ("", "n", "b", "r", "q")[promotion]


# --- attacks -------------------------------------------------------------------------------


@njit(uint64(int64, uint64, int64), cache=CACHE)
def _ray(square, occupied, direction):  # type: ignore[no-untyped-def]
    attacks = RAYS[direction, square]
    blockers = attacks & occupied
    if blockers:
        # Everything beyond the nearest blocker is hidden behind it.
        blocker = lsb(blockers) if direction < WEST else msb(blockers)
        attacks ^= RAYS[direction, blocker]
    return attacks


@njit(uint64(int64, uint64), cache=CACHE)
def bishop_attacks(square, occupied):  # type: ignore[no-untyped-def]
    return (
        _ray(square, occupied, NORTH_EAST)
        | _ray(square, occupied, NORTH_WEST)
        | _ray(square, occupied, SOUTH_EAST)
        | _ray(square, occupied, SOUTH_WEST)
    )


@njit(uint64(int64, uint64), cache=CACHE)
def rook_attacks(square, occupied):  # type: ignore[no-untyped-def]
    return (
        _ray(square, occupied, NORTH)
        | _ray(square, occupied, EAST)
        | _ray(square, occupied, SOUTH)
        | _ray(square, occupied, WEST)
    )


@njit(uint64(POSITION, int64, int64, uint64), cache=CACHE)
def attackers(pos, square, colour, occupied):  # type: ignore[no-untyped-def]
    """Pieces of colour that attack square, with sliders blocked by occupied.

    Like python-chess's attackers_mask, the result is not masked with occupied: a caller that
    removes pieces from occupied for an exchange masks them out itself.
    """
    base = colour * 6
    pieces = pos.pieces
    found = PAWN_ATTACKS[1 - colour, square] & pieces[base + PAWN]
    found |= KNIGHT_ATTACKS[square] & pieces[base + KNIGHT]
    found |= KING_ATTACKS[square] & pieces[base + KING]
    diagonal = pieces[base + BISHOP] | pieces[base + QUEEN]
    if diagonal:
        found |= bishop_attacks(square, occupied) & diagonal
    straight = pieces[base + ROOK] | pieces[base + QUEEN]
    if straight:
        found |= rook_attacks(square, occupied) & straight
    return found


@njit(boolean(POSITION, int64, int64), cache=CACHE)
def is_attacked(pos, square, by):  # type: ignore[no-untyped-def]
    occupied = pos.colours[WHITE] | pos.colours[BLACK]
    return attackers(pos, square, by, occupied) != U0


@njit(boolean(POSITION), cache=CACHE)
def in_check(pos):  # type: ignore[no-untyped-def]
    king = pos.pieces[pos.side * 6 + KING]
    return is_attacked(pos, lsb(king), 1 - pos.side)


# --- generation ----------------------------------------------------------------------------


@njit(int64(MOVES, int64, int64, int64, int64, int64), cache=CACHE)
def _add(moves, count, from_square, to_square, promotion, flag):  # type: ignore[no-untyped-def]
    moves[count] = encode(from_square, to_square, promotion, flag)
    return count + 1


@njit(int64(MOVES, int64, int64, uint64), cache=CACHE)
def _add_targets(moves, count, from_square, targets):  # type: ignore[no-untyped-def]
    while targets:
        count = _add(moves, count, from_square, lsb(targets), 0, NORMAL)
        targets &= targets - U1
    return count


@njit(boolean(POSITION, int64, uint64, int64, int64, int64), cache=CACHE)
def _can_castle(pos, right, between, king, crossed, landing):  # type: ignore[no-untyped-def]
    """The squares between king and rook are empty, and the king does not start in, pass
    through or land on an attacked square.

    Written as early returns: one long `and` chain with these calls in it took numba about
    five seconds to compile per chain.
    """
    if (pos.castling & right) == 0:
        return False
    if ((pos.colours[WHITE] | pos.colours[BLACK]) & between) != U0:
        return False
    them = 1 - pos.side
    if is_attacked(pos, king, them) or is_attacked(pos, crossed, them):
        return False
    return not is_attacked(pos, landing, them)


@njit(int64(POSITION, MOVES, boolean), cache=CACHE)
def generate(pos, moves, quiescence):  # type: ignore[no-untyped-def]
    """Write pseudo-legal moves into moves and return how many.

    With quiescence set, only what the quiescence search looks at: captures, including en
    passant and every capturing promotion, and pushes that promote to a queen.
    """
    us = pos.side
    own = pos.colours[us]
    enemy = pos.colours[1 - us]
    occupied = own | enemy
    targets = enemy if quiescence else ~own
    base = us * 6
    count = 0

    forward = 8 if us == WHITE else -8
    start_rank = RANK_2 if us == WHITE else RANK_7
    last_rank = RANK_8 if us == WHITE else RANK_1
    pawns = pos.pieces[base + PAWN]
    while pawns:
        from_square = lsb(pawns)
        pawns &= pawns - U1
        ahead = from_square + forward
        if pos.mailbox[ahead] == EMPTY:
            if (SQUARE_BB[ahead] & last_rank) != U0:
                count = _add(moves, count, from_square, ahead, QUEEN, NORMAL)
                if not quiescence:
                    for promotion in (ROOK, BISHOP, KNIGHT):
                        count = _add(moves, count, from_square, ahead, promotion, NORMAL)
            elif not quiescence:
                count = _add(moves, count, from_square, ahead, 0, NORMAL)
                on_start = (SQUARE_BB[from_square] & start_rank) != U0
                if on_start and pos.mailbox[ahead + forward] == EMPTY:
                    count = _add(moves, count, from_square, ahead + forward, 0, DOUBLE_PUSH)
        captures = PAWN_ATTACKS[us, from_square] & enemy
        while captures:
            to_square = lsb(captures)
            captures &= captures - U1
            if (SQUARE_BB[to_square] & last_rank) != U0:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    count = _add(moves, count, from_square, to_square, promotion, NORMAL)
            else:
                count = _add(moves, count, from_square, to_square, 0, NORMAL)
        if pos.ep >= 0 and (PAWN_ATTACKS[us, from_square] & SQUARE_BB[pos.ep]) != U0:
            count = _add(moves, count, from_square, pos.ep, 0, EN_PASSANT)

    knights = pos.pieces[base + KNIGHT]
    while knights:
        from_square = lsb(knights)
        knights &= knights - U1
        count = _add_targets(moves, count, from_square, KNIGHT_ATTACKS[from_square] & targets)

    diagonal = pos.pieces[base + BISHOP] | pos.pieces[base + QUEEN]
    while diagonal:
        from_square = lsb(diagonal)
        diagonal &= diagonal - U1
        attacks = bishop_attacks(from_square, occupied)
        count = _add_targets(moves, count, from_square, attacks & targets)

    straight = pos.pieces[base + ROOK] | pos.pieces[base + QUEEN]
    while straight:
        from_square = lsb(straight)
        straight &= straight - U1
        attacks = rook_attacks(from_square, occupied)
        count = _add_targets(moves, count, from_square, attacks & targets)

    king = lsb(pos.pieces[base + KING])
    count = _add_targets(moves, count, king, KING_ATTACKS[king] & targets)

    # Castling: the rights guarantee king and rook are home.
    if not quiescence and pos.castling != 0:
        if us == WHITE:
            if _can_castle(pos, WHITE_KINGSIDE, SQUARE_BB[5] | SQUARE_BB[6], 4, 5, 6):
                count = _add(moves, count, 4, 6, 0, CASTLING)
            between = SQUARE_BB[1] | SQUARE_BB[2] | SQUARE_BB[3]
            if _can_castle(pos, WHITE_QUEENSIDE, between, 4, 3, 2):
                count = _add(moves, count, 4, 2, 0, CASTLING)
        else:
            if _can_castle(pos, BLACK_KINGSIDE, SQUARE_BB[61] | SQUARE_BB[62], 60, 61, 62):
                count = _add(moves, count, 60, 62, 0, CASTLING)
            between = SQUARE_BB[57] | SQUARE_BB[58] | SQUARE_BB[59]
            if _can_castle(pos, BLACK_QUEENSIDE, between, 60, 59, 58):
                count = _add(moves, count, 60, 58, 0, CASTLING)
    return count


# --- make and unmake -----------------------------------------------------------------------


@njit(types.void(POSITION, int64, int64, int64), cache=CACHE)
def _save(pos, move, captured, piece):  # type: ignore[no-untyped-def]
    ply = pos.ply
    pos.undo_piece[ply] = piece
    pos.undo_moves[ply] = move
    pos.undo_captured[ply] = captured
    pos.undo_castling[ply] = pos.castling
    pos.undo_ep[ply] = pos.ep
    pos.undo_halfmove[ply] = min(pos.halfmove, 30_000)
    pos.undo_keys[ply] = pos.key
    pos.undo_ep_keys[ply] = pos.ep_key
    pos.ply = ply + 1
    pos.accumulator_ok[ply + 1, :] = 0


@njit(int64(POSITION), cache=CACHE)
def _restore(pos):  # type: ignore[no-untyped-def]
    pos.ply -= 1
    ply = pos.ply
    pos.side = 1 - pos.side
    pos.castling = pos.undo_castling[ply]
    pos.ep = pos.undo_ep[ply]
    pos.halfmove = pos.undo_halfmove[ply]
    pos.key = pos.undo_keys[ply]
    pos.ep_key = pos.undo_ep_keys[ply]
    return ply


@njit(types.void(POSITION), cache=CACHE)
def unmake_move(pos):  # type: ignore[no-untyped-def]
    ply = _restore(pos)
    move = pos.undo_moves[ply]
    captured = pos.undo_captured[ply]
    us = pos.side
    from_square = move & 63
    to_square = (move >> 6) & 63
    promotion = (move >> 12) & 7
    flag = (move >> 15) & 3
    key = pos.key  # _place and _remove update the key; the saved one is already right

    moved = pos.mailbox[to_square]
    _remove(pos, to_square, moved)
    _place(pos, from_square, us * 6 + PAWN if promotion else moved)
    if captured != EMPTY:
        if flag == EN_PASSANT:
            _place(pos, to_square - 8 if us == WHITE else to_square + 8, captured)
        else:
            _place(pos, to_square, captured)
    if flag == CASTLING:
        rook = us * 6 + ROOK
        if to_square == 6:
            _remove(pos, 5, rook)
            _place(pos, 7, rook)
        elif to_square == 2:
            _remove(pos, 3, rook)
            _place(pos, 0, rook)
        elif to_square == 62:
            _remove(pos, 61, rook)
            _place(pos, 63, rook)
        else:
            _remove(pos, 59, rook)
            _place(pos, 56, rook)
    pos.key = key


@njit(boolean(POSITION, int64), cache=CACHE)
def make_move(pos, move):  # type: ignore[no-untyped-def]
    """Play move. If it leaves the mover in check, take it back and return False."""
    us = pos.side
    them = 1 - us
    from_square = move & 63
    to_square = (move >> 6) & 63
    promotion = (move >> 12) & 7
    flag = (move >> 15) & 3
    piece = pos.mailbox[from_square]
    captured_square = to_square
    if flag == EN_PASSANT:
        captured_square = to_square - 8 if us == WHITE else to_square + 8
    captured = pos.mailbox[captured_square]
    _save(pos, move, captured, piece)

    pos.key ^= pos.ep_key ^ ZOBRIST_CASTLING[pos.castling]
    _remove(pos, from_square, piece)
    if captured != EMPTY:
        _remove(pos, captured_square, captured)
    _place(pos, to_square, us * 6 + promotion if promotion else piece)
    if flag == CASTLING:
        rook = us * 6 + ROOK
        if to_square == 6:
            _remove(pos, 7, rook)
            _place(pos, 5, rook)
        elif to_square == 2:
            _remove(pos, 0, rook)
            _place(pos, 3, rook)
        elif to_square == 62:
            _remove(pos, 63, rook)
            _place(pos, 61, rook)
        else:
            _remove(pos, 56, rook)
            _place(pos, 59, rook)

    pos.castling &= CASTLING_KEEP[from_square] & CASTLING_KEEP[to_square]
    pos.halfmove = 0 if piece % 6 == PAWN or captured != EMPTY else pos.halfmove + 1
    pos.ep = (from_square + to_square) // 2 if flag == DOUBLE_PUSH else -1
    pos.side = them
    pos.ep_key = _en_passant_key(pos)
    pos.key ^= ZOBRIST_CASTLING[pos.castling] ^ ZOBRIST_SIDE ^ pos.ep_key

    if is_attacked(pos, lsb(pos.pieces[us * 6 + KING]), them):
        unmake_move(pos)
        return False
    return True


@njit(types.void(POSITION), cache=CACHE)
def make_null(pos):  # type: ignore[no-untyped-def]
    """Pass the turn, for null-move pruning. Like python-chess, the halfmove clock still ticks."""
    _save(pos, NULL_MOVE, EMPTY, EMPTY)
    pos.key ^= pos.ep_key ^ ZOBRIST_SIDE
    pos.ep = -1
    pos.ep_key = U0
    pos.halfmove += 1
    pos.side = 1 - pos.side


@njit(types.void(POSITION), cache=CACHE)
def unmake_null(pos):  # type: ignore[no-untyped-def]
    _restore(pos)


# --- position queries ----------------------------------------------------------------------


@njit(boolean(POSITION), cache=CACHE)
def has_pieces(pos):  # type: ignore[no-untyped-def]
    """Whether the side to move has anything besides pawns and its king."""
    base = pos.side * 6
    return (
        pos.pieces[base + KNIGHT]
        | pos.pieces[base + BISHOP]
        | pos.pieces[base + ROOK]
        | pos.pieces[base + QUEEN]
    ) != U0


@njit(boolean(POSITION, int64), cache=CACHE)
def _insufficient_for(pos, colour):  # type: ignore[no-untyped-def]
    """python-chess's has_insufficient_material, for one side."""
    own = pos.colours[colour]
    other = pos.colours[1 - colour]
    pieces = pos.pieces
    pawns = pieces[PAWN] | pieces[6 + PAWN]
    knights = pieces[KNIGHT] | pieces[6 + KNIGHT]
    bishops = pieces[BISHOP] | pieces[6 + BISHOP]
    rooks = pieces[ROOK] | pieces[6 + ROOK]
    queens = pieces[QUEEN] | pieces[6 + QUEEN]
    kings = pieces[KING] | pieces[6 + KING]
    if own & (pawns | rooks | queens):
        return False
    if own & knights:
        return popcount(own) <= 2 and not other & ~kings & ~queens
    if own & bishops:
        same_colour = not bishops & DARK_SQUARES or not bishops & LIGHT_SQUARES
        return same_colour and not pawns and not knights
    return True


@njit(boolean(POSITION), cache=CACHE)
def insufficient_material(pos):  # type: ignore[no-untyped-def]
    return _insufficient_for(pos, WHITE) and _insufficient_for(pos, BLACK)


SEE_VALUES = np.array([100, 320, 330, 500, 900, 20_000], dtype=np.int64)


@njit(int64(POSITION, int64), cache=CACHE)
def see(pos, move):  # type: ignore[no-untyped-def]
    """Net centipawns from playing out the whole exchange on the target square.

    MVV-LVA alone ranks a queen taking a defended pawn as a fine capture. This plays the
    swap-off out, recomputing attackers against a shrinking occupancy so pieces behind the
    ones that just traded off join the exchange.
    """
    from_square = move & 63
    target = (move >> 6) & 63
    en_passant = ((move >> 15) & 3) == EN_PASSANT
    attacker = pos.mailbox[from_square]
    if attacker == EMPTY:
        return 0
    victim = pos.mailbox[target]
    if en_passant:
        captured_value = SEE_VALUES[PAWN]
    else:
        captured_value = SEE_VALUES[victim % 6] if victim != EMPTY else 0

    occupied = (pos.colours[WHITE] | pos.colours[BLACK]) & ~SQUARE_BB[from_square]
    if en_passant:
        behind = target - 8 if pos.side == WHITE else target + 8
        occupied &= ~SQUARE_BB[behind]

    gains = np.empty(34, dtype=np.int64)
    gains[0] = captured_value
    count = 1
    on_square = SEE_VALUES[attacker % 6]
    side = 1 - pos.side
    while count < 33:
        found = attackers(pos, target, side, occupied) & occupied
        if not found:
            break
        square = -1
        value = 0
        for piece_type in range(6):
            subset = found & pos.pieces[side * 6 + piece_type]
            if subset:
                square = lsb(subset)
                value = SEE_VALUES[piece_type]
                break
        if square < 0:
            break
        gains[count] = on_square - gains[count - 1]
        count += 1
        if max(-gains[count - 2], gains[count - 1]) < 0:
            break
        on_square = value
        occupied &= ~SQUARE_BB[square]
        side = 1 - side

    for index in range(count - 1, 0, -1):
        gains[index - 1] = -max(-gains[index - 1], gains[index])
    return gains[0]
