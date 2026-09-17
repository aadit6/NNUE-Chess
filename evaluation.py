"""Evaluation for the compiled search: the network, and the hand-written fallback.

The network's forward pass lives in nnue_eval.py and is already compiled; this module feeds it the
position's bitboards and caches its scores. The hand-written evaluation is the one the engine
played before the network, ported unchanged, and runs only if the weights file is missing.
"""

import numba
import numpy as np
from numba import boolean, float32, float64, int32, int64, njit, types, uint64
from numba.experimental import structref

import nnue_eval
from position import (
    ACCUMULATOR_WIDTH,
    BISHOP,
    BLACK,
    CACHE,
    CASTLING,
    EMPTY,
    EN_PASSANT,
    KING,
    KNIGHT,
    NULL_MOVE,
    PAWN,
    POSITION,
    QUEEN,
    ROOK,
    U0,
    U1,
    WHITE,
    lsb,
    popcount,
)

# --- the hand-written evaluation -----------------------------------------------------------

PIECE_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
PHASE_MAX = 24

BISHOP_PAIR = 30
ROOK_OPEN_FILE = 20
ROOK_SEMI_OPEN_FILE = 10
DOUBLED_PAWN = 12
ISOLATED_PAWN = 12
PASSED_PAWN = np.array([0, 10, 15, 25, 45, 70, 100, 0], dtype=np.int64)
SHIELD_NEAR = 10
SHIELD_FAR = 5
TEMPO = 10

# Tables are written from White's point of view with rank 8 on the top row.
_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]  # fmt: skip
_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]  # fmt: skip
_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]  # fmt: skip
_ROOK = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]  # fmt: skip
_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]  # fmt: skip
_KING_MIDDLEGAME = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]  # fmt: skip
_KING_ENDGAME = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]  # fmt: skip


def _tables(table: list[int], value: int) -> np.ndarray:
    """Square-indexed (a1 = 0) tables, row 0 for White and row 1 for Black, value folded in."""
    white = [value + table[square ^ 56] for square in range(64)]
    black = [value + table[square] for square in range(64)]
    return np.array([white, black], dtype=np.int64)


# PIECE_SQUARE[piece type, colour, square], and the king's two phases.
PIECE_SQUARE = np.array(
    [
        _tables(_PAWN, 100),
        _tables(_KNIGHT, 320),
        _tables(_BISHOP, 330),
        _tables(_ROOK, 500),
        _tables(_QUEEN, 900),
    ]
)
KING_MIDDLEGAME = _tables(_KING_MIDDLEGAME, 0)
KING_ENDGAME = _tables(_KING_ENDGAME, 0)

FILES = np.array([0x0101_0101_0101_0101 << file for file in range(8)], dtype=np.uint64)
RANKS = np.array([0xFF << (8 * rank) for rank in range(8)], dtype=np.uint64)


def _masks() -> tuple[np.ndarray, ...]:
    files = [int(mask) for mask in FILES]
    ranks = [int(mask) for mask in RANKS]
    adjacent = [(files[f - 1] if f > 0 else 0) | (files[f + 1] if f < 7 else 0) for f in range(8)]
    passed = np.zeros((2, 64), dtype=np.uint64)
    near = np.zeros((2, 64), dtype=np.uint64)
    far = np.zeros((2, 64), dtype=np.uint64)
    for square in range(64):
        file, rank = square & 7, square >> 3
        span = files[file] | adjacent[file]
        passed[WHITE, square] = span & sum(ranks[rank + 1 :])
        passed[BLACK, square] = span & sum(ranks[:rank])
        near[WHITE, square] = span & (ranks[rank + 1] if rank < 7 else 0)
        far[WHITE, square] = span & (ranks[rank + 2] if rank < 6 else 0)
        near[BLACK, square] = span & (ranks[rank - 1] if rank > 0 else 0)
        far[BLACK, square] = span & (ranks[rank - 2] if rank > 1 else 0)
    return np.array(adjacent, dtype=np.uint64), passed, near, far


ADJACENT_FILES, PASSED_MASKS, SHIELD_NEAR_MASKS, SHIELD_FAR_MASKS = _masks()


@njit(int64(POSITION), cache=CACHE)
def classical_evaluate(pos):  # type: ignore[no-untyped-def]
    """Hand-written evaluation in centipawns for the side to move, used without weights."""
    pieces = pos.pieces
    white = pos.colours[WHITE]
    black = pos.colours[BLACK]
    pawns = pieces[PAWN] | pieces[6 + PAWN]
    knights = pieces[KNIGHT] | pieces[6 + KNIGHT]
    bishops = pieces[BISHOP] | pieces[6 + BISHOP]
    rooks = pieces[ROOK] | pieces[6 + ROOK]
    queens = pieces[QUEEN] | pieces[6 + QUEEN]
    white_pawns = pawns & white
    black_pawns = pawns & black

    phase = min(PHASE_MAX, popcount(knights | bishops) + 2 * popcount(rooks) + 4 * popcount(queens))
    score = 0
    middlegame = 0
    endgame = 0

    bb = white_pawns
    while bb:
        square = lsb(bb)
        bb &= bb - U1
        score += PIECE_SQUARE[PAWN, WHITE, square]
        if (black_pawns & PASSED_MASKS[WHITE, square]) == U0:
            bonus = PASSED_PAWN[square >> 3]
            middlegame += bonus // 2
            endgame += bonus
    bb = black_pawns
    while bb:
        square = lsb(bb)
        bb &= bb - U1
        score -= PIECE_SQUARE[PAWN, BLACK, square]
        if (white_pawns & PASSED_MASKS[BLACK, square]) == U0:
            bonus = PASSED_PAWN[7 - (square >> 3)]
            middlegame -= bonus // 2
            endgame -= bonus

    for piece_type in (KNIGHT, BISHOP, QUEEN):
        bb = pieces[piece_type]
        while bb:
            score += PIECE_SQUARE[piece_type, WHITE, lsb(bb)]
            bb &= bb - U1
        bb = pieces[6 + piece_type]
        while bb:
            score -= PIECE_SQUARE[piece_type, BLACK, lsb(bb)]
            bb &= bb - U1
    if popcount(pieces[BISHOP]) >= 2:
        score += BISHOP_PAIR
    if popcount(pieces[6 + BISHOP]) >= 2:
        score -= BISHOP_PAIR

    bb = pieces[ROOK]
    while bb:
        square = lsb(bb)
        bb &= bb - U1
        score += PIECE_SQUARE[ROOK, WHITE, square]
        file_mask = FILES[square & 7]
        if (pawns & file_mask) == U0:
            score += ROOK_OPEN_FILE
        elif (white_pawns & file_mask) == U0:
            score += ROOK_SEMI_OPEN_FILE
    bb = pieces[6 + ROOK]
    while bb:
        square = lsb(bb)
        bb &= bb - U1
        score -= PIECE_SQUARE[ROOK, BLACK, square]
        file_mask = FILES[square & 7]
        if (pawns & file_mask) == U0:
            score -= ROOK_OPEN_FILE
        elif (black_pawns & file_mask) == U0:
            score -= ROOK_SEMI_OPEN_FILE

    for file in range(8):
        on_file = popcount(white_pawns & FILES[file])
        if on_file:
            if on_file > 1:
                score -= DOUBLED_PAWN * (on_file - 1)
            if (white_pawns & ADJACENT_FILES[file]) == U0:
                score -= ISOLATED_PAWN * on_file
        on_file = popcount(black_pawns & FILES[file])
        if on_file:
            if on_file > 1:
                score += DOUBLED_PAWN * (on_file - 1)
            if (black_pawns & ADJACENT_FILES[file]) == U0:
                score += ISOLATED_PAWN * on_file

    white_king = lsb(pieces[KING])
    black_king = lsb(pieces[6 + KING])
    middlegame += KING_MIDDLEGAME[WHITE, white_king] - KING_MIDDLEGAME[BLACK, black_king]
    endgame += KING_ENDGAME[WHITE, white_king] - KING_ENDGAME[BLACK, black_king]
    middlegame += SHIELD_NEAR * popcount(white_pawns & SHIELD_NEAR_MASKS[WHITE, white_king])
    middlegame += SHIELD_FAR * popcount(white_pawns & SHIELD_FAR_MASKS[WHITE, white_king])
    middlegame -= SHIELD_NEAR * popcount(black_pawns & SHIELD_NEAR_MASKS[BLACK, black_king])
    middlegame -= SHIELD_FAR * popcount(black_pawns & SHIELD_FAR_MASKS[BLACK, black_king])

    total = score + (middlegame * phase + endgame * (PHASE_MAX - phase)) // PHASE_MAX
    return total + TEMPO if pos.side == WHITE else TEMPO - total


# --- the network ---------------------------------------------------------------------------

# A search reaches the same positions again through transpositions: a third to a half of the
# evaluations in one measured search were repeats. The cache is kept across moves within a game.
CACHE_SIZE = 1 << 19


class NetworkType(types.StructRef):
    def preprocess_fields(self, fields):  # type: ignore[no-untyped-def]
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class Network(structref.StructRefProxy):
    """The weights, passed to the compiled search as data.

    numba bakes a global array into the compiled code as a constant. The feature transformer is
    about 92 MB in float32, so it travels as a field instead.
    """

    @property
    def loaded(self) -> bool:
        return bool(_loaded(self))


NETWORK_FIELDS = (
    ("loaded", boolean),
    ("ft_weight", float32[:, ::1]),
    ("ft_psqt", float32[:, ::1]),
    ("dense", float32[::1]),
    ("tables", int64[::1]),
    ("columns", float32[:, :, ::1]),  # the first hidden layer's weights by input, see head()
    ("scale", float64),
    ("cache_keys", uint64[::1]),
    ("cache_scores", int32[::1]),
    ("cache_filled", boolean[::1]),
)
structref.register(NetworkType)
structref.define_proxy(Network, NetworkType, [name for name, _ in NETWORK_FIELDS])
NETWORK = NetworkType(NETWORK_FIELDS)


@njit(boolean(NETWORK), cache=CACHE)
def _loaded(net):  # type: ignore[no-untyped-def]
    return net.loaded


def _network(loaded: bool, parts: tuple[np.ndarray, ...], scale: float) -> Network:
    # Contiguous, aligned and writable, the layout NETWORK was typed with.
    ft_weight, ft_psqt, dense, tables = (
        np.require(part, requirements=("C", "A", "W")) for part in parts
    )
    values = {
        "loaded": bool(loaded),
        "ft_weight": ft_weight,
        "ft_psqt": ft_psqt,
        "dense": dense,
        "tables": tables,
        "columns": (
            nnue_eval.layer_one_columns(dense, ft_weight.shape[1])
            if loaded
            else np.zeros((1, 1, nnue_eval.L2), dtype=np.float32)
        ),
        "scale": float(scale),
        "cache_keys": np.zeros(CACHE_SIZE, dtype=np.uint64),
        "cache_scores": np.zeros(CACHE_SIZE, dtype=np.int32),
        "cache_filled": np.zeros(CACHE_SIZE, dtype=np.bool_),
    }
    network = Network(*(values[name] for name, _ in NETWORK_FIELDS))
    if loaded and ft_weight.shape[1] != ACCUMULATOR_WIDTH:
        raise ValueError(f"network width {ft_weight.shape[1]}, positions hold {ACCUMULATOR_WIDTH}")
    assert numba.typeof(network) == NETWORK
    return network


def load_network() -> Network:
    """The network if its weights ship beside this file, otherwise a Network that says not."""
    loaded = nnue_eval.load()
    parts = (nnue_eval.FT_WEIGHT, nnue_eval.FT_PSQT, nnue_eval.DENSE, nnue_eval.TABLES)
    return _network(loaded, parts, nnue_eval.SCALE)


@njit(types.void(POSITION, NETWORK, int64, int64, int64), cache=CACHE)
def _apply_move(pos, net, ply, perspective, king_square):  # type: ignore[no-untyped-def]
    """Turn the accumulator at ply into the one at ply + 1: the inputs the move there changed.

    Never called for a move of this side's own king, which changes every input it has.
    """
    accumulator = pos.accumulators[ply + 1, perspective]
    psqt = pos.psqt[ply + 1, perspective]
    accumulator[:] = pos.accumulators[ply, perspective]
    psqt[:] = pos.psqt[ply, perspective]
    move = pos.undo_moves[ply]
    if move == NULL_MOVE:
        return
    piece = np.int64(pos.undo_piece[ply])
    captured = np.int64(pos.undo_captured[ply])
    from_square = move & 63
    to_square = (move >> 6) & 63
    promotion = (move >> 12) & 7
    flag = (move >> 15) & 3
    colour = piece // 6
    weight, table, tables = net.ft_weight, net.ft_psqt, net.tables

    index = nnue_eval.feature_index(piece, from_square, king_square, perspective, tables)
    nnue_eval.remove_feature(accumulator, psqt, weight, table, index)
    arrived = colour * 6 + promotion if promotion else piece
    index = nnue_eval.feature_index(arrived, to_square, king_square, perspective, tables)
    nnue_eval.add_feature(accumulator, psqt, weight, table, index)
    if captured != EMPTY:
        captured_square = to_square
        if flag == EN_PASSANT:
            captured_square = to_square - 8 if colour == WHITE else to_square + 8
        index = nnue_eval.feature_index(captured, captured_square, king_square, perspective, tables)
        nnue_eval.remove_feature(accumulator, psqt, weight, table, index)
    if flag == CASTLING:
        rook = colour * 6 + ROOK
        if to_square == 6:
            rook_from, rook_to = 7, 5
        elif to_square == 2:
            rook_from, rook_to = 0, 3
        elif to_square == 62:
            rook_from, rook_to = 63, 61
        else:
            rook_from, rook_to = 56, 59
        index = nnue_eval.feature_index(rook, rook_from, king_square, perspective, tables)
        nnue_eval.remove_feature(accumulator, psqt, weight, table, index)
        index = nnue_eval.feature_index(rook, rook_to, king_square, perspective, tables)
        nnue_eval.add_feature(accumulator, psqt, weight, table, index)


@njit(types.void(POSITION, NETWORK, int64), cache=CACHE)
def update_accumulator(pos, net, perspective):  # type: ignore[no-untyped-def]
    """Bring one side's accumulator for the current ply up to date.

    Walks back to the last ply whose accumulator is known, then replays the moves since. If this
    side's king moved in between, its every input changed, so it is rebuilt from the board.
    Accumulators are filled in only when a position is evaluated, so a node cut off before its
    evaluation, or a move found illegal, costs nothing here.
    """
    ply = pos.ply
    if pos.accumulator_ok[ply, perspective]:
        return
    own_king = perspective * 6 + KING
    known = ply
    while known > 0 and not pos.accumulator_ok[known, perspective]:
        if pos.undo_piece[known - 1] == own_king:
            break
        known -= 1
    if not pos.accumulator_ok[known, perspective]:
        nnue_eval.refresh(
            pos.accumulators[ply, perspective],
            pos.psqt[ply, perspective],
            pos.pieces,
            perspective,
            net.ft_weight,
            net.ft_psqt,
            net.dense,
            net.tables,
        )
        pos.accumulator_ok[ply, perspective] = 1
        return
    king_square = lsb(pos.pieces[own_king])
    for earlier in range(known, ply):
        _apply_move(pos, net, earlier, perspective, king_square)
        pos.accumulator_ok[earlier + 1, perspective] = 1


@njit(float32(POSITION, NETWORK), cache=CACHE)
def network_raw(pos, net):  # type: ignore[no-untyped-def]
    """The network's output for the side to move, from the incrementally kept accumulators."""
    update_accumulator(pos, net, WHITE)
    update_accumulator(pos, net, BLACK)
    bucket = (popcount(pos.colours[WHITE] | pos.colours[BLACK]) - 1) // 4
    ply = pos.ply
    return nnue_eval.head(
        pos.accumulators[ply, WHITE],
        pos.accumulators[ply, BLACK],
        pos.psqt[ply, WHITE],
        pos.psqt[ply, BLACK],
        pos.side,
        bucket,
        net.dense,
        net.columns,
    )


@njit(int64(POSITION, NETWORK), cache=CACHE)
def network_evaluate(pos, net):  # type: ignore[no-untyped-def]
    slot = np.int64(pos.key & np.uint64(CACHE_SIZE - 1))
    if net.cache_filled[slot] and net.cache_keys[slot] == pos.key:
        return np.int64(net.cache_scores[slot])
    score = np.int64(np.float64(network_raw(pos, net)) * net.scale)  # truncates, like int()
    net.cache_keys[slot] = pos.key
    net.cache_scores[slot] = score
    net.cache_filled[slot] = True
    return score


@njit(int64(POSITION, NETWORK), cache=CACHE)
def evaluate(pos, net):  # type: ignore[no-untyped-def]
    """Centipawns for the side to move."""
    if net.loaded:
        return network_evaluate(pos, net)
    return np.int64(classical_evaluate(pos))


@njit(types.void(NETWORK), cache=CACHE)
def clear_cache(net):  # type: ignore[no-untyped-def]
    net.cache_filled[:] = False
