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
    BISHOP,
    BLACK,
    CACHE,
    KING,
    KNIGHT,
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
        "scale": float(scale),
        "cache_keys": np.zeros(CACHE_SIZE, dtype=np.uint64),
        "cache_scores": np.zeros(CACHE_SIZE, dtype=np.int32),
        "cache_filled": np.zeros(CACHE_SIZE, dtype=np.bool_),
    }
    network = Network(*(values[name] for name, _ in NETWORK_FIELDS))
    assert numba.typeof(network) == NETWORK
    return network


def load_network() -> Network:
    """The network if its weights ship beside this file, otherwise a Network that says not."""
    loaded = nnue_eval.load()
    parts = (nnue_eval.FT_WEIGHT, nnue_eval.FT_PSQT, nnue_eval.DENSE, nnue_eval.TABLES)
    return _network(loaded, parts, nnue_eval.SCALE)


@njit(int64(POSITION, NETWORK), cache=CACHE)
def network_evaluate(pos, net):  # type: ignore[no-untyped-def]
    slot = np.int64(pos.key & np.uint64(CACHE_SIZE - 1))
    if net.cache_filled[slot] and net.cache_keys[slot] == pos.key:
        return np.int64(net.cache_scores[slot])
    pieces = pos.pieces
    raw = nnue_eval.forward(
        pieces[PAWN] | pieces[6 + PAWN],
        pieces[KNIGHT] | pieces[6 + KNIGHT],
        pieces[BISHOP] | pieces[6 + BISHOP],
        pieces[ROOK] | pieces[6 + ROOK],
        pieces[QUEEN] | pieces[6 + QUEEN],
        pieces[KING] | pieces[6 + KING],
        pos.colours[WHITE],
        pos.colours[BLACK],
        pos.side,
        net.ft_weight,
        net.ft_psqt,
        net.dense,
        net.tables,
    )
    score = np.int64(np.float64(raw) * net.scale)  # truncates towards zero, like int()
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
