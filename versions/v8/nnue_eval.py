"""Neural evaluation: a HalfKAv2_hm network trained from scratch with nnue-pytorch.

Each side sees the board from its own king. The board is mirrored so that king stands on files
e-h, which leaves 32 king squares, and each of them has its own 704 piece-square inputs (both
kings share one plane): 22,528 inputs per side feeding 1024 accumulator units. The side to
move's accumulator comes first. Each is clipped to [0, 1) and its two halves are multiplied
together, giving 1024 values. One of eight small layer stacks, picked by the number of pieces
on the board, turns them into a score: 1024 -> 32 -> 32 -> 1, each hidden layer passing both
its square and itself through a clipped ReLU, plus a skip connection from the first layer and
a material-like PSQT term averaged over the two sides.

The accumulators are rebuilt from python-chess's bitboards on every call, inside one numba
function: python-chess makes and unmakes the moves, and the network only ever sees positions.
Every loop reads a one-dimensional row and sums into a float32 scalar. Written with flat
offsets or a float64 running sum, the same 1024 -> 32 layer measured six times slower,
because the compiler stops vectorising it.
"""

from pathlib import Path

import chess
import numpy as np
from numba import njit

WEIGHTS = Path(__file__).resolve().parent / "weights" / "nnue.pt"
L2 = 32
L3 = 32
PLANES_PER_BUCKET = 704
# The network scores in Stockfish's internal units. The search's pruning margins were set
# against the hand-written evaluation, pawn = 100. Fitted over 462 opening positions, the net
# after 60 epochs matched that scale at 0.352 (after one epoch it had been 0.304).
TO_CENTIPAWNS = 100.0 / 285.0

# Bucket of each king square once the board is mirrored so the king is on files e-h.
# fmt: off
KING_BUCKETS = [
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4,
    -1, -1, -1, -1, 3, 2, 1, 0,
]
# fmt: on

# Lowest set bit to square index by de Bruijn multiplication: numba has no bit-scan builtin.
_DEBRUIJN = 0x03F79D71B4CB0A89
_BIT_TO_SQUARE = [0] * 64
for _square in range(64):
    _BIT_TO_SQUARE[(((1 << _square) * _DEBRUIJN) & 0xFFFF_FFFF_FFFF_FFFF) >> 58] = _square
TABLES = np.array(_BIT_TO_SQUARE + KING_BUCKETS, dtype=np.int64)

DEBRUIJN = np.uint64(_DEBRUIJN)
ONE = np.uint64(1)
TWO = np.uint64(2)
FOUR = np.uint64(4)
SHIFT_56 = np.uint64(56)
SHIFT_58 = np.uint64(58)
M1 = np.uint64(0x5555_5555_5555_5555)
M2 = np.uint64(0x3333_3333_3333_3333)
M4 = np.uint64(0x0F0F_0F0F_0F0F_0F0F)
H01 = np.uint64(0x0101_0101_0101_0101)

SIGNATURE = (
    "float32(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, int64,"
    " float32[:, ::1], float32[:, ::1], float32[::1], int64[::1])"
)


@njit(SIGNATURE, fastmath=True, cache=False)
def forward(  # type: ignore[no-untyped-def]
    pawns, knights, bishops, rooks, queens, kings, white, black, stm, ft_weight, ft_psqt, dense,
    tables,
):
    """Score for the side to move, in network units. stm is 0 for White and 1 for Black.

    dense holds the accumulator bias, four activation constants, then one block per bucket:
    first layer weights and biases, second layer, output layer.
    """
    l1 = ft_weight.shape[1]
    half = l1 // 2
    zero = np.float32(0.0)

    occupied = white | black
    x = occupied - ((occupied >> ONE) & M1)
    x = (x & M2) + ((x >> TWO) & M2)
    x = (x + (x >> FOUR)) & M4
    bucket = (np.int64((x * H01) >> SHIFT_56) - 1) // 4

    bias = dense[:l1]
    max_ft = dense[l1]
    max_hidden = dense[l1 + 1]
    l0_correction = dense[l1 + 2]
    sqr_correction = dense[l1 + 3]

    boards = (pawns, knights, bishops, rooks, queens, kings)
    accumulator = np.empty((2, l1), dtype=np.float32)
    psqt = np.zeros(2, dtype=np.float32)
    for side in range(2):
        own = white if side == 0 else black
        king = kings & own
        king_square = tables[((king & (~king + ONE)) * DEBRUIJN) >> SHIFT_58]
        flip = (7 if (king_square & 7) < 4 else 0) ^ (56 if side == 1 else 0)
        base = tables[64 + (king_square ^ flip)] * PLANES_PER_BUCKET
        acc = accumulator[side]
        for i in range(l1):
            acc[i] = bias[i]
        for piece in range(6):
            for colour in range(2):
                plane = piece * 2 + (1 if colour != side else 0)
                if plane > 10:  # both kings share the last plane
                    plane = 10
                bb = boards[piece] & (white if colour == 0 else black)
                while bb != 0:
                    square = tables[((bb & (~bb + ONE)) * DEBRUIJN) >> SHIFT_58]
                    index = base + plane * 64 + (square ^ flip)
                    row = ft_weight[index]
                    for i in range(l1):
                        acc[i] += row[i]
                    psqt[side] += ft_psqt[index, bucket]
                    bb &= bb - ONE

    mine = accumulator[stm]
    theirs = accumulator[1 - stm]
    l0 = np.empty(l1, dtype=np.float32)
    for i in range(half):
        l0[i] = (
            min(max(mine[i], zero), max_ft) * min(max(mine[half + i], zero), max_ft) * l0_correction
        )
        l0[half + i] = (
            min(max(theirs[i], zero), max_ft)
            * min(max(theirs[half + i], zero), max_ft)
            * l0_correction
        )

    block = L2 * l1 + L2 + L3 * 2 * L2 + L3 + 2 * L2 + 2 * L3 + 1
    offset = l1 + 4 + bucket * block
    w1 = dense[offset : offset + L2 * l1]
    offset += L2 * l1
    b1 = dense[offset : offset + L2]
    offset += L2
    w2 = dense[offset : offset + L3 * 2 * L2]
    offset += L3 * 2 * L2
    b2 = dense[offset : offset + L3]
    offset += L3
    w_out = dense[offset : offset + 2 * L2 + 2 * L3]
    b_out = dense[offset + 2 * L2 + 2 * L3]

    h1 = np.empty(2 * L2, dtype=np.float32)
    skip = zero
    for j in range(L2):
        weights = w1[j * l1 : (j + 1) * l1]
        z = zero
        for i in range(l1):
            z += weights[i] * l0[i]
        z += b1[j]
        if j == L2 - 2:
            skip += z
        elif j == L2 - 1:
            skip -= z
        h1[j] = min(max(z * z * sqr_correction, zero), max_hidden)
        h1[L2 + j] = min(max(z, zero), max_hidden)

    h2 = np.empty(2 * L3, dtype=np.float32)
    for j in range(L3):
        weights = w2[j * 2 * L2 : (j + 1) * 2 * L2]
        z = zero
        for i in range(2 * L2):
            z += weights[i] * h1[i]
        z += b2[j]
        h2[j] = min(max(z * z * sqr_correction, zero), max_hidden)
        h2[L3 + j] = min(max(z, zero), max_hidden)

    out = zero
    for i in range(2 * L2):
        out += w_out[i] * h1[i]
    for i in range(2 * L3):
        out += w_out[2 * L2 + i] * h2[i]
    out += b_out + skip

    half_psqt = (psqt[0] - psqt[1]) * np.float32(0.5)
    return out + half_psqt if stm == 0 else out - half_psqt


FT_WEIGHT = np.zeros((1, 2), dtype=np.float32)
FT_PSQT = np.zeros((1, 8), dtype=np.float32)
DENSE = np.zeros(1, dtype=np.float32)
SCALE = 0.0


def load(path: Path = WEIGHTS) -> bool:
    """Read exported weights. Returns False when there is no weights file."""
    global FT_WEIGHT, FT_PSQT, DENSE, SCALE
    if not path.is_file():
        return False
    import torch

    tensors = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
    FT_WEIGHT = np.ascontiguousarray(tensors["ft_weight"].float().numpy())
    FT_PSQT = np.ascontiguousarray(tensors["ft_psqt"].float().numpy())
    parts = [
        tensors["ft_bias"].numpy(),
        np.array(
            [tensors["max_ft_activation"], tensors["max_hidden_activation"],
             tensors["l0_correction"], tensors["sqr_correction"]],
            dtype=np.float32,
        ),
    ]
    for bucket in range(tensors["l1_weight"].shape[0]):
        parts += [
            tensors["l1_weight"][bucket].numpy().ravel(),
            tensors["l1_bias"][bucket].numpy(),
            tensors["l2_weight"][bucket].numpy().ravel(),
            tensors["l2_bias"][bucket].numpy(),
            tensors["out_weight"][bucket].numpy().ravel(),
            tensors["out_bias"][bucket].numpy(),
        ]
    DENSE = np.ascontiguousarray(np.concatenate(parts).astype(np.float32))
    SCALE = float(tensors["nnue2score"]) * TO_CENTIPAWNS
    raw(chess.Board())  # keep first-call overhead inside the import budget
    return True


def raw(board: chess.Board) -> float:
    occupied = board.occupied_co
    return float(forward(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        occupied[chess.WHITE], occupied[chess.BLACK], 0 if board.turn == chess.WHITE else 1,
        FT_WEIGHT, FT_PSQT, DENSE, TABLES,
    ))


def evaluate(board: chess.Board) -> int:
    """Centipawns for the side to move."""
    return int(raw(board) * SCALE)
