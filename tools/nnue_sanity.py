"""Is a trained network sane, and is its centipawn scale right for the search's margins?

    uv run python tools/nnue_sanity.py <nnue.pt>

Prints the network and the classical evaluation side by side on positions with an obvious
answer, then fits classical ~ k * network over positions from real openings. The search's
margins were set against the classical scale, so k far from 1 means TO_CENTIPAWNS needs k times
its current value. The network is read with v6's evaluator, whose TO_CENTIPAWNS is 100 / 328.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "versions" / "v5"))
import agent as v5  # noqa: E402  (the hand-written evaluation)
from harness.rules import OPENINGS  # noqa: E402

sys.path.insert(0, str(REPO / "versions" / "v6"))
import nnue_eval  # noqa: E402

assert nnue_eval.load(Path(sys.argv[1])), "no weights file"


def without(fen: str, square: str) -> str:
    board = chess.Board(fen)
    board.remove_piece_at(chess.parse_square(square))
    return board.fen()


START = chess.STARTING_FEN
cases = [
    ("start position", START),
    ("start, Black to move", START.replace(" w ", " b ")),
    ("White up a knight (b8 gone)", without(START, "b8")),
    ("White up a rook (a8 gone)", without(START, "a8")),
    ("White up a queen (d8 gone)", without(START, "d8")),
    ("Black up a queen (d1 gone)", without(START, "d1")),
    ("White up a queen, Black to move", without(START, "d8").replace(" w ", " b ")),
    ("KQ v K", "8/8/8/4k3/8/8/8/3QK3 w - - 0 1"),
    ("KR v K, Black to move", "8/8/8/4k3/8/8/8/3RK3 b - - 0 1"),
    ("K+P v K, pawn on 7th", "8/4P3/8/8/8/2k5/8/4K3 w - - 0 1"),
]
print(f"{'position':34s} {'network':>8s} {'classical':>9s}   (centipawns for the side to move)")
for label, fen in cases:
    board = chess.Board(fen)
    print(f"{label:34s} {nnue_eval.evaluate(board):+8d} {v5.evaluate(board):+9d}")

rng = random.Random(5)
net, classical = [], []
for _, fen in OPENINGS:
    for _ in range(60):
        board = chess.Board(fen)
        for _ in range(rng.randrange(0, 30)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_check():
            continue
        net.append(nnue_eval.evaluate(board))
        classical.append(v5.evaluate(board))
a, b = np.array(net, dtype=float), np.array(classical, dtype=float)
keep = (np.abs(b) < 1500) & (np.abs(a) < 1500)
k = float((a[keep] * b[keep]).sum() / (a[keep] ** 2).sum())
r = float(np.corrcoef(a[keep], b[keep])[0, 1])
print(
    f"\n{keep.sum()} positions from the openings: corr(network, classical) = {r:.2f}, "
    f"classical ~ {k:.2f} x network"
)
print(
    f"current TO_CENTIPAWNS {nnue_eval.TO_CENTIPAWNS:.4f}; matching the classical scale would be "
    f"{nnue_eval.TO_CENTIPAWNS * k:.4f}"
)
