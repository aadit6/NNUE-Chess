"""Evaluation benchmark: what one call of the network costs, each way it can be run.

    uv run python tools/bench_eval.py [REPEATS]

The engine can reach the same score by two routes, and this times both over the 24 tactics
positions, at the -O2 the agent sets:

- forward: rebuild both accumulators and read the first hidden layer as 32 dot products over all
  1024 inputs. This is what versions/v9b and versions/v13 play.
- refresh + head: rebuild both accumulators, then walk the nonzero inputs instead, adding each
  one's 32 weights. This is the arithmetic the build at the root ends at.
- head alone: the same head with the accumulators already up to date, which is what an
  incremental evaluation pays once its accumulators are kept move by move.

Times are the best of five passes, so a busy machine reads slow rather than wrong. Only the
ratios travel between machines.
"""

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("NUMBA_OPT", "2")  # agent.py's setting, see the note there

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import numpy as np  # noqa: E402

import nnue_eval  # noqa: E402

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PASSES = 5

if not nnue_eval.load():
    raise SystemExit(f"no weights at {nnue_eval.WEIGHTS}")
FT_WEIGHT, FT_PSQT = nnue_eval.FT_WEIGHT, nnue_eval.FT_PSQT
DENSE, TABLES = nnue_eval.DENSE, nnue_eval.TABLES
COLUMNS = nnue_eval.layer_one_columns(DENSE, FT_WEIGHT.shape[1])

tactics = json.loads((Path(__file__).resolve().parent / "tactics.json").read_text())
boards = [chess.Board(entry["fen"]) for entry in tactics]


def bitboards(board: chess.Board) -> np.ndarray:
    """The 12 piece bitboards refresh() reads, indexed colour * 6 + type."""
    planes = np.zeros(12, dtype=np.uint64)
    types = (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    sides = (board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK])
    for colour, occupied in enumerate(sides):
        for piece_type, plane in enumerate(types):
            planes[colour * 6 + piece_type] = np.uint64(plane & occupied)
    return planes


PIECES = [bitboards(board) for board in boards]
SIDES = [0 if board.turn == chess.WHITE else 1 for board in boards]
BUCKETS = [(chess.popcount(board.occupied) - 1) // 4 for board in boards]

accumulator = np.zeros((2, FT_WEIGHT.shape[1]), dtype=np.float32)
psqt = np.zeros((2, 8), dtype=np.float32)


def run_forward(repeats: int) -> float:
    total = 0.0
    for _ in range(repeats):
        for board, side in zip(boards, SIDES, strict=True):
            total += float(
                nnue_eval.forward(
                    board.pawns,
                    board.knights,
                    board.bishops,
                    board.rooks,
                    board.queens,
                    board.kings,
                    board.occupied_co[chess.WHITE],
                    board.occupied_co[chess.BLACK],
                    side,
                    FT_WEIGHT,
                    FT_PSQT,
                    DENSE,
                    TABLES,
                )
            )
    return total


def refresh_both(pieces: np.ndarray) -> None:
    for perspective in range(2):
        nnue_eval.refresh(
            accumulator[perspective],
            psqt[perspective],
            pieces,
            perspective,
            FT_WEIGHT,
            FT_PSQT,
            DENSE,
            TABLES,
        )


def read_head(side: int, bucket: int) -> float:
    return float(
        nnue_eval.head(
            accumulator[0], accumulator[1], psqt[0], psqt[1], side, bucket, DENSE, COLUMNS
        )
    )


def run_refresh_head(repeats: int) -> float:
    total = 0.0
    for _ in range(repeats):
        for pieces, side, bucket in zip(PIECES, SIDES, BUCKETS, strict=True):
            refresh_both(pieces)
            total += read_head(side, bucket)
    return total


def run_head(repeats: int) -> float:
    refresh_both(PIECES[0])
    total = 0.0
    for _ in range(repeats * len(boards)):
        total += read_head(SIDES[0], BUCKETS[0])
    return total


# The two routes have to agree before their times mean anything. The sums come out in a different
# order, so the last bits differ.
for board, pieces, side, bucket in zip(boards, PIECES, SIDES, BUCKETS, strict=True):
    refresh_both(pieces)
    sparse = read_head(side, bucket)
    dense = float(
        nnue_eval.forward(
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
            board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK],
            side,
            FT_WEIGHT,
            FT_PSQT,
            DENSE,
            TABLES,
        )
    )
    if abs(sparse - dense) > 1e-3:
        raise SystemExit(f"{board.fen()}: head {sparse}, forward {dense}")
print(f"the two routes agree on all {len(boards)} positions")


def _pass(run: Callable[[int], float]) -> float:
    started = time.perf_counter()
    run(REPEATS)
    return time.perf_counter() - started


calls = REPEATS * len(boards)
for label, run in (
    ("forward (rebuild, dense)", run_forward),
    ("refresh + head (rebuild, sparse)", run_refresh_head),
    ("head alone (accumulators kept)", run_head),
):
    run(2)  # compile, and warm the weights into cache
    best = min(_pass(run) for _ in range(PASSES))
    print(f"{label:34s} {best / calls * 1e6:6.1f} us a call, best of {PASSES} over {calls:,} calls")
