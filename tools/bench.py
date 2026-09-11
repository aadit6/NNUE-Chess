"""Fixed-depth search benchmark: nodes, time and chosen move per position, for one agent.

    uv run python bench.py versions/v10 [DEPTH]

Every position gets a fresh Searcher, so no hash table or history carries between positions.
Positions are the 24 tactics from analysis/tactics.json and the 8 sample openings. Node counts
are deterministic, so they can be compared at any time; wall time only means something on an
otherwise idle machine.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
agent_dir = (ROOT / sys.argv[1]).resolve()
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 7
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(agent_dir))

import chess  # noqa: E402

import agent  # noqa: E402
from harness.rules import OPENINGS  # noqa: E402

tactics = json.loads((Path(__file__).resolve().parent / "tactics.json").read_text())
positions = [(f"tactic {i + 1}", t["fen"], t["best"]) for i, t in enumerate(tactics)]
positions += [(name, fen, None) for name, fen in OPENINGS]

agent.MAX_DEPTH = depth
total_nodes = 0
total_time = 0.0
solved = 0
for label, fen, best in positions:
    searcher = agent.Searcher()
    board = chess.Board(fen)
    started = time.perf_counter()
    move = searcher.choose(board, 1e12, 1e12)
    elapsed = time.perf_counter() - started
    total_nodes += searcher.nodes
    total_time += elapsed
    hit = best is not None and move.uci() == best
    solved += hit
    mark = "" if best is None else (" solved" if hit else f" missed ({best})")
    print(
        f"{label:20s} {move.uci():6s} nodes {searcher.nodes:8d} time {elapsed:6.2f}s{mark}",
        flush=True,
    )

print(
    f"\n{sys.argv[1]} depth {depth}: {total_nodes:,} nodes, {total_time:.1f} s, "
    f"{total_nodes / total_time:,.0f} nodes/s, tactics solved {solved}/{len(tactics)}"
)
