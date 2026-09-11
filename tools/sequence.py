"""Search every position one side faced in real games with one Searcher, as a game does.

    uv run python sequence.py versions/v12 [DEPTH] [PGN ...]

The fixed-depth bench gives each position a fresh Searcher, so it cannot see what the hash table
carries from move to move, or what happens when it fills. This keeps one Searcher per game and
searches our side's positions in order, reporting nodes and time per game and in total.
"""

import sys
import time
from pathlib import Path

import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
agent_dir = (ROOT / sys.argv[1]).resolve()
depth = int(sys.argv[2])
games = [Path(p) for p in sys.argv[3:]]
sys.path.insert(0, str(agent_dir))

import agent  # noqa: E402

agent.MAX_DEPTH = depth
total_nodes = 0
total_time = 0.0
total_positions = 0
for path in games:
    with path.open() as file:
        game = chess.pgn.read_game(file)
    assert game is not None
    ours = chess.WHITE if game.headers.get("White", "").startswith("v") else chess.BLACK
    searcher = agent.Searcher()
    board = game.board()
    nodes = positions = 0
    started = time.perf_counter()
    largest = 0
    for move in game.mainline_moves():
        if board.turn == ours:
            searcher.choose(board.copy(), 1e12, 1e12)
            nodes += searcher.nodes
            positions += 1
            largest = max(largest, len(searcher.table))
        board.push(move)
    elapsed = time.perf_counter() - started
    total_nodes += nodes
    total_time += elapsed
    total_positions += positions
    print(
        f"{path.name}: {positions} positions, {nodes:,} nodes, {elapsed:.1f} s, "
        f"largest table {largest:,}",
        flush=True,
    )

print(
    f"\n{sys.argv[1]} depth {depth}: {total_positions} positions, {total_nodes:,} nodes, "
    f"{total_time:.1f} s"
)
