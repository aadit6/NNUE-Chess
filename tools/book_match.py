"""Play two agents on only the sample openings the book covers, using the harness's own referee.

uv run python book_match.py AGENT OPPONENT [GAMES]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.arena import FAST_BASE_MS, FAST_INCREMENT_MS, _report  # noqa: E402
from harness.referee import play_match  # noqa: E402
from harness.rules import OPENINGS, PLY_CAP  # noqa: E402
from harness.sandbox import local  # noqa: E402

COVERED = ("French Winawer", "Grunfeld Defence", "Sicilian Sveshnikov")
PGN_DIR = Path(__file__).parent / "games"

agent, opponent = (ROOT / sys.argv[1]).resolve(), (ROOT / sys.argv[2]).resolve()
games = int(sys.argv[3]) if len(sys.argv) > 3 else 6
openings = [(name, fen) for name, fen in OPENINGS if name in COVERED]
PGN_DIR.mkdir(exist_ok=True)

wins = draws = losses = 0
for index in range(games):
    name, fen = openings[(index // 2) % len(openings)]
    plays_white = index % 2 == 0
    white, black = (agent, opponent) if plays_white else (opponent, agent)
    outcome = play_match(
        local(white, index),
        local(black, index),
        FAST_BASE_MS,
        FAST_INCREMENT_MS,
        ply_cap=PLY_CAP,
        start_fen=fen,
    )
    (PGN_DIR / f"game-{index + 1:03d}.pgn").write_text(outcome.pgn + "\n")
    colour = "white" if plays_white else "black"
    print(
        f"Game {index + 1}/{games}, {name} as {colour}, {outcome.result} by {outcome.termination}",
        flush=True,
    )
    if outcome.result == "draw":
        draws += 1
    elif outcome.result in ("white", "black"):
        if (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1

print(f"\n{sys.argv[1]} vs {sys.argv[2]} on {', '.join(COVERED)}")
_report(wins, draws, losses)
