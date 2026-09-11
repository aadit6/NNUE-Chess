# ai-chessathon

Team **Claude's Gambit**'s entry for [AI Chessathon](https://aichessathon.com): a pure-Python
alpha-beta chess engine, plus the sparring partners and measurements used to decide what went
into it.

**Current build: v5.** It rates about 1700 against a Stockfish ladder at fast time control,
and every version since has been kept only when it beat the one before. Last updated
11 Sep 2026.

## The competition

The whole submission is `agent.py`, which exposes one function:

```python
def get_move(fen: str, time_left_ms: int) -> str: ...
```

The platform zips it, starts one process per game and plays it against other teams' agents in
hourly rated rounds. The conditions shape every design decision below:

- 120 s + 0.5 s per side on one core of an AMD EPYC, with 2 GB and no network.
- A fixed Python environment: python-chess, numpy, numba, torch and onnxruntime. No native
  binaries, so C extensions and Cython are out.
- No borrowed engines or networks, and no tables of engine evaluations to look up during play.
- Games start from curated opening positions rather than the standard start.

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes.

## The engine

### Search

| Technique | What it does here |
|---|---|
| Iterative deepening, aspiration windows | Searches depth 1, 2, 3... and keeps the last finished result. Windows start ±40 cp around the previous score. |
| Principal variation search | Fail-soft negamax. Only the first move gets a full window. |
| Transposition table | 1M entries keyed on the position, depth-preferred replacement, kept across moves in a game. |
| Move ordering | TT move, then captures by MVV-LVA with losing captures demoted by static exchange evaluation (SEE), then killers, then history. |
| Internal iterative deepening | From depth 5, a node with no TT move runs a shallow search first to find one. |
| Null-move pruning | Skips a turn. If the position still fails high, prunes it. |
| Reverse futility, futility, late move pruning | Prune nodes whose static eval is far outside the window, up to depths 6, 4 and 5. |
| Late move reductions | Quiet moves late in the ordering search shallower, by a log-based table. |
| Mate distance pruning, check extensions | Prefer the shortest mate, and search one ply deeper when in check. |
| Quiescence search | Captures only, with stand-pat, delta pruning and SEE pruning of losing captures. |

Time management gives each move `time_left / 34` plus 85% of the increment, which the engine
measures rather than trusts, and switches to `/ 45` under 25 seconds. A timeout raises out of
the search and returns the last completed depth.

### Evaluation

Material and piece-square tables tapered between middlegame and endgame by phase, plus:

- passed pawns by rank (10 to 100 cp), doubled and isolated pawns (−12 each)
- rooks on open (+20) and semi-open (+10) files
- the bishop pair (+30)
- a king pawn shield (+10 near, +5 far)
- tempo (+10)

## How strong it is

### Stockfish ladder

v1, the first complete engine, played 20 games against Stockfish 19 at each of nine
`UCI_LimitStrength` levels, at the harness's fast time control of 10 s + 0.1 s.

| Stockfish Elo | 1400 | 1500 | 1600 | 1700 | 1800 | 1900 | 2000 | 2100 | 2200 |
|---|---|---|---|---|---|---|---|---|---|
| Our score | 80% | 60% | 57.5% | 45% | 10% | 37.5% | 30% | 35% | 12.5% |

A logistic fit over all 180 games puts it at **1704 Elo (95% interval 1640 to 1766)**.
Stockfish calibrates `UCI_Elo` at longer time controls, so read this as a relative scale rather
than a leaderboard prediction.

### Versions

Each version was measured head to head against its predecessor with `harness/arena.py`.

| Version | What changed | Measured |
|---|---|---|
| v1 | PVS, TT, null move, LMR, quiescence, tapered eval | 1704 Elo on the ladder above |
| v2a | SEE in ordering and quiescence; clock management rewritten | 50% vs Stockfish 1700 over 20 games (v1: 45%) |
| v3 | Reverse futility, futility and late move pruning; log LMR table | Folded into v4's test |
| v4 | Mate distance pruning; depth-preferred TT replacement | **+108 Elo vs v2a**, 65.0% over 20 games (−18 to +270) |
| **v5** | Internal iterative deepening | **+89 Elo vs v4**, 62.5% over 16 games (−11 to +206) |

### Tried and rejected

Every idea below was implemented and measured, and the numbers decided. Several are standard
in stronger engines and still lost here.

| Idea | Result | Why it lost |
|---|---|---|
| Attack-based king danger | **−127 Elo**, 32.5% over 20 games (−286 to −8) | Halved evaluation speed. The ply it cost was worth more than what it knew. |
| Razoring | **−89 Elo**, 37.5% over 16 games | Prunes on the static eval, which is not accurate enough to bet on. |
| Countermove heuristic | Neutral | Razoring with countermove read −22, razoring alone −89. |
| Null-move reduction 3 | 25% vs Stockfish 1700 over 8 games | Reduction 2 read 62.5% on the same test. |
| Staged move generation | 1.42× slower | Asking python-chess for captures and quiets separately costs more than it saves. |

### Rated games on the platform

| Round | Colour | Opponent | Result | Mean loss | Median loss | Blunders |
|---|---|---|---|---|---|---|
| 102 | White | The Biggest Blunderer | Draw, threefold repetition | 62.9 cp | 21 cp | 3 |
| 103 | Black | imperialists | Lost, checkmate | 79.5 cp | 23 cp | 3 |

Losses are centipawns given up per move against Stockfish at depth 16 on one thread. On the
match hardware the engine reached depth 8 to 10 at about 35k nodes per second. For scale, the
team that beat Loki 3.0 in round 105 played at 33.4 cp mean loss with no blunders.

## Where the strength goes

### Mistakes

Stockfish re-scored 8,215 of our moves from the ladder games:

| | Opening | Middlegame | Endgame | All |
|---|---|---|---|---|
| Mean loss | 65.8 cp | 84.9 cp | 48.1 cp | 66.0 cp |
| Blunders (over 300 cp) | 4.2% | 5.6% | 2.7% | 4.1% |

The evaluation agrees with Stockfish's at **r = 0.21** over 3,241 positions. Its main defect is
compression toward zero: in round 102 it read +90 cp in a position Stockfish scored +702.

### Search compared with Loki 3.0

[Loki 3.0](https://github.com/BimmerBass/Loki) is an open-source C++ engine and the
highest-rated engine on the platform's leaderboard. v4 scored 0/8 against it. To separate
search quality from speed, both engines were given the same positions and depths:

| | Ours (Python) | Loki 3.0 (C++) |
|---|---|---|
| Effective branching factor | 2.3 to 2.7 | 2.6 to 2.8 |
| Tactics solved, of 24 | 23 | 23 |
| Median nodes to solve | 219 | 84 |

The search finds the same moves, and needs about 2.6× more nodes to get there. At a branching
factor near 2.5 that costs roughly one ply on every move.

### Speed

Profiling a 4-second middlegame search:

| Where the time goes | Share |
|---|---|
| Our code: `evaluate`, move ordering, `see`, `search` | 25% |
| python-chess (move generation, `push`, attack masks) and builtins | 43% |

python-chess generates legal moves about 24k times a second on this machine, and the search
visits about 25k nodes a second, so the engine runs at the library's ceiling. Making all of our
own code free would buy at most 1.33×. The large remaining gap to compiled engines is the
interpreter, which is why numba only pays if it replaces move generation too.

## Bugs that cost games

- **Illegal moves.** `board.push()` ran outside `try/finally`, so a timeout unwound past
  `board.pop()` and left the board off its root. 5 of 16 test games ended on an illegal move.
  Every push is now paired with a `finally: board.pop()`, and the chosen move is checked
  against a fresh board before it is returned.
- **Silent logs.** Output to a pipe is block-buffered and dies with the process. Every
  diagnostic print now flushes.

## Layout

```
agent.py              the submission, currently v5
versions/v1 ... v5    each measured build, playable as an arena opponent
versions/fixed        the engine stopped at FIXED_DEPTH, for depth-matched comparisons
versions/nodecap      the engine stopped at NODE_LIMIT nodes, for node-matched comparisons
sparring/stockfish    Stockfish over UCI at SPAR_ELO
sparring/loki         Loki 3.0.0 at full strength
sparring/loki_depth   Loki pinned to FIXED_DEPTH
baselines/            the starter's random, greedy, minimax and numba agents
harness/              local copy of the platform's protocol and clock, never edited
docs/IDEAS.md         the starter's notes on where engine strength comes from
```

Kept out of git: `engines/` (the Loki binary), `analysis/` (ladder parsing, Elo fitting and
Stockfish diagnosis scripts with their data), and game logs and PGNs.

## Running it

```
make setup    # uv sync, python 3.12
make gate     # ruff, mypy and two games that must finish cleanly
make zip      # build submission.zip and smoke it the way the platform does
```

Measure a change. A few games give a direction and 16 to 20 decide:

```
uv run python -m harness.arena --agent versions/v5 --opponent versions/v4 --games 16
SPAR_ELO=1700 uv run python -m harness.arena --opponent sparring/stockfish --games 8
uv run python -m harness.arena --opponent sparring/loki --games 8
```

The arena defaults to 10 s + 0.1 s. Add `--base-ms 120000 --increment-ms 500` for the real
clock. The Stockfish partner expects `stockfish` on `PATH` or a path in `SPAR_ENGINE`. Fetch
Loki with:

```
gh release download v3.0.0 --repo BimmerBass/Loki --pattern Loki3.0.0-x64.exe --dir engines
```

## Fair play

Stockfish and Loki run only as separate processes during local testing. `make zip` ships
`agent.py` alone, and nothing in it imports, calls or embeds another engine. Loki is GPL-3.0
and is not redistributed here.

## Next

1. **Texel-tune the evaluation.** Fitting the weights to game results attacks the r = 0.21
   evaluation and costs no speed. It is the largest lever left.
2. **Jit move generation with numba.** Raising the node-rate ceiling means replacing
   python-chess in the search loop, which is a rewrite.

## License

MIT. Built on [aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter).
