# NNUE-Chess

Team **Claude's Gambit**'s entry for [AI Chessathon](https://aichessathon.com): a Python chess
engine that searches in compiled code and evaluates with a neural network we trained from scratch
on a rented RTX 4090.

<p align="center">
  <img src="docs/img/game.gif" alt="Beating Stockfish capped at UCI_Elo 2750, with the network's evaluation beside the board" width="760">
</p>

The whole submission is a zip with `agent.py` at its root, exposing one function:

```python
def get_move(fen: str, time_left_ms: int) -> str: ...
```

It plays on one core with 120 s + 0.5 s per side, no network and no GPU, and everything it needs —
search, network, weights, opening book — ships inside 50 MB.
[aichessathon.com/docs](https://aichessathon.com/docs) is the canonical rulebook and it changes.

## What it can do

| | |
|---|---|
| The event | **7 / 13** in the 13-round final Swiss, 11 Sep 2026: 6 wins, 2 draws, 5 losses, and no game lost to an illegal move, a crash or a flag fall |
| Stockfish, capped | 75% at `UCI_Elo` 2000, 80% at 2250, 70% at 2500, 55% at 2750, ten games each at 10 s + 0.1 s |
| Against its own past | **+6 =2 −0** over the Python engine that played the event, at 10 s + 0.1 s |
| At the real clock | median depth 10 and about 2 s a move |

<p align="center">
  <img src="docs/img/ladder.png" alt="Score against capped Stockfish, first version against the network build" width="720">
</p>

Read the ladder as a relative scale, not a rating. The scores fall far more slowly than the Elo
model predicts, which says more about `UCI_Elo` at ten-second games than about us, and ten games
per level is ten games.

## How it works

`agent.py` holds only what runs once a move: the clock, the opening book, iterative deepening, and
the guarantee of a legal answer whatever goes wrong. The search itself is three modules that numba
compiles to machine code as they are imported.

| Module | What it holds |
|---|---|
| `position.py` | Bitboards, move generation, make and unmake, Zobrist keys, static exchange evaluation |
| `evaluation.py` | The network's accumulators and score cache, and a hand-written evaluation as the fallback |
| `search.py` | Principal variation search: the table, move ordering, pruning, reductions, quiescence |

python-chess reads the FEN, plays the book and checks the reply, but never enters the search: one
`board.push()` costs about 13 µs, where a whole compiled node costs well under one.

### The search

| Technique | What it does here |
|---|---|
| Iterative deepening, aspiration windows | Depth 1, 2, 3... keeping the last finished result; windows start ±40 cp around the previous score |
| Principal variation search | Fail-soft negamax; only the first move gets a full window |
| Transposition table | 1M entries, depth-preferred replacement, kept across moves in a game |
| Move ordering | Hash move, captures by MVV-LVA with losing ones demoted by static exchange evaluation, killers, history |
| Null-move, futility and late move pruning | Skip a turn and prune if the position still fails high; prune nodes whose static eval is far outside the window |
| Late move reductions | Quiet moves late in the ordering search shallower, by a log-based table |
| Internal iterative deepening | From depth 5, a node with no hash move searches shallow first to find one |
| Quiescence search | Captures only, with stand-pat, delta pruning and exchange pruning |

Each move gets `time_left / 34` plus 85% of the increment, which the engine measures rather than
trusts, and `/ 45` under 25 seconds. At the real clock that is depth 8 to 11. With 150 ms left it
answers from move ordering without searching at all.

<p align="center">
  <img src="docs/img/search.gif" alt="Iterative deepening on a tactic from our games: the preferred move at each depth" width="720">
</p>

### The evaluation

A **HalfKAv2_hm** network in the shape Stockfish's trainer produces, run by our own code in
`nnue_eval.py`:

| Part | Shape |
|---|---|
| Inputs | 22,528 sparse features: 32 king buckets × 704 piece-square planes, mirrored so the king is on files e to h |
| Feature transformer | 22,528 → 1,024 a side, clamped to [0, 1], then each side's halves multiplied pairwise |
| Layer stacks | 8, chosen by piece count: 1,024 → 32 → 64 → 32 → 64 → output, reading both hidden layers |
| Size | about 23.5M parameters, 23.07M of them in the feature transformer |

The weights ship as `weights/nnue.pt`, 48 MB, the feature transformer in float16 and the rest in
float32. Our forward pass matches nnue-pytorch's to 1.7e-6, and gives the same score with colours
swapped or files mirrored. One evaluation costs 35 µs from scratch and 12 µs when the accumulators
are already current, and scores are cached by transposition key. The search's pruning margins were
tuned in centipawns against the hand-written evaluation, so a fit over 462 positions scales the
network to match.

### The opening book

`book/hiarcs_ref.bin` is the HIARCS Reference Book Lite — games between players rated 2550 or more
up to 2009, no computers and no correspondence — converted from Arena's `.abk` tree by
`tools/abk_to_polyglot.py`, which walks all 2,346,362 entries, merges transpositions, weights each
move by the games it was played in and keeps moves played at least three times. The agent plays the
most-played move without searching.

The rules allow opening tables only for positions at move 20 or lower, so the book is trimmed there
and the agent checks the move number too. The book is **not in this repository**: HIARCS allows
personal use and forbids rehosting. Without it the agent prints `opening book: none` and searches
from move one.

## How it was built

### Training the network

Nothing was started from an existing network: no resumed checkpoint, and no published net
fine-tuned or re-exported. The training data is Stockfish-scored positions, which the rules allow.

| | |
|---|---|
| Data | six binpacks from [jshriver/historic-binpacks](https://huggingface.co/datasets/jshriver/historic-binpacks), about 3.0 billion positions scored at depth 9, read from `/dev/shm` |
| Trainer | [official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) as it ships: `HalfKAv2_hm^`, L1 1024, `--lambda 1.0`, batch 16,384 |
| Hardware | one rented vast.ai container with an RTX 4090, where the CPU data loader set the pace rather than the GPU |
| Run | 121 epochs over 12.1 billion positions in about 2.5 GPU hours; validation loss 0.00343 → 0.00156 |

<p align="center">
  <img src="docs/img/training.gif" alt="Training and validation loss drawn epoch by epoch" width="720">
</p>

`training/export_halfka.py` turns the checkpoint into the float tensors `nnue_eval.py` reads, and
`tools/nnue_sanity.py` checks it against positions with an obvious answer: +301 a knight up, +782 a
queen up, +489 in KQ v K. The scripts that rented and drove the box are in `training/vast/`.

### The versions

Each was measured head to head with `harness/arena.py`. A few games give a direction and 16 to 20
decide, so the early rows are directions.

| Version | What changed | Measured |
|---|---|---|
| v1 | PVS, transposition table, null move, LMR, quiescence, tapered hand-written eval | 1704 Elo on the Stockfish ladder |
| v2a | Exchange evaluation in ordering and quiescence; clock management rewritten | 50% vs Stockfish 1700, where v1 scored 45% |
| v4 | Reverse futility, futility and late move pruning; mate distance pruning | +108 Elo vs v2a over 20 games |
| v5 | Internal iterative deepening | +89 Elo vs v4 over 16 games |
| v7 | The network replaces the hand-written evaluation | 8-0 vs v5, 8-0 vs Stockfish 1700, +3 =1 −0 vs Loki 3.0 |
| v9b | The final network, and the HIARCS book trimmed to move 20 | The build that played the event, 7 / 13 |
| v13 | The whole search moved into compiled code | 13× the nodes a second over the same positions, the same 24 / 24 tactics |
| **v14** | Accumulators kept up to date move by move | +6 =2 −0 vs v9b, level with v13. The build at the root |

Each lives in `versions/` and is playable as an arena opponent, so any two can be matched against
each other. Three ordering and table experiments, v10 to v12, came out inside the noise and were
not adopted.

### From Python to compiled code

Profiling the Python engine blamed the language rather than the algorithm: a four-second search
spent about 43% of its time inside python-chess and 25% in its own bookkeeping. So v13 moved the
search into numba without changing a single rule, which is worth about 13× the nodes a second on
the same positions. Two details paid for that — writing out every compiled function's argument
types, without which numba compiles a separate copy per constant a caller passes and start-up blows
the 90 s budget, and `NUMBA_OPT=2`, which measured 12% faster than numba's default -O3.

v14 then kept the accumulators up to date move by move instead of rebuilding them on every call.
It works, and it came out level with v13, because the rebuild was not where the time went: a sparse
first layer over kept accumulators costs more than a dense one, and the search often finds no
up-to-date ancestor to walk forward from and rebuilds anyway. Giving the kept accumulators the
dense layer is the next thing to try.

`make verify` is what makes a rewrite that size safe to trust. It checks perft, legal moves and
hash keys against python-chess, then exchanges, both evaluations and the incrementally kept
accumulators against the Python engine they replaced and against a full rebuild, and exits
non-zero on the first mismatch.

## Running it

```
make setup    # uv sync, python 3.12
make verify   # the compiled engine against python-chess and the Python engine it replaced
make gate     # ruff, mypy and two games that must finish cleanly
make zip      # build submission.zip with weights/ and book/, then smoke it like the platform
make play     # one game against a baseline at the real clock
```

Measure a change:

```
uv run python -m harness.arena --agent . --opponent versions/v9b --games 8
SPAR_ELO=2500 uv run python -m harness.arena --agent . --opponent sparring/stockfish --games 10
uv run python tools/bench.py . 7            # nodes, time and tactics at a fixed depth
uv run python tools/bench_eval.py           # what one evaluation costs, each way it can run
```

The arena defaults to 10 s + 0.1 s; add `--base-ms 120000 --increment-ms 500` for the real clock.
The Stockfish partner expects `stockfish` on `PATH` or a path in `SPAR_ENGINE`.

Two things worth knowing before timing anything. Compiling costs about a minute every time a
process starts, which an arena pays twice a game; `NNUE_CHESS_CACHE_DIR=.numba_cache` caches it
between local runs, but it changes what a comparison says, so time a change with it off. And a
laptop is not a stable ruler — the same deterministic search has measured a factor of two apart on
different days — so only node counts and same-session ratios are worth writing down.

Rebuild the book from HIARCS's free download, personal use only:

```
uv run python tools/abk_to_polyglot.py HiarcsBookRef2550Lite.abk book/hiarcs_ref.bin 115000
```

## What's here

```
agent.py              the submission: the clock, the book, iterative deepening, a legal answer
position.py           the board, compiled: bitboards, move generation, make and unmake, keys
evaluation.py         the accumulators and score cache, and the hand-written fallback
search.py             principal variation search, compiled
nnue_eval.py          the network's arithmetic in numba, whole and split for incremental use
weights/nnue.pt       the network (48 MB)
book/                 the HIARCS book, not in git; rebuild with tools/abk_to_polyglot.py
versions/             every measured build, each playable as an arena opponent
training/             the vast.ai scripts, both runs' metrics, the exporter and the chart script
tools/                verify_compiled.py, bench.py, bench_eval.py, and the book and media scripts
sparring/             Stockfish over UCI at SPAR_ELO, and Loki 3.0 full strength or depth-pinned
baselines/            the starter's random, greedy, minimax and numba agents
harness/              local copy of the platform's protocol and clock, never edited
docs/img/             the GIFs and charts in this README
```

Kept out of git: the opening books and per-version weights, third-party engine binaries, zips and
game logs. One copy of the network, at `weights/nnue.pt`, is enough.

The zip `make zip` builds is 49.1 MB unzipped against the 50 MB cap, imports nothing outside the
platform's stack, opens no sockets or subprocesses, writes nothing, and answers legally with 120 s,
800 ms and 150 ms on the clock. Its one tight number is start-up: compiling everything takes 68 s
of the 90 s import budget, measured on a laptop rather than on the platform's core.

## Fair play

- **Network:** trained by us from random initialisation with nnue-pytorch, on Stockfish-scored
  positions as the rules allow. No published network was fine-tuned, re-exported or resumed from.
- **Engine:** `nnue_eval.py` is our own inference, not a port of an engine's, and `position.py`,
  `evaluation.py` and `search.py` are ours, written from the Python engine that came before them.
  Plain source, no native binaries, nothing generated or obfuscated.
- **Sparring:** Stockfish and Loki ran only as separate processes in local testing, and were never
  imported or embedded. Loki is GPL-3.0 and is not redistributed here.
- **Book:** human master games only, used only at move 20 or lower, read with `chess.polyglot`.

## License

MIT, except the opening book and third-party engines, which are not included. Built on
[aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter).
