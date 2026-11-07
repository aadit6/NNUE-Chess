# NNUE-Chess

Chess engine in Python. Numba-compiled alpha-beta search, evaluated by a HalfKAv2_hm NNUE trained
from scratch.

![Python](https://img.shields.io/badge/python-3.12-3776ab?style=flat-square)
![Numba](https://img.shields.io/badge/numba-JIT-00a3e0?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-black?style=flat-square)

<p align="center">
  <img src="docs/img/game.gif" alt="NNUE-Chess vs Stockfish at UCI_Elo 2750" width="720">
</p>

```python
def get_move(fen: str, time_left_ms: int) -> str: ...
```

One core, 120 s + 0.5 s, no GPU, under 50 MB. Built for [AI Chessathon](https://aichessathon.com).

## Results

| Opponent | Score |
|---|---|
| AI Chessathon final, 13-round Swiss | 7 / 13 (+6 =2 −5) |
| Stockfish, `UCI_Elo` 2000 | 75% |
| Stockfish, `UCI_Elo` 2500 | 70% |
| Stockfish, `UCI_Elo` 2750 | 55% |

Stockfish rows: 10 games each at 10 s + 0.1 s. Median depth 10 at the real clock.

## Engine

| | |
|---|---|
| Search | PVS, iterative deepening, aspiration windows, TT, null-move / futility / LMP, LMR, IID, quiescence with SEE |
| Eval | HalfKAv2_hm: 22,528 → 2×1,024 → 8 layer stacks, ~23.5M params, incremental accumulators |
| Board | Bitboards, Zobrist hashing, make/unmake, all compiled with numba |
| Book | HIARCS master games, capped at move 20 (not redistributed) |

## Network

Trained from random init with [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)
on ~3B Stockfish-scored positions: 121 epochs, ~2.5 h on one RTX 4090, validation loss
0.00343 → 0.00156. Inference is our own, in `nnue_eval.py`.

## Usage

```sh
make setup    # uv sync
make verify   # compiled engine vs python-chess and the reference implementation
make gate     # ruff, mypy, smoke games
make zip      # build submission.zip
make play     # one game at the real clock
```

```sh
uv run python -m harness.arena --agent . --opponent versions/v9b --games 8
SPAR_ELO=2500 uv run python -m harness.arena --agent . --opponent sparring/stockfish --games 10
```

## Layout

```
agent.py        entry point: clock, book, iterative deepening
position.py     board representation and move generation
search.py       principal variation search
evaluation.py   accumulators, score cache, fallback eval
nnue_eval.py    network inference
weights/        nnue.pt (48 MB)
training/       training scripts and metrics
versions/       every measured build, playable as an opponent
harness/        local copy of the platform runner
tools/          benchmarks, verification, book conversion
```

## License

MIT. Opening book and third-party engines not included. Based on
[aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter).
