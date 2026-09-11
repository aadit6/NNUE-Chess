"""Turn the Lichess evaluation database into training shards for training/train.py.

    curl -sL https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc \
        | python training/prepare.py --input - --limit 30000000 --out data

Each line is one position with one or more Stockfish evaluations. We keep the deepest one and
drop positions a static evaluation should not be asked about: mate scores, the side to move in
check, and positions whose best move is a capture or a promotion. Lichess gives centipawns from
White's side, so the score is flipped when Black is to move.

Reading stops at --limit kept positions, which also stops the download when piped as above.
"""

import argparse
import io
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np

PAD = 768
CLAMP = 2000
CHUNK = 10_000


def lines_from(source: str) -> Iterator[str]:
    if source == "-":
        sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        yield from sys.stdin
    elif source.endswith(".zst"):
        import zstandard

        with open(source, "rb") as raw:
            stream = zstandard.ZstdDecompressor().stream_reader(raw)
            try:
                yield from io.TextIOWrapper(stream, encoding="utf-8")
            except zstandard.ZstdError:
                return  # a partial download ends mid-frame
    else:
        with open(source, encoding="utf-8") as handle:
            yield from handle


def parse(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Counter[str]]:
    features = np.full((len(lines), 32), PAD, dtype=np.int16)
    stm = np.zeros(len(lines), dtype=np.int8)
    score = np.zeros(len(lines), dtype=np.int16)
    dropped: Counter[str] = Counter()
    kept = 0
    for line in lines:
        try:
            record = json.loads(line)
            best = max(record["evals"], key=lambda e: (e["depth"], e["knodes"]))
            pv = best["pvs"][0]
            if "cp" not in pv:
                dropped["mate score"] += 1
                continue
            board = chess.Board(record["fen"])
            # Analysis boards can be set up by hand: extra queens, missing kings.
            if board.occupied.bit_count() > 32 or board.kings.bit_count() != 2:
                dropped["not a game position"] += 1
                continue
            if board.is_check():
                dropped["in check"] += 1
                continue
            move = chess.Move.from_uci(pv["line"].split()[0])
            if board.is_capture(move) or move.promotion:
                dropped["best move captures"] += 1
                continue
        except (ValueError, KeyError, IndexError):
            dropped["unreadable"] += 1
            continue
        column = 0
        for colour in (chess.WHITE, chess.BLACK):
            side = 0 if colour == chess.WHITE else 1
            for piece in chess.PIECE_TYPES:
                for square in chess.scan_forward(board.pieces_mask(piece, colour)):
                    features[kept, column] = (side * 6 + piece - 1) * 64 + square
                    column += 1
        cp = max(-CLAMP, min(CLAMP, int(pv["cp"])))
        stm[kept] = 0 if board.turn == chess.WHITE else 1
        score[kept] = cp if board.turn == chess.WHITE else -cp
        kept += 1
    return features[:kept], stm[:kept], score[:kept], dropped


def batches(lines: Iterator[str], size: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    for line in lines:
        chunk.append(line)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="-", help="'-' for stdin, a .zst file, or plain jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30_000_000)
    parser.add_argument("--shard", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pending_rows = kept = read = shards = 0
    dropped: Counter[str] = Counter()

    def flush() -> None:
        nonlocal pending, pending_rows, shards
        if not pending:
            return
        path = args.out / f"shard_{shards:04d}.npz"
        np.savez(
            path,
            features=np.concatenate([p[0] for p in pending]),
            stm=np.concatenate([p[1] for p in pending]),
            score=np.concatenate([p[2] for p in pending]),
        )
        print(f"  wrote {path.name}: {pending_rows:,} positions", flush=True)
        pending, pending_rows, shards = [], 0, shards + 1

    with Pool(args.workers) as pool:
        # Bounded rounds keep memory flat: the pool would otherwise swallow the whole stream.
        round_size = 2 * (pool._processes or 1)  # type: ignore[attr-defined]
        source = batches(lines_from(args.input), CHUNK)
        while kept < args.limit:
            work = [chunk for _, chunk in zip(range(round_size), source, strict=False)]
            if not work:
                break
            read += sum(len(chunk) for chunk in work)
            for features, stm, score, lost in pool.map(parse, work):
                take = min(len(score), args.limit - kept)
                pending.append((features[:take], stm[:take], score[:take]))
                pending_rows += take
                kept += take
                dropped.update(lost)
                if pending_rows >= args.shard:
                    flush()
            rate = read / (time.perf_counter() - started)
            print(f"read {read:,}, kept {kept:,} ({rate:,.0f} lines/s)", flush=True)
    flush()
    print(f"done: kept {kept:,} of {read:,} in {shards} shards; dropped {dict(dropped)}")


if __name__ == "__main__":
    main()
