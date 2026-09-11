"""Convert an Arena .abk book tree into a Polyglot .bin, keeping the most played moves.

    uv run python abk_to_polyglot.py BOOK.abk OUT.bin MAX_ENTRIES

Transpositions are merged by Polyglot key and their game counts summed. Weight is the number of
games the move was played in, scaled per position to fit 16 bits, so the most played move wins.
"""

import struct
import sys
import time
from collections import defaultdict

import chess
import chess.polyglot
import numpy as np

ENTRY = np.dtype(
    [
        ("from", "u1"),
        ("to", "u1"),
        ("promotion", "i1"),
        ("priority", "u1"),
        ("games", "<i4"),
        ("won", "<i4"),
        ("lost", "<i4"),
        ("hz", "<i4"),
        ("child", "<i4"),
        ("sibling", "<i4"),
    ]
)
ROOT = 900
PROMOTIONS = {0: None, 1: chess.ROOK, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.QUEEN}
POLYGLOT_PROMOTION = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
CASTLE_ROOK = {chess.G1: chess.H1, chess.C1: chess.A1, chess.G8: chess.H8, chess.C8: chess.A8}
LAST_OPENING_MOVE = 20

source, out, max_entries = sys.argv[1], sys.argv[2], int(sys.argv[3])
entries = np.fromfile(source, dtype=ENTRY)
count = len(entries)
frm, to, promo = entries["from"].tolist(), entries["to"].tolist(), entries["promotion"].tolist()
games, won, lost = entries["games"].tolist(), entries["won"].tolist(), entries["lost"].tolist()
child, sibling = entries["child"].tolist(), entries["sibling"].tolist()
print(f"{count:,} abk entries")

stats: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
black_first: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
illegal = visited = 0
board = chess.Board()
sys.setrecursionlimit(20_000)


def encode(move: chess.Move) -> int:
    target = CASTLE_ROOK[move.to_square] if board.is_castling(move) else move.to_square
    return (
        chess.square_file(target)
        | chess.square_rank(target) << 3
        | chess.square_file(move.from_square) << 6
        | chess.square_rank(move.from_square) << 9
        | POLYGLOT_PROMOTION[move.promotion] << 12
    )


def walk(index: int) -> None:
    global illegal, visited
    key = chess.polyglot.zobrist_hash(board)
    while ROOT <= index < count:
        visited += 1
        move = chess.Move(frm[index], to[index], PROMOTIONS.get(abs(promo[index])))
        # The rules allow opening tables only for positions whose move number is 20 or lower,
        # and no line reaches back below move 20 once it has passed it.
        if board.fullmove_number > LAST_OPENING_MOVE:
            return
        if board.is_legal(move):
            record = stats[(key, encode(move))]
            record[0] += games[index]
            record[1] += won[index]
            record[2] += lost[index]
            if board.ply() == 1:
                first = black_first[board.san(move)]
                first[0] += games[index]
                first[1] += won[index]
                first[2] += lost[index]
            if ROOT <= child[index] < count:
                board.push(move)
                walk(child[index])
                board.pop()
        else:
            illegal += 1
        index = sibling[index]


started = time.time()
walk(ROOT)
print(
    f"walked {visited:,} entries in {time.time() - started:.0f} s, {illegal:,} illegal, "
    f"{len(stats):,} distinct moves"
)
top = sorted(black_first.items(), key=lambda item: -item[1][0])[:3]
print("black first replies (games, won, lost):", top)

ranked = sorted(stats.items(), key=lambda item: -item[1][0])
threshold = ranked[min(max_entries, len(ranked)) - 1][1][0]
kept = [item for item in ranked if item[1][0] >= threshold]
if len(kept) > max_entries:
    kept = [item for item in ranked if item[1][0] > threshold]
print(f"kept {len(kept):,} moves played in at least {kept[-1][1][0]} games")

by_key: dict[int, list[tuple[int, int]]] = defaultdict(list)
for (key, move16), (played, _, _) in kept:
    by_key[key].append((move16, played))
with open(out, "wb") as file:
    for key in sorted(by_key):
        moves = by_key[key]
        scale = min(1.0, 65535 / max(played for _, played in moves))
        for move16, played in sorted(moves, key=lambda item: -item[1]):
            file.write(struct.pack(">QHHI", key, move16, max(1, round(played * scale)), 0))
print(f"wrote {out}: {len(kept) * 16:,} bytes, {len(by_key):,} positions")
