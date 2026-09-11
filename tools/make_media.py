"""Render the README's animations and charts into docs/img.

    uv run --with pillow --with matplotlib python tools/make_media.py

game.gif      v9 beating Stockfish capped at UCI_Elo 2750, with the network's own evaluation of
              every position as a bar and a running graph
search.gif    iterative deepening on a tactic: the move the search prefers at each depth
training.gif  training and validation loss, drawn epoch by epoch over the 121 epochs
ladder.png    score against capped Stockfish for v1 (hand-written evaluation) and v9 (network)
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import chess
import chess.pgn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from plot_curves import BATCH_SIZE, merge, read_run  # noqa: E402

import agent  # noqa: E402
import nnue_eval  # noqa: E402
from harness.rules import OPENINGS  # noqa: E402

SQ = 52
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
LAST_LIGHT, LAST_DARK = (214, 205, 110), (178, 160, 62)
GROUND = (27, 26, 24)
PANEL = (39, 37, 34)
INK = (238, 236, 231)
MUTED = (152, 147, 139)
GREEN = (98, 168, 74)
AMBER = (224, 160, 58)
GLYPHS = {
    chess.KING: "♚",
    chess.QUEEN: "♛",
    chess.ROOK: "♜",
    chess.BISHOP: "♝",
    chess.KNIGHT: "♞",
    chess.PAWN: "♟",
}
REGULAR = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


# --- board ---------------------------------------------------------------------------------


def corner(square: chess.Square, flip: bool) -> tuple[int, int]:
    file, rank = chess.square_file(square), chess.square_rank(square)
    column, row = (7 - file, rank) if flip else (file, 7 - rank)
    return column * SQ, row * SQ


def centre(square: chess.Square, flip: bool) -> tuple[float, float]:
    x, y = corner(square, flip)
    return x + SQ / 2, y + SQ / 2


def draw_board(
    board: chess.Board,
    last: chess.Move | None = None,
    arrow: chess.Move | None = None,
    arrow_colour: tuple[int, int, int] = GREEN,
    flip: bool = False,
) -> Image.Image:
    image = Image.new("RGB", (8 * SQ, 8 * SQ))
    draw = ImageDraw.Draw(image)
    pieces = font(int(SQ * 0.8))
    for square in chess.SQUARES:
        x, y = corner(square, flip)
        light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        colour = LIGHT if light else DARK
        if last is not None and square in (last.from_square, last.to_square):
            colour = LAST_LIGHT if light else LAST_DARK
        draw.rectangle((x, y, x + SQ - 1, y + SQ - 1), fill=colour)
        piece = board.piece_at(square)
        if piece is not None:
            white = piece.color == chess.WHITE
            draw.text(
                (x + SQ / 2, y + SQ / 2 + 2),
                GLYPHS[piece.piece_type],
                font=pieces,
                anchor="mm",
                fill=(252, 251, 248) if white else (24, 23, 21),
                stroke_width=2 if white else 0,
                stroke_fill=(24, 23, 21),
            )
    labels = font(11, bold=True)
    files = "abcdefgh"
    for i in range(8):
        on_light = (i + 7) % 2 == 0
        draw.text(
            (i * SQ + SQ - 3, 8 * SQ - 2),
            files[7 - i] if flip else files[i],
            font=labels,
            anchor="rd",
            fill=DARK if on_light else LIGHT,
        )
        on_light = i % 2 == 0
        draw.text(
            (3, i * SQ + 2),
            str(i + 1) if flip else str(8 - i),
            font=labels,
            anchor="lt",
            fill=DARK if on_light else LIGHT,
        )
    if arrow is None:
        return image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    (x0, y0), (x1, y1) = centre(arrow.from_square, flip), centre(arrow.to_square, flip)
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    head = SQ * 0.45
    bx, by = x1 - ux * head, y1 - uy * head
    rgba = (*arrow_colour, 200)
    pen.line((x0, y0, bx, by), fill=rgba, width=int(SQ * 0.2))
    spread = head * 0.62
    pen.polygon(
        [(x1, y1), (bx - uy * spread, by + ux * spread), (bx + uy * spread, by - ux * spread)],
        fill=rgba,
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )
    print(
        f"wrote {path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB"
    )


def expected(centipawns: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-centipawns / 400.0))


def opening_name(fen: str) -> str:
    return next((name for name, start in OPENINGS if start == fen), "curated opening")


# --- game.gif ------------------------------------------------------------------------------


def white_view(board: chess.Board) -> float:
    if board.is_checkmate():
        return -10_000.0 if board.turn == chess.WHITE else 10_000.0
    score = nnue_eval.evaluate(board)
    return float(score if board.turn == chess.WHITE else -score)


def game_gif() -> None:
    with (ROOT / "results" / "games" / "v9_vs_stockfish_2750.pgn").open() as file:
        game = chess.pgn.read_game(file)
    assert game is not None
    ours_white = game.headers["White"].startswith("v")
    board = game.board()
    opening = opening_name(board.fen())
    history: list[float] = []
    frames: list[Image.Image] = []
    moves = [None, *game.mainline_moves()]
    width, height = 16 + 18 + 12 + 8 * SQ + 22 + 262 + 16, 16 + 8 * SQ + 16
    for move in moves:
        label = "start"
        if move is not None:
            number = board.fullmove_number
            san = board.san(move)
            label = f"{number}. {san}" if board.turn == chess.WHITE else f"{number}... {san}"
            board.push(move)
        cp = white_view(board)
        history.append(cp)

        frame = Image.new("RGB", (width, height), GROUND)
        draw = ImageDraw.Draw(frame)
        # evaluation bar: White's share grows from the side White plays from
        bar_x, bar_top = 16, 16
        white_share = expected(max(-2000.0, min(2000.0, cp)))
        draw.rectangle((bar_x, bar_top, bar_x + 17, bar_top + 8 * SQ - 1), fill=(24, 23, 21))
        filled = round(8 * SQ * white_share)
        if filled > 0 and ours_white:
            draw.rectangle(
                (bar_x, bar_top + 8 * SQ - filled, bar_x + 17, bar_top + 8 * SQ - 1), fill=INK
            )
        elif filled > 0:
            draw.rectangle((bar_x, bar_top, bar_x + 17, bar_top + filled - 1), fill=INK)
        board_x = bar_x + 18 + 12
        frame.paste(draw_board(board, move, flip=not ours_white), (board_x, bar_top))

        panel_x = board_x + 8 * SQ + 22
        draw.rounded_rectangle((panel_x, 16, panel_x + 262, height - 16), radius=10, fill=PANEL)
        x = panel_x + 18
        draw.text((x, 32), "Claude's Gambit v9", font=font(17, bold=True), fill=INK)
        draw.text((x, 56), "vs Stockfish, UCI_Elo 2750", font=font(13), fill=MUTED)
        draw.text((x, 76), f"{opening} · 10 s + 0.1 s", font=font(12), fill=MUTED)
        draw.text((x, 118), label, font=font(24, bold=True), fill=INK)
        if board.is_checkmate():
            verdict = "checkmate, v9 wins"
            shown = "#"
        else:
            verdict = "network evaluation, White's view"
            shown = f"{cp / 100:+.2f}"
        draw.text(
            (x, 162), shown, font=font(30, bold=True), fill=GREEN if board.is_checkmate() else INK
        )
        draw.text((x, 202), verdict, font=font(12), fill=MUTED)

        # running graph of the network's evaluation, as White's expected score
        gx0, gy0, gw, gh = x, 236, 226, 150
        draw.rectangle((gx0, gy0, gx0 + gw, gy0 + gh), outline=(70, 67, 62))
        draw.line((gx0, gy0 + gh / 2, gx0 + gw, gy0 + gh / 2), fill=(70, 67, 62))
        points = [
            (
                gx0 + gw * i / max(1, len(moves) - 1),
                gy0 + gh * (1 - expected(max(-2000.0, min(2000.0, v)))),
            )
            for i, v in enumerate(history)
        ]
        if len(points) > 1:
            draw.line(points, fill=INK, width=2)
        px, py = points[-1]
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=GREEN)
        draw.text(
            (gx0, gy0 + gh + 8), "White winning ↑   Black winning ↓", font=font(11), fill=MUTED
        )
        frames.append(frame)

    durations = [1500] + [420] * (len(frames) - 2) + [4000]
    save_gif(frames, durations, OUT / "game.gif")


# --- search.gif ----------------------------------------------------------------------------


def search_to(fen: str, depth: int) -> tuple[chess.Move, int, int, int]:
    agent.MAX_DEPTH = depth
    searcher = agent.Searcher()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        move = searcher.choose(chess.Board(fen), 1e12, 1e12)
    found = re.search(r"depth (\d+) score ([+-]?\d+) nodes (\d+) time (\d+)ms", buffer.getvalue())
    if found is None:
        return move, 0, searcher.nodes, 0
    return move, int(found[2]), searcher.nodes, int(found[4])


def search_gif(deepest: int = 8) -> None:
    tactics = json.loads((ROOT / "tools" / "tactics.json").read_text())
    chosen = None
    for tactic in tactics:
        board = chess.Board(tactic["fen"])
        best = chess.Move.from_uci(tactic["best"])
        if search_to(tactic["fen"], 1)[0] != best and search_to(tactic["fen"], 6)[0] == best:
            chosen = tactic
            break
    assert chosen is not None, "no tactic whose answer changes with depth"
    fen = chosen["fen"]
    board = chess.Board(fen)
    best = chess.Move.from_uci(chosen["best"])
    flip = board.turn == chess.BLACK
    rows: list[tuple[int, str, int, int, int, bool]] = []
    frames: list[Image.Image] = []
    width, height = 16 + 8 * SQ + 22 + 300 + 16, 16 + 8 * SQ + 16
    for depth in range(1, deepest + 1):
        move, score, nodes, spent = search_to(fen, depth)
        solved = move == best
        rows.append((depth, board.san(move), score, nodes, spent, solved))
        frame = Image.new("RGB", (width, height), GROUND)
        draw = ImageDraw.Draw(frame)
        frame.paste(
            draw_board(board, arrow=move, arrow_colour=GREEN if solved else AMBER, flip=flip),
            (16, 16),
        )
        panel_x = 16 + 8 * SQ + 22
        draw.rounded_rectangle((panel_x, 16, panel_x + 300, height - 16), radius=10, fill=PANEL)
        x = panel_x + 18
        side = "White" if board.turn == chess.WHITE else "Black"
        draw.text((x, 32), "Iterative deepening", font=font(18, bold=True), fill=INK)
        draw.text((x, 58), f"{side} to move · a tactic from our games", font=font(12), fill=MUTED)
        draw.text(
            (x, 96), "depth   move      eval      nodes", font=font(12, bold=True), fill=MUTED
        )
        for row_index, (d, san, sc, nd, _, ok) in enumerate(rows):
            y = 118 + row_index * 26
            colour = GREEN if ok else AMBER
            draw.text((x + 12, y), f"{d}", font=font(15, bold=True), fill=INK)
            draw.text((x + 52, y), san, font=font(15, bold=True), fill=colour)
            draw.text((x + 124, y), f"{sc / 100:+.2f}", font=font(15), fill=INK)
            draw.text((x + 262, y), f"{nd:,}", font=font(15), fill=INK, anchor="ra")
        note = "found the winning move" if solved else "still looking"
        draw.text((x, height - 60), note, font=font(14, bold=True), fill=GREEN if solved else AMBER)
        draw.text(
            (x, height - 38), f"{spent / 1000:.1f} s at depth {depth}", font=font(12), fill=MUTED
        )
        frames.append(frame)
    durations = [1300] * (len(frames) - 1) + [4500]
    save_gif(frames, durations, OUT / "search.gif")


# --- training.gif --------------------------------------------------------------------------


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#898781")
    ax.tick_params(colors="#52514e", labelsize=9)


def training_gif() -> None:
    logs = ROOT / "training" / "logs"
    curves, resumes = merge([read_run(logs / f"version_{i}_metrics.csv") for i in (0, 1)])
    epochs, train, val = curves.epochs, curves.train_loss, curves.val_loss
    seen = [step * BATCH_SIZE / 1e9 for step in curves.epoch_steps]
    resume_epoch = round(resumes[0] * BATCH_SIZE / 100_000_000) if resumes else None

    figure, ax = plt.subplots(figsize=(8, 4.5), dpi=100, layout="constrained")
    figure.set_facecolor("#fcfcfb")
    style(ax)
    ax.set_xlim(0, epochs[-1] + 2)
    ax.set_ylim(0.0014, 0.0050)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.4f}"))
    ax.set_xlabel("epoch (100M positions each)", color="#52514e")
    ax.set_ylabel("loss", color="#52514e")
    ax.set_title(
        "Training the network from scratch: HalfKAv2_hm, L1 1024, one RTX 4090",
        loc="left",
        fontsize=12,
        color="#0b0b0b",
        fontweight="bold",
    )
    (train_line,) = ax.plot([], [], color="#eb6834", linewidth=2, label="training loss")
    (val_line,) = ax.plot([], [], color="#2a78d6", linewidth=2, label="validation loss")
    (dot,) = ax.plot(
        [], [], "o", color="#2a78d6", markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2
    )
    resume_line = ax.axvline(resume_epoch or 0, color="#898781", linewidth=1, linestyle=(0, (4, 3)))
    resume_line.set_visible(False)
    readout = ax.text(
        0.98, 0.94, "", transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#0b0b0b"
    )
    ax.legend(frameon=False, loc="upper center", fontsize=9, labelcolor="#52514e")

    frames = [*range(1, len(epochs) + 1, 2), len(epochs)] + [len(epochs)] * 24

    def update(count: int) -> None:
        train_line.set_data(epochs[:count], train[:count])
        val_line.set_data(epochs[:count], val[:count])
        dot.set_data([epochs[count - 1]], [val[count - 1]])
        resume_line.set_visible(resume_epoch is not None and epochs[count - 1] > resume_epoch)
        readout.set_text(
            f"epoch {epochs[count - 1]}\n{seen[count - 1]:.1f} billion positions\n"
            f"validation loss {val[count - 1]:.5f}"
        )

    animation = FuncAnimation(figure, update, frames=frames)
    path = OUT / "training.gif"
    animation.save(path, writer=PillowWriter(fps=14))
    plt.close(figure)
    print(
        f"wrote {path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB"
    )


# --- ladder.png ----------------------------------------------------------------------------


def ladder_png() -> None:
    results = ROOT / "results"
    v1 = json.loads((results / "stockfish_ladder_v1.json").read_text())
    v9 = json.loads((results / "stockfish_ladder_v9.json").read_text())
    fit = json.loads((results / "stockfish_elofit_v9.json").read_text())

    def score(row: dict[str, float]) -> float:
        return 100 * (row["wins"] + row["draws"] / 2) / (row["wins"] + row["draws"] + row["losses"])

    figure, ax = plt.subplots(figsize=(9, 5), dpi=150, layout="constrained")
    figure.set_facecolor("#fcfcfb")
    style(ax)
    ax.axhline(50, color="#898781", linewidth=1, linestyle=(0, (4, 3)))
    ax.plot(
        [r["elo"] for r in v1],
        [score(r) for r in v1],
        "-o",
        color="#2a78d6",
        linewidth=2,
        markersize=6,
        label="v1: hand-written evaluation, 20 games per level (fit 1704)",
    )
    ax.plot(
        [r["elo"] for r in v9],
        [score(r) for r in v9],
        "-o",
        color="#eb6834",
        linewidth=2,
        markersize=6,
        label=f"v9: trained network + book, 10 games per level (fit {fit['rating']})",
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel("Stockfish strength (UCI_Elo)", color="#52514e")
    ax.set_ylabel("our score (%)", color="#52514e")
    ax.set_title(
        "Score against capped Stockfish at 10 s + 0.1 s",
        loc="left",
        fontsize=12.5,
        color="#0b0b0b",
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "same harness, same curated openings; UCI_Elo is calibrated at slower games, so "
        "read the fits as relative",
        transform=ax.transAxes,
        fontsize=9,
        color="#52514e",
    )
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor="#52514e")
    path = OUT / "ladder.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ladder_png()
    training_gif()
    game_gif()
    search_gif()


if __name__ == "__main__":
    main()
