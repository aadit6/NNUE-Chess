"""Plot the network's training curves from Lightning's metrics.csv logs.

    uv run --with matplotlib python training/plot_curves.py \
        version_0/metrics.csv version_1/metrics.csv --out training/curves

Give the logs in the order they ran. A later log resumes from the last checkpoint of the one before
it, so the global step, not Lightning's epoch counter, places every point: resuming re-uses the
counter of the checkpoint's final epoch while training on new positions.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

BATCH_SIZE = 16_384
EPOCH_SIZE = 100_000_000
START_LR = 4.375e-4
GAMMA = 0.995

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"


@dataclass
class Run:
    """Loss every logging interval, and training and validation loss per epoch."""

    steps: list[int] = field(default_factory=list)
    step_loss: list[float] = field(default_factory=list)
    epoch_steps: list[int] = field(default_factory=list)
    epoch_counters: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    @property
    def epochs(self) -> list[int]:
        """Epochs of training completed at each logged epoch, counted across resumed runs."""
        return [epoch_of(step) for step in self.epoch_steps]


def epoch_of(step: int) -> int:
    return round(step * BATCH_SIZE / EPOCH_SIZE)


def billions(step: int) -> float:
    return step * BATCH_SIZE / 1e9


def read_run(path: Path) -> Run:
    run = Run()
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["train_loss"]:
                run.steps.append(int(row["step"]))
                run.step_loss.append(float(row["train_loss"]))
            elif row["train_loss_epoch"]:
                run.epoch_steps.append(int(row["step"]))
                run.epoch_counters.append(int(row["epoch"]))
                run.train_loss.append(float(row["train_loss_epoch"]))
            elif row["val_loss_epoch"]:
                run.val_loss.append(float(row["val_loss_epoch"]))
    complete = len(run.val_loss)  # an epoch still validating when the log was copied is left out
    del run.epoch_steps[complete:], run.epoch_counters[complete:], run.train_loss[complete:]
    return run


def merge(runs: list[Run]) -> tuple[Run, list[int]]:
    """One continuous run, and the global steps each later run resumed from."""
    merged = Run()
    for run in runs:
        merged.steps += run.steps
        merged.step_loss += run.step_loss
        merged.epoch_steps += run.epoch_steps
        merged.epoch_counters += run.epoch_counters
        merged.train_loss += run.train_loss
        merged.val_loss += run.val_loss
    return merged, [run.epoch_steps[-1] for run in runs[:-1]]


def style(ax: Axes, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="semibold", pad=24)
    ax.text(
        0, 1.025, subtitle, transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, va="bottom"
    )
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def mark_resumes(ax: Axes, xs: list[float]) -> None:
    for x in xs:
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            "resumed from checkpoint",
            xy=(x, 1),
            xycoords=("data", "axes fraction"),
            xytext=(5, -2),
            textcoords="offset points",
            color=MUTED,
            fontsize=8.5,
            va="top",
        )


def legend(ax: Axes) -> None:
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        loc="upper right",
        bbox_to_anchor=(1, 0.93),
    )


def loss_axis(ax: Axes, decimals: int) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.{decimals}f}"))


def plot_epoch_loss(ax: Axes, curves: Run, resumes: list[int], first_epoch: int) -> None:
    keep = [i for i, epoch in enumerate(curves.epochs) if epoch >= first_epoch]
    epochs = [curves.epochs[i] for i in keep]
    val = [curves.val_loss[i] for i in keep]
    ax.plot(
        epochs,
        [curves.train_loss[i] for i in keep],
        color=ORANGE,
        linewidth=2,
        label="training loss, epoch mean",
    )
    ax.plot(epochs, val, color=BLUE, linewidth=2, label="validation loss")
    ax.plot(
        epochs[-1],
        val[-1],
        "o",
        color=BLUE,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    ax.annotate(
        f"epoch {epochs[-1]}: {val[-1]:.5f}",
        xy=(epochs[-1], val[-1]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        va="top",
        color=INK,
        fontsize=9,
    )
    mark_resumes(ax, [epoch_of(step) for step in resumes])
    if first_epoch == 1:
        style(
            ax,
            "Loss per epoch",
            "each epoch is 100M positions; validation samples 1M positions",
            "epoch",
            "loss",
        )
        loss_axis(ax, 4)
    else:
        subtitle = "the same curves on a tighter scale, to show how far they have flattened"
        style(ax, f"Loss per epoch, from epoch {first_epoch}", subtitle, "epoch", "loss")
        loss_axis(ax, 5)
    legend(ax)


def plot_step_loss(ax: Axes, curves: Run, resumes: list[int]) -> None:
    interval = curves.steps[1] - curves.steps[0]
    xs = [billions(step) for step in curves.steps]
    ax.plot(
        xs,
        curves.step_loss,
        color=BLUE_LIGHT,
        linewidth=1,
        label=f"batch loss, every {interval:,} batches",
    )
    epoch_xs = [billions(step) for step in curves.epoch_steps]
    ax.plot(epoch_xs, curves.train_loss, color=BLUE, linewidth=2, label="epoch mean")
    mark_resumes(ax, [billions(step) for step in resumes])
    subtitle = f"batches of {BATCH_SIZE:,} positions, noisy because each is scored on its own"
    style(
        ax, "Training loss over positions seen", subtitle, "positions trained on (billions)", "loss"
    )
    loss_axis(ax, 4)
    legend(ax)


def plot_improvement(ax: Axes, curves: Run, resumes: list[int], first_epoch: int = 10) -> None:
    val = curves.val_loss
    change = [100 * (before - after) / before for before, after in pairwise(val)]
    average = [
        sum(change[max(0, i - 4) : i + 1]) / len(change[max(0, i - 4) : i + 1])
        for i in range(len(change))
    ]
    keep = [i for i, epoch in enumerate(curves.epochs[1:]) if epoch >= first_epoch]
    epochs = [curves.epochs[i + 1] for i in keep]
    ax.bar(
        epochs,
        [change[i] for i in keep],
        width=0.75,
        color=BLUE_LIGHT,
        label="change from previous epoch",
    )
    ax.plot(epochs, [average[i] for i in keep], color=BLUE, linewidth=2, label="5-epoch average")
    ax.axhline(0, color=MUTED, linewidth=1)
    mark_resumes(ax, [epoch_of(step) + 0.5 for step in resumes])
    subtitle = (
        "percent drop in validation loss; above zero is better, and 1M-position validation is noisy"
    )
    style(
        ax,
        f"Validation improvement per epoch, from epoch {first_epoch}",
        subtitle,
        "epoch",
        "improvement (%)",
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.1f}"))
    legend(ax)


def plot_learning_rate(ax: Axes, curves: Run, resumes: list[int]) -> None:
    rates = [START_LR * GAMMA**counter for counter in curves.epoch_counters]
    ax.plot(curves.epochs, rates, color=BLUE, linewidth=2, drawstyle="steps-mid")
    mark_resumes(ax, [epoch_of(step) + 0.5 for step in resumes])
    subtitle = (
        f"{START_LR} x {GAMMA} per epoch, rebuilt from the schedule because the trainer does not "
        "log it"
    )
    style(ax, "Learning rate", subtitle, "epoch", "learning rate")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2e}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv logs.")
    parser.add_argument(
        "logs", nargs="+", type=Path, help="metrics.csv files, in the order they ran"
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "curves")
    args = parser.parse_args()

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURFACE})
    curves, resumes = merge([read_run(path) for path in args.logs])
    args.out.mkdir(parents=True, exist_ok=True)

    charts: dict[str, Callable[[Axes], None]] = {
        "loss_per_epoch.png": lambda ax: plot_epoch_loss(ax, curves, resumes, first_epoch=1),
        "loss_from_epoch_20.png": lambda ax: plot_epoch_loss(ax, curves, resumes, first_epoch=20),
        "loss_per_step.png": lambda ax: plot_step_loss(ax, curves, resumes),
        "validation_improvement.png": lambda ax: plot_improvement(ax, curves, resumes),
        "learning_rate.png": lambda ax: plot_learning_rate(ax, curves, resumes),
    }
    for name, draw in charts.items():
        figure, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
        draw(ax)
        figure.savefig(args.out / name, dpi=150)
        plt.close(figure)

    epochs, seen = curves.epochs[-1], billions(curves.epoch_steps[-1])
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.6), layout="constrained")
    for ax, name in zip(axes.flat, list(charts)[:4], strict=True):
        charts[name](ax)
    headline = (
        f"HalfKAv2_hm network, L1 1024: {epochs} epochs, {seen:.1f} billion positions trained on"
    )
    figure.suptitle(headline, x=0.01, ha="left", fontsize=15, color=INK, fontweight="semibold")
    figure.savefig(args.out / "overview.png", dpi=150)
    plt.close(figure)

    print(f"{epochs} epochs, {seen:.2f}B positions, validation loss {curves.val_loss[-1]:.5f}")
    print(f"wrote {len(charts) + 1} charts to {args.out}")


if __name__ == "__main__":
    main()
