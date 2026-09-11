"""Train the evaluation network on positions Stockfish scored.

Reads the .npz shards written by training/prepare.py:

    features  int16 [N, 32]  White's view: (colour * 6 + piece) * 64 + square, padded with 768
    stm       int8  [N]      side to move, 0 for White and 1 for Black
    score     int16 [N]      centipawns for the side to move, clamped to +-2000

and writes the best weights so far to --out, the tensors agent.py loads at import.

The network sees the 768 piece-square inputs from both sides through one shared layer of H
clipped-ReLU units, side to move first, then 32 ReLU units and one output. The output times
400 is centipawns for the side to move, and the loss compares win probabilities, so a 50 cp
miss in a level position costs far more than the same miss at +1500.

Every --eval-seconds it scores the held-out positions, appends a row to --log and redraws
--plot. The learning rate follows a cosine over --minutes. To stop early, create a file named
STOP beside the log, or press Ctrl+C: either way the best weights are already on disk.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAD = 768
SCALE = 400.0


def mirror_table() -> torch.Tensor:
    """Maps a White-view input to the same piece seen by Black: colours swap, ranks flip."""
    table = torch.full((PAD + 1,), PAD, dtype=torch.long)
    for colour in range(2):
        for piece in range(6):
            for square in range(64):
                table[(colour * 6 + piece) * 64 + square] = ((1 - colour) * 6 + piece) * 64 + (
                    square ^ 56
                )
    return table


class Net(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ft = nn.EmbeddingBag(PAD + 1, hidden, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.full((hidden,), 0.5))
        self.l1 = nn.Linear(2 * hidden, 32)
        self.l2 = nn.Linear(32, 1)
        with torch.no_grad():
            nn.init.normal_(self.ft.weight, std=0.02)
            self.ft.weight[PAD].zero_()

    def forward(self, white: torch.Tensor, black: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        from_white = self.ft(white) + self.ft_bias
        from_black = self.ft(black) + self.ft_bias
        black_moves = stm.bool().unsqueeze(1)
        mover = torch.where(black_moves, from_black, from_white)
        other = torch.where(black_moves, from_white, from_black)
        x = torch.cat((mover, other), dim=1).clamp(0.0, 1.0)
        return self.l2(torch.relu(self.l1(x))).squeeze(1)


def load(folder: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shards = sorted(folder.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no .npz shards in {folder}")
    features, stm, score = [], [], []
    for shard in shards:
        with np.load(shard) as data:
            features.append(data["features"])
            stm.append(data["stm"])
            score.append(data["score"])
    return (
        torch.from_numpy(np.concatenate(features)),
        torch.from_numpy(np.concatenate(stm)),
        torch.from_numpy(np.concatenate(score)),
    )


def export(net: Net, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "w1": net.ft.weight[:PAD],
        "b1": net.ft_bias,
        "w2": net.l1.weight,
        "b2": net.l1.bias,
        "w3": net.l2.weight[0],
        "b3": net.l2.bias,
    }
    torch.save({k: v.detach().float().cpu().contiguous() for k, v in tensors.items()}, path)


def plot(log: Path, image: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with log.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    seen = [float(r["positions_m"]) for r in rows]
    figure, (losses, fit) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    losses.plot(seen, [float(r["train_loss"]) for r in rows], label="training")
    losses.plot(seen, [float(r["val_loss"]) for r in rows], label="validation", marker="o", ms=3)
    losses.set_ylabel("loss (win-probability MSE)")
    losses.set_title(title)
    losses.grid(alpha=0.3)
    losses.legend()
    fit.plot(seen, [float(r["val_error_cp"]) for r in rows], color="tab:red", marker="o", ms=3)
    fit.set_ylabel("mean error, level positions (cp)")
    fit.set_xlabel("positions trained on (millions)")
    fit.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(image, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/nnue.pt"))
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--minutes", type=float, default=40.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val", type=float, default=0.01)
    parser.add_argument("--eval-seconds", type=float, default=60.0)
    parser.add_argument("--log", type=Path, default=Path("train_log.csv"))
    parser.add_argument("--plot", type=Path, default=Path("loss.png"))
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, stm, score = load(args.data)
    n = len(score)
    order = torch.randperm(n)
    n_val = max(1, int(n * args.val))
    val, train = order[:n_val].to(device), order[n_val:].to(device)
    features, stm, score = features.to(device), stm.to(device), score.to(device)
    mirror = mirror_table().to(device)
    title = f"L1 = {args.hidden}, {n / 1e6:.1f}M positions"
    print(f"{n:,} positions ({len(train):,} train, {n_val:,} validation) on {device}", flush=True)

    net = Net(args.hidden).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    stop_file = args.log.parent / "STOP"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            ["minutes", "positions_m", "lr", "train_loss", "val_loss", "val_error_cp", "val_r"]
        )

    def predict(rows: torch.Tensor) -> torch.Tensor:
        white = features[rows].long()
        return net(white, mirror[white], stm[rows])

    def validate() -> tuple[float, float, float]:
        net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, n_val, args.batch):
                outputs.append(predict(val[start : start + args.batch]) * SCALE)
        net.train()
        cp, want = torch.cat(outputs), score[val].float()
        loss = ((torch.sigmoid(cp / SCALE) - torch.sigmoid(want / SCALE)) ** 2).mean().item()
        level = want.abs() < 300
        error = (cp[level] - want[level]).abs().mean().item()
        r = torch.corrcoef(torch.stack((cp, want)))[0, 1].item()
        return loss, error, r

    started = time.perf_counter()
    last_eval = started
    budget = args.minutes * 60
    best = math.inf
    seen = 0
    running = torch.zeros((), device=device)
    batches = 0

    def checkpoint() -> None:
        nonlocal best, running, batches, last_eval
        val_loss, error, r = validate()
        minutes = (time.perf_counter() - started) / 60
        train_loss = (running / max(batches, 1)).item()
        lr = optimiser.param_groups[0]["lr"]
        with args.log.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    f"{minutes:.2f}",
                    f"{seen / 1e6:.2f}",
                    f"{lr:.2e}",
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{error:.1f}",
                    f"{r:.4f}",
                ]
            )
        note = ""
        if val_loss < best:
            best = val_loss
            export(net, args.out)
            note = f"  saved {args.out}"
        print(
            f"{minutes:5.1f} min  {seen / 1e6:7.1f}M seen  train {train_loss:.5f}  "
            f"val {val_loss:.5f}  error {error:.0f} cp  r {r:.3f}  lr {lr:.1e}{note}",
            flush=True,
        )
        plot(args.log, args.plot, title)
        running = torch.zeros((), device=device)
        batches = 0
        last_eval = time.perf_counter()

    try:
        while True:
            shuffled = train[torch.randperm(len(train), device=device)]
            for start in range(0, len(train), args.batch):
                elapsed = time.perf_counter() - started
                fraction = min(1.0, elapsed / budget)
                warm = min(1.0, elapsed / (0.02 * budget))
                lr = args.lr * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * fraction)))
                for group in optimiser.param_groups:
                    group["lr"] = lr

                rows = shuffled[start : start + args.batch]
                target = torch.sigmoid(score[rows].float() / SCALE)
                loss = ((torch.sigmoid(predict(rows)) - target) ** 2).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                running += loss.detach()
                batches += 1
                seen += len(rows)

                now = time.perf_counter()
                if now - last_eval >= args.eval_seconds:
                    checkpoint()
                    if stop_file.exists():
                        print(f"found {stop_file}, stopping", flush=True)
                        return
                if now - started >= budget:
                    checkpoint()
                    print("time budget reached", flush=True)
                    return
    except KeyboardInterrupt:
        print("interrupted, scoring the current weights once more", flush=True)
        checkpoint()


if __name__ == "__main__":
    main()
