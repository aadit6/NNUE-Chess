"""Export a trained nnue-pytorch HalfKAv2_hm checkpoint into the tensors agent.py loads.

Run inside the nnue-pytorch checkout on the training machine:

    python export_halfka.py runs/l1024/lightning_logs/version_0/checkpoints/last.ckpt nnue.pt

The feature transformer keeps nnue-pytorch's export layout, 22,528 inputs per side: 32 king
buckets times 704 piece-squares, with both kings sharing one plane and the factoriser folded
in. It is stored as float16 so the file fits the 50 MB cap. The eight layer stacks, one per
piece-count bucket, stay float32. No .nnue file is involved: nnue_eval.py evaluates these floats
directly, the way nnue-pytorch's own forward pass does.
"""

import argparse
import sys
from pathlib import Path

import torch


def export(checkpoint: str, l1: int) -> dict[str, object]:
    """Needs nnue-pytorch's `model` package importable."""
    from model import NNUE, NNUELightningConfig

    config = NNUELightningConfig(features="HalfKAv2_hm^")
    config.model_config.L1 = l1
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    nnue = NNUE(config=config)
    nnue.load_state_dict(state["state_dict"])
    nnue.eval()
    net = nnue.model
    quantization = net.quantization

    with torch.no_grad():
        ft = net.input.get_export_weights().float()
        stacks = list(net.layer_stacks.get_coalesced_layer_stacks())
        ft_weight = ft[:, :l1].to(torch.float16).contiguous()
        if not torch.isfinite(ft_weight).all():
            raise SystemExit("feature transformer weights overflow float16")
        tensors: dict[str, object] = {
            "l1": l1,
            "ft_weight": ft_weight,
            "ft_psqt": ft[:, l1:],
            "ft_bias": net.input.bias[:l1],
            "l1_weight": torch.stack([s[0].weight for s in stacks]),
            "l1_bias": torch.stack([s[0].bias for s in stacks]),
            "l2_weight": torch.stack([s[1].weight for s in stacks]),
            "l2_bias": torch.stack([s[1].bias for s in stacks]),
            "out_weight": torch.stack([s[2].weight for s in stacks]),
            "out_bias": torch.stack([s[2].bias for s in stacks]),
            "max_ft_activation": float(quantization.max_ft_activation),
            "max_hidden_activation": float(quantization.max_hidden_activation),
            "l0_correction": float(quantization.l0_correction_factor),
            "sqr_correction": float(quantization.sqr_crelu_correction_factor),
            "nnue2score": float(quantization.nnue2score),
        }
    # Plain tensors only: parameters would drag autograd state into the file.
    return {
        k: v.detach().to(v.dtype if v.dtype == torch.float16 else torch.float32).contiguous()
        if isinstance(v, torch.Tensor)
        else v
        for k, v in tensors.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--l1", type=int, default=1024)
    parser.add_argument("--repo", default=".", help="path to the nnue-pytorch checkout")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    tensors = export(args.checkpoint, args.l1)
    torch.save(tensors, args.out)
    print(f"wrote {args.out}: {Path(args.out).stat().st_size / 1e6:.2f} MB")
    for name, value in tensors.items():
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else value
        print(f"  {name:22s} {shape}")


if __name__ == "__main__":
    main()
