"""Probe-direction similarity across experiments.

Reads the saved probe_weights.pt files for the three v2 (transition)
probes and reports pairwise cosine similarity at each (layer, offset)
cell. Used to test whether the three behaviors share one "I'm
misaligning" direction or whether they have three independent ones.

Usage (from repo root):

    # Compare within-hack (positive-subset) probes at the v2 best layer:
    python tools/probe_similarity.py \\
        --layer 9 --offset 0 --label-variant transition \\
        --experiments sandbag manipulation deception

    # Or sweep all (layer, offset) cells and report similarity heatmap:
    python tools/probe_similarity.py --sweep --label-variant transition

The script expects probe_weights.pt files to live at
``results/<exp>/<exp>/results{_<variant>}/np10_mo10_ep200/<subset>/probe_weights.pt``
i.e. the local mirror produced by `make pull-<exp>` or
`modal run <exp>_experiment/run.py --stage probe`.

probe_weights.pt is saved by ``sweep_layers_and_offsets`` as:
    {"W": tensor(L, K, H), "b": tensor(L, K), "layers": [...], "offsets": [...]}

where L is layer count, K = max_offset+1, H is hidden dim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


SUBSET_BY_EXP = {
    "sandbag": "sandbag_prompts",
    "manipulation": "manipulation_prompts",
    "deception": "deception_prompts",
}


def _variant_suffix(label_variant: str) -> str:
    return "" if label_variant == "full" else f"_{label_variant}"


def _weights_path(
    experiment: str,
    subset: str,
    run_name: str = "np10_mo10_ep200",
    label_variant: str = "transition",
) -> Path:
    suffix = _variant_suffix(label_variant)
    return Path(
        f"results/{experiment}/{experiment}/results{suffix}/{run_name}/{subset}/probe_weights.pt"
    )


def _load_weights(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make pull-{path.parts[1]}-v2` (or the v1 equivalent) first."
        )
    return torch.load(path, weights_only=False)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan")
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return float("nan")
    return float((a @ b / (na * nb)).item())


def cmd_pair(args) -> None:
    """Pairwise similarity at a single (layer, offset)."""
    weights = {}
    for exp in args.experiments:
        subset = args.subset or SUBSET_BY_EXP[exp]
        path = _weights_path(exp, subset, label_variant=args.label_variant)
        d = _load_weights(path)
        # d["W"] has shape (L, K, H)
        if args.layer >= d["W"].shape[0]:
            raise SystemExit(f"layer {args.layer} out of range for {exp} (max L{d['W'].shape[0]-1})")
        if args.offset > d["offsets"][-1]:
            raise SystemExit(f"offset {args.offset} out of range for {exp} (max k={d['offsets'][-1]})")
        weights[exp] = d["W"][args.layer, args.offset].float()
        print(f"  loaded {exp} L{args.layer} k={args.offset}  norm={weights[exp].norm().item():.3f}")

    print()
    print(f"=== Cosine similarity at L{args.layer}, k={args.offset}, variant={args.label_variant!r} ===")
    exps = list(weights.keys())
    print(f"{'':>14s} " + " ".join(f"{e:>14s}" for e in exps))
    for a in exps:
        row = f"{a:>14s} "
        for b in exps:
            c = cosine(weights[a], weights[b])
            row += f"{c:>14.3f} "
        print(row)

    # Also output as JSON
    out = {
        "layer": args.layer,
        "offset": args.offset,
        "label_variant": args.label_variant,
        "experiments": exps,
        "cosine_matrix": [
            [cosine(weights[a], weights[b]) for b in exps] for a in exps
        ],
        "norms": {e: float(weights[e].norm().item()) for e in exps},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved -> {args.out}")


def cmd_sweep(args) -> None:
    """Sweep all (layer, offset) cells and compute pairwise similarities."""
    loaded = {}
    for exp in args.experiments:
        subset = args.subset or SUBSET_BY_EXP[exp]
        path = _weights_path(exp, subset, label_variant=args.label_variant)
        loaded[exp] = _load_weights(path)

    # Use the offsets / layers from the first experiment; verify others match
    ref = loaded[args.experiments[0]]
    layers = ref["layers"]
    offsets = ref["offsets"]
    H = ref["W"].shape[-1]
    for exp, d in loaded.items():
        if d["layers"] != layers or d["offsets"] != offsets or d["W"].shape[-1] != H:
            raise SystemExit(
                f"{exp} probe weights have mismatched shape: "
                f"layers={d['layers']} offsets={d['offsets']} H={d['W'].shape[-1]}"
            )

    exps = list(loaded.keys())
    pairs = [(a, b) for i, a in enumerate(exps) for b in exps[i + 1:]]

    out_grid = {}
    for a, b in pairs:
        Wa = loaded[a]["W"].float()
        Wb = loaded[b]["W"].float()
        # Cosine sim per (layer, offset)
        dot = (Wa * Wb).sum(dim=-1)
        na = Wa.norm(dim=-1)
        nb = Wb.norm(dim=-1)
        cos = dot / (na * nb).clamp(min=1e-8)
        cos[torch.isnan(cos)] = float("nan")
        out_grid[f"{a}__vs__{b}"] = cos.tolist()

    out = {
        "label_variant": args.label_variant,
        "layers": layers,
        "offsets": offsets,
        "experiments": exps,
        "cosine_grids": out_grid,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved -> {args.out}")
    else:
        for pair, grid in out_grid.items():
            print(f"\n=== {pair} ===")
            print("layer".rjust(5), " ".join(f"k={k:>2d}" for k in offsets))
            for li, layer in enumerate(layers):
                row = f"L{layer:>3d} " + " ".join(f"{grid[li][oi]:>+.3f}" for oi in range(len(offsets)))
                print(row)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-variant", default="transition", choices=("full", "transition"))
    parser.add_argument(
        "--experiments", nargs="+", default=["sandbag", "manipulation", "deception"],
        help="Which experiments to compare (default: all 3).",
    )
    parser.add_argument("--subset", default=None,
                        help="Override the subset directory name. Defaults to "
                             "<experiment>_prompts (the within-hack subset).")
    parser.add_argument("--out", default=None, help="Optional path to write a JSON summary.")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all (layer, offset) cells instead of a single one.")
    parser.add_argument("--layer", type=int, default=9,
                        help="Layer for the single-cell mode (default: L9, the v2 best layer).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset for the single-cell mode (default: k=0).")
    args = parser.parse_args()

    if args.sweep:
        cmd_sweep(args)
    else:
        cmd_pair(args)


if __name__ == "__main__":
    main()
