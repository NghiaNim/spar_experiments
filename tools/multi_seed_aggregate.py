"""Aggregate multi-seed probe sweeps into mean ± 95% bootstrap CI per cell.

Reads ``results/<exp>/<exp>/results_transition/np10_mo10_ep200_seed{0..4}/<subset>/results.json``
for an experiment and reports, per (layer, offset) cell:

    PR-AUC   mean ± [low, high]
    ROC-AUC  mean ± [low, high]

Also reports the best-layer-per-offset distribution (which layer "wins" at
each k across seeds), which is the actual reviewer-question: is the
"L9 dominates" claim stable across seeds, or does the winning layer
move around inside the noise band?

Usage (from repo root):

    pip install numpy  # if not installed
    python3 tools/multi_seed_aggregate.py --experiment deception
    python3 tools/multi_seed_aggregate.py --experiment sandbag --subset all
    python3 tools/multi_seed_aggregate.py --experiment manipulation --seeds 0 1 2 3 4 \\
        --out results/manipulation/manipulation/multi_seed_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.stderr.write(
        "multi_seed_aggregate needs numpy locally. Install with: pip install numpy\n"
    )
    sys.exit(2)


SUBSET_BY_EXP = {
    "sandbag": "sandbag_prompts",
    "manipulation": "manipulation_prompts",
    "deception": "deception_prompts",
}


def _bootstrap_ci(values: list[float], n_iter: int = 1000, alpha: float = 0.05,
                  rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Mean + percentile bootstrap CI."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    boots = rng.choice(arr, size=(n_iter, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def _load_seed(exp: str, subset: str, seed: int, label_variant: str = "transition",
               np_per_pos: int = 10, mo: int = 10, ep: int = 200) -> dict | None:
    suffix = "" if label_variant == "full" else f"_{label_variant}"
    rn = f"np{np_per_pos}_mo{mo}_ep{ep}_seed{seed}"
    p = Path(f"results/{exp}/{exp}/results{suffix}/{rn}/{subset}/results.json")
    if not p.exists():
        sys.stderr.write(f"WARNING: missing {p}\n")
        return None
    return json.load(open(p))


def aggregate(exp: str, subset: str, seeds: list[int], label_variant: str = "transition") -> dict:
    rng = np.random.default_rng(0)
    seed_results = [_load_seed(exp, subset, s, label_variant=label_variant) for s in seeds]
    seed_results = [r for r in seed_results if r is not None]
    if not seed_results:
        raise SystemExit(f"no seed results found for {exp}/{subset}")

    layers = seed_results[0]["layers"]
    offsets = seed_results[0]["offsets"]
    # Map (layer, offset) -> list of metrics across seeds
    per_cell: dict[tuple[int, int], dict[str, list[float]]] = {}
    for sr in seed_results:
        for row in sr["rows"]:
            key = (row["layer"], row["offset"])
            d = per_cell.setdefault(key, {"pr_auc": [], "auc": [], "f1": [], "pos_rate_test": []})
            d["pr_auc"].append(row["pr_auc"])
            d["auc"].append(row["auc"])
            d["f1"].append(row["f1"])
            d["pos_rate_test"].append(row["pos_rate_test"])

    agg_rows = []
    for (L, k), d in sorted(per_cell.items()):
        m_pr, lo_pr, hi_pr = _bootstrap_ci(d["pr_auc"], rng=rng)
        m_roc, lo_roc, hi_roc = _bootstrap_ci(d["auc"], rng=rng)
        agg_rows.append({
            "layer": L, "offset": k, "n_seeds": len(d["pr_auc"]),
            "pr_auc_mean": m_pr, "pr_auc_lo": lo_pr, "pr_auc_hi": hi_pr,
            "roc_auc_mean": m_roc, "roc_auc_lo": lo_roc, "roc_auc_hi": hi_roc,
            "pos_rate_mean": statistics.mean(d["pos_rate_test"]) if d["pos_rate_test"] else float("nan"),
        })

    # Per-offset best-layer histogram across seeds
    best_layer_dist = {k: {} for k in offsets}
    for sr in seed_results:
        for k in offsets:
            kcells = [r for r in sr["rows"] if r["offset"] == k]
            if not kcells:
                continue
            best = max(kcells, key=lambda c: c["auc"] if c["auc"] == c["auc"] else -1)
            best_layer_dist[k][best["layer"]] = best_layer_dist[k].get(best["layer"], 0) + 1

    # Mean ROC at offset k across seeds at each layer
    per_offset_best = []
    for k in offsets:
        kcells = [r for r in agg_rows if r["offset"] == k]
        best_roc = max(kcells, key=lambda r: r["roc_auc_mean"] if r["roc_auc_mean"] == r["roc_auc_mean"] else -1)
        best_pr = max(kcells, key=lambda r: r["pr_auc_mean"] if r["pr_auc_mean"] == r["pr_auc_mean"] else -1)
        per_offset_best.append({
            "offset": k,
            "best_roc_mean_L": best_roc["layer"],
            "best_roc_mean": best_roc["roc_auc_mean"],
            "best_roc_lo": best_roc["roc_auc_lo"],
            "best_roc_hi": best_roc["roc_auc_hi"],
            "best_pr_mean_L": best_pr["layer"],
            "best_pr_mean": best_pr["pr_auc_mean"],
            "best_pr_lo": best_pr["pr_auc_lo"],
            "best_pr_hi": best_pr["pr_auc_hi"],
            "best_layer_dist_across_seeds": best_layer_dist[k],
        })

    return {
        "experiment": exp,
        "subset": subset,
        "label_variant": label_variant,
        "seeds": seeds,
        "n_seeds_loaded": len(seed_results),
        "layers": layers,
        "offsets": offsets,
        "rows": agg_rows,
        "per_offset_best": per_offset_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=("sandbag", "manipulation", "deception"))
    parser.add_argument("--subset", default=None,
                        help="Subset directory (default: <exp>_prompts).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--label-variant", default="transition", choices=("full", "transition"))
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    subset = args.subset or SUBSET_BY_EXP[args.experiment]

    out = aggregate(args.experiment, subset, args.seeds, args.label_variant)
    print(f"\n=== Multi-seed aggregate: {args.experiment} / {subset} ({out['n_seeds_loaded']}/{len(args.seeds)} seeds loaded) ===\n")
    print(f"{'k':>3} {'best L (mean ROC)':>18} {'mean ROC':>10} {'[lo, hi]':>20} {'best L stability':>18}")
    for r in out["per_offset_best"]:
        dist = r["best_layer_dist_across_seeds"]
        # Show the mode (most common best layer) and how often it won
        mode_L = max(dist.items(), key=lambda kv: kv[1])
        stab = f"L{mode_L[0]} ({mode_L[1]}/{out['n_seeds_loaded']})"
        print(f"  {r['offset']:>2} {r['best_roc_mean_L']:>18} {r['best_roc_mean']:>10.3f}  "
              f"[{r['best_roc_lo']:.3f}, {r['best_roc_hi']:.3f}]  {stab:>18}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
