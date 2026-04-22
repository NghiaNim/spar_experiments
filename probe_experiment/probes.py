"""Steps 4 & 5: GPU-vectorized linear probe sweep across (layer, offset).

Pipeline for a single (layer, offset) cell — all done in a vectorized way
across layers with one Adam loop per offset:

  1. Concatenate (activation_t, label_{t+k}) pairs across all sentences.
  2. Split by sentence into train / test (whole sentences only → no token
     adjacency leakage).
  3. Rebalance TRAIN only: keep every positive, subsample negatives to
     ``neg_per_pos * n_positives``. Test stays at its natural ~1% rate so
     reported metrics reflect realistic deployment.
  4. Train L logistic probes simultaneously on GPU via AdamW + einsum.
  5. Per layer: tune the decision threshold to maximize train F1, then apply
     that threshold to the natural test set.
  6. Report accuracy-at-tuned-threshold, F1-at-tuned-threshold, ROC-AUC, and
     PR-AUC (average precision). PR-AUC is the imbalance-aware headline
     metric; F1 is the practical decision-quality metric.

Two sweeps are produced by default:
  - ``all/``            — every completion (~1-2% positive rate)
  - ``harm_prompts/``   — only completions whose seed was harm-inducing
                          (positive rate usually several times higher, so F1
                          and PR-AUC become much more informative).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------- data prep ----------


def _split_sentences(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test = np.zeros(n, dtype=bool)
    test[perm[:n_test]] = True
    return ~test, test


def _build_offset_tensors(
    acts: list[torch.Tensor],       # each [L, T_c, H] fp16
    labels: list[list[int]],
    offset: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    sentence_keep: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_tr_parts, y_tr_parts, X_te_parts, y_te_parts = [], [], [], []
    for sid, (a, lab) in enumerate(zip(acts, labels)):
        if sentence_keep is not None and not sentence_keep[sid]:
            continue
        T = a.shape[1]
        if len(lab) != T or offset >= T:
            continue
        valid = T - offset
        x = a[:, :valid, :].to(torch.float32).transpose(0, 1).contiguous()  # [valid, L, H]
        y = torch.tensor(lab[offset : offset + valid], dtype=torch.float32)
        if train_mask[sid]:
            X_tr_parts.append(x); y_tr_parts.append(y)
        elif test_mask[sid]:
            X_te_parts.append(x); y_te_parts.append(y)

    H = acts[0].shape[2] if acts else 0
    L = acts[0].shape[0] if acts else 0

    def _cat(parts, fallback_shape):
        return torch.cat(parts, dim=0) if parts else torch.zeros(*fallback_shape, dtype=torch.float32)

    X_tr = _cat(X_tr_parts, (0, L, H))
    y_tr = torch.cat(y_tr_parts) if y_tr_parts else torch.zeros(0)
    X_te = _cat(X_te_parts, (0, L, H))
    y_te = torch.cat(y_te_parts) if y_te_parts else torch.zeros(0)
    return X_tr, y_tr, X_te, y_te


def _rebalance_train(
    X_tr: torch.Tensor, y_tr: torch.Tensor, neg_per_pos: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep every positive; subsample negatives to ``neg_per_pos * n_pos``.

    If ``neg_per_pos <= 0`` or there aren't enough positives to rebalance,
    return the inputs unchanged.
    """
    if neg_per_pos <= 0 or y_tr.numel() == 0:
        return X_tr, y_tr
    pos_idx = torch.where(y_tr == 1)[0]
    neg_idx = torch.where(y_tr == 0)[0]
    n_pos = int(pos_idx.numel())
    if n_pos == 0:
        return X_tr, y_tr
    n_neg_keep = min(int(neg_idx.numel()), int(round(neg_per_pos * n_pos)))
    rng = np.random.default_rng(seed)
    neg_perm = rng.permutation(int(neg_idx.numel()))[:n_neg_keep]
    neg_keep = neg_idx[torch.from_numpy(neg_perm).long()]
    keep = torch.cat([pos_idx, neg_keep])
    shuf = rng.permutation(int(keep.numel()))
    keep = keep[torch.from_numpy(shuf).long()]
    return X_tr[keep], y_tr[keep]


# ---------- probe training ----------


def _best_threshold_f1(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Return (best_threshold, best_f1) found along the full PR curve."""
    if len(set(y_true.tolist())) < 2:
        return 0.5, float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns arrays of length n_thresholds + 1 for
    # precision/recall, and n_thresholds for thresholds; drop the trailing
    # point so shapes line up.
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.maximum(denom, 1e-12), 0.0)
    if len(f1) == 0:
        return 0.5, float("nan")
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def _train_probes_batched(
    X_tr: torch.Tensor,   # [N_tr, L, H]
    y_tr: torch.Tensor,   # [N_tr]
    X_te: torch.Tensor,   # [N_te, L, H]
    y_te: torch.Tensor,   # [N_te]
    device: str,
    num_epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
) -> dict:
    N_tr, L, H = X_tr.shape
    y_tr_np = y_tr.numpy().astype(int)
    y_te_np = y_te.numpy().astype(int)

    pos_rate_te = float(y_te_np.mean()) if len(y_te_np) else float("nan")
    maj = max(pos_rate_te, 1 - pos_rate_te) if len(y_te_np) else float("nan")

    nan_layer = [float("nan")] * L
    bad_return = {
        "accuracy": list(nan_layer), "f1": list(nan_layer),
        "auc": list(nan_layer), "pr_auc": list(nan_layer),
        "threshold": list(nan_layer), "train_f1_at_best_thresh": list(nan_layer),
        "n_train": int(N_tr), "n_test": int(len(y_te)),
        "pos_rate_test": pos_rate_te, "majority_baseline": maj,
    }
    if N_tr == 0 or y_tr.sum().item() == 0 or y_tr.sum().item() == N_tr or len(y_te) == 0:
        return bad_return

    X_tr_d = X_tr.to(device); y_tr_d = y_tr.to(device); X_te_d = X_te.to(device)

    pos_rate = y_tr.mean().item()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-8)], device=device)

    W = torch.zeros(L, H, device=device, requires_grad=True)
    b = torch.zeros(L, device=device, requires_grad=True)
    optim = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
    y_tile = y_tr_d[:, None].expand(-1, L)

    for _ in range(num_epochs):
        logits = torch.einsum("nlh,lh->nl", X_tr_d, W) + b[None, :]
        loss = F.binary_cross_entropy_with_logits(
            logits, y_tile, pos_weight=pos_weight, reduction="mean"
        )
        optim.zero_grad(); loss.backward(); optim.step()

    with torch.no_grad():
        proba_tr = torch.sigmoid(torch.einsum("nlh,lh->nl", X_tr_d, W) + b[None, :]).cpu().numpy()
        proba_te = torch.sigmoid(torch.einsum("nlh,lh->nl", X_te_d, W) + b[None, :]).cpu().numpy()

    acc_list, f1_list, auc_list, prauc_list, thresh_list, train_f1_list = [], [], [], [], [], []
    n_classes_te = len(set(y_te_np.tolist()))

    for l in range(L):
        thresh, train_f1 = _best_threshold_f1(y_tr_np, proba_tr[:, l])
        preds = (proba_te[:, l] >= thresh).astype(int)
        try: acc = float(accuracy_score(y_te_np, preds))
        except Exception: acc = float("nan")
        try: f1 = float(f1_score(y_te_np, preds))
        except Exception: f1 = float("nan")
        try: auc = float(roc_auc_score(y_te_np, proba_te[:, l])) if n_classes_te > 1 else float("nan")
        except Exception: auc = float("nan")
        try: prauc = float(average_precision_score(y_te_np, proba_te[:, l])) if n_classes_te > 1 else float("nan")
        except Exception: prauc = float("nan")

        acc_list.append(acc); f1_list.append(f1); auc_list.append(auc)
        prauc_list.append(prauc); thresh_list.append(thresh); train_f1_list.append(train_f1)

    return {
        "accuracy": acc_list, "f1": f1_list, "auc": auc_list, "pr_auc": prauc_list,
        "threshold": thresh_list, "train_f1_at_best_thresh": train_f1_list,
        "n_train": int(N_tr), "n_test": int(len(y_te_np)),
        "pos_rate_test": pos_rate_te, "majority_baseline": maj,
    }


# ---------- top-level sweep ----------


def sweep_layers_and_offsets(
    activations_path: str,
    labels_path: str,
    out_dir: str,
    max_offset: int = 10,
    test_frac: float = 0.2,
    seed: int = 0,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    prompt_filter: str | None = None,   # None | "harm" | "benign"
    highlight_layers: list[int] | None = None,
) -> dict:
    data = torch.load(activations_path, weights_only=False)
    acts: list[torch.Tensor] = data["activations"]
    prompt_kinds: list[str] = data.get("prompt_kinds", ["harm"] * len(acts))
    n_layer_stack = int(data["n_layer_stack"])
    with open(labels_path) as f:
        labels: list[list[int]] = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_keep = np.ones(len(acts), dtype=bool)
    if prompt_filter is not None:
        sentence_keep = np.array([k == prompt_filter for k in prompt_kinds], dtype=bool)
        if sentence_keep.sum() == 0:
            raise RuntimeError(f"no sentences match prompt_filter={prompt_filter!r}")

    print(
        f"sweep [{prompt_filter or 'all'}]  device={device}  "
        f"{n_layer_stack} layers  offsets 0..{max_offset}  "
        f"n_sentences={int(sentence_keep.sum())}/{len(acts)}  neg_per_pos={neg_per_pos}"
    )

    train_mask, test_mask = _split_sentences(len(acts), test_frac, seed)
    # intersect masks with the prompt filter so we only use kept sentences
    train_mask = train_mask & sentence_keep
    test_mask = test_mask & sentence_keep
    offsets = list(range(max_offset + 1))
    layers = list(range(n_layer_stack))

    acc_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    f1_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    auc_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    prauc_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    thresh_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    maj_by_offset = np.full(len(offsets), np.nan)
    pos_rate_by_offset = np.full(len(offsets), np.nan)
    rows: list[dict] = []

    for oi, k in enumerate(offsets):
        X_tr, y_tr, X_te, y_te = _build_offset_tensors(
            acts, labels, k, train_mask, test_mask, sentence_keep=sentence_keep
        )
        X_tr_b, y_tr_b = _rebalance_train(X_tr, y_tr, neg_per_pos, seed=seed + k)
        res = _train_probes_batched(X_tr_b, y_tr_b, X_te, y_te, device=device, num_epochs=num_epochs)
        for li in range(n_layer_stack):
            acc_grid[li, oi] = res["accuracy"][li]
            f1_grid[li, oi] = res["f1"][li]
            auc_grid[li, oi] = res["auc"][li]
            prauc_grid[li, oi] = res["pr_auc"][li]
            thresh_grid[li, oi] = res["threshold"][li]
            rows.append({
                "layer": li, "offset": k,
                "accuracy": res["accuracy"][li],
                "f1": res["f1"][li],
                "auc": res["auc"][li],
                "pr_auc": res["pr_auc"][li],
                "threshold": res["threshold"][li],
                "train_f1_at_best_thresh": res["train_f1_at_best_thresh"][li],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "pos_rate_test": res["pos_rate_test"],
                "majority_baseline": res["majority_baseline"],
            })
        maj_by_offset[oi] = res["majority_baseline"]
        pos_rate_by_offset[oi] = res["pos_rate_test"]

        best_layer_f1 = int(np.nanargmax(f1_grid[:, oi])) if not np.all(np.isnan(f1_grid[:, oi])) else -1
        best_layer_pr = int(np.nanargmax(prauc_grid[:, oi])) if not np.all(np.isnan(prauc_grid[:, oi])) else -1
        print(
            f"offset {k}: n_tr={res['n_train']:>5} (bal.)  n_te={res['n_test']:>5}  "
            f"pos_te={res['pos_rate_test']:.2%}  "
            f"best F1={np.nanmax(f1_grid[:, oi]):.3f}@L{best_layer_f1}  "
            f"best PR-AUC={np.nanmax(prauc_grid[:, oi]):.3f}@L{best_layer_pr}  "
            f"best AUC={np.nanmax(auc_grid[:, oi]):.3f}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(
            {
                "prompt_filter": prompt_filter,
                "neg_per_pos": neg_per_pos,
                "layers": layers,
                "offsets": offsets,
                "accuracy": acc_grid.tolist(),
                "f1": f1_grid.tolist(),
                "auc": auc_grid.tolist(),
                "pr_auc": prauc_grid.tolist(),
                "threshold": thresh_grid.tolist(),
                "majority_baseline_by_offset": maj_by_offset.tolist(),
                "pos_rate_by_offset": pos_rate_by_offset.tolist(),
                "rows": rows,
            },
            f,
            indent=2,
        )

    _plot_heatmap(f1_grid, layers, offsets, out / "heatmap_f1.png",
                  title="F1 @ tuned threshold")
    _plot_heatmap(prauc_grid, layers, offsets, out / "heatmap_pr_auc.png",
                  title="PR-AUC (average precision)")
    _plot_heatmap(auc_grid, layers, offsets, out / "heatmap_auc.png",
                  title="ROC-AUC")

    if highlight_layers is None:
        if len(layers) <= 6:
            highlight_layers = layers
        else:
            picks = np.linspace(0, len(layers) - 1, 6).round().astype(int)
            highlight_layers = [layers[i] for i in picks]

    _plot_lines(f1_grid, layers, offsets, highlight_layers,
                baseline=None, path=out / "f1_vs_offset.png",
                ylabel="F1 @ tuned threshold", baseline_label=None)
    _plot_lines(prauc_grid, layers, offsets, highlight_layers,
                baseline=pos_rate_by_offset, path=out / "pr_auc_vs_offset.png",
                ylabel="PR-AUC", baseline_label="random baseline (= pos rate)")
    _plot_lines(auc_grid, layers, offsets, highlight_layers,
                baseline=np.full(len(offsets), 0.5), path=out / "auc_vs_offset.png",
                ylabel="ROC-AUC", baseline_label="random baseline (0.5)")

    return {"layers": layers, "offsets": offsets}


def run_full_sweep(
    activations_path: str,
    labels_path: str,
    out_dir: str,
    run_name: str | None = None,
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
) -> str:
    """Produce two sweeps (full corpus, harm-prompt subset), keyed by run_name.

    Output layout::

        {out_dir}/{run_name}/all/           results.json + *.png
        {out_dir}/{run_name}/harm_prompts/  results.json + *.png

    If ``run_name`` is ``None``, an auto-generated name is used that encodes
    the probe hyperparameters, e.g. ``np10_mo10_ep200`` — so different probe
    configurations can co-exist on the volume without overwriting each other.
    """
    if run_name is None:
        np_tag = ("nobal" if neg_per_pos <= 0 else f"np{float(neg_per_pos):g}")
        run_name = f"{np_tag}_mo{max_offset}_ep{num_epochs}"
    print(f"probe sweep run_name = {run_name!r}")

    subsets = [("all", None), ("harm_prompts", "harm")]
    base = f"{out_dir}/{run_name}"
    for name, pf in subsets:
        print(f"\n=== probe sweep: {run_name}/{name} ===")
        try:
            sweep_layers_and_offsets(
                activations_path=activations_path,
                labels_path=labels_path,
                out_dir=f"{base}/{name}",
                max_offset=max_offset,
                num_epochs=num_epochs,
                neg_per_pos=neg_per_pos,
                prompt_filter=pf,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"subset {name!r} failed: {exc}")

    # manifest at the run-name level so it's easy to see what produced these numbers
    from pathlib import Path
    import json as _json
    Path(base).mkdir(parents=True, exist_ok=True)
    with open(Path(base) / "config.json", "w") as f:
        _json.dump(
            {
                "run_name": run_name,
                "max_offset": max_offset,
                "num_epochs": num_epochs,
                "neg_per_pos": neg_per_pos,
                "seed": seed,
            },
            f,
            indent=2,
        )
    return run_name


# ---------- plots ----------


def _plot_heatmap(grid: np.ndarray, layers, offsets, path: Path, title: str) -> None:
    plt.figure(figsize=(max(6.0, 0.4 * len(offsets) + 3), max(4.0, 0.3 * len(layers) + 2)))
    plt.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=(offsets[0] - 0.5, offsets[-1] + 0.5, layers[0] - 0.5, layers[-1] + 0.5),
        vmin=0.0, vmax=1.0, cmap="viridis",
    )
    plt.colorbar(label=title)
    plt.xticks(offsets)
    plt.yticks(layers)
    plt.xlabel("offset k  (probe predicts label of token t+k)")
    plt.ylabel("transformer layer  (0 = embeddings)")
    plt.title(f"Harm probe — {title} across layers × offsets")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"saved plot -> {path}")


def _plot_lines(
    grid: np.ndarray,
    layers,
    offsets,
    highlight_layers,
    baseline: np.ndarray | None,
    path: Path,
    ylabel: str,
    baseline_label: str | None,
) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for L in highlight_layers:
        idx = layers.index(L)
        plt.plot(offsets, grid[idx], marker="o", label=f"layer {L}")
    if baseline is not None and not np.all(np.isnan(baseline)):
        plt.plot(offsets, baseline, linestyle="--", color="gray",
                 label=baseline_label or "baseline")
    plt.xlabel("offset k  (probe predicts label of token t+k)")
    plt.ylabel(ylabel)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.title(f"Harm probe — {ylabel} vs offset (per layer)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"saved plot -> {path}")
