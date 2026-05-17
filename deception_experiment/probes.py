"""Steps 4 & 5: GPU-vectorized linear probe sweep across (layer, offset).

Deception variant. Identical machinery to ``manipulation_experiment.probes``;
the only differences are the default positive-class subset name
(``deception_prompts``) and the plot titles.

Two sweeps are produced by default:

  - ``all/``                 — every completion. Hack and honest use
                                different system prompts, so a probe
                                here can partially learn "is this a
                                hack-class completion?" rather than
                                token-level lookahead. Use with caution.
  - ``deception_prompts/``   — only completions whose seed used the
                                hack (deception) system prompt. All
                                samples share the same hack persona,
                                so the class-presence confound is
                                removed.
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


def _split_sentences(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test = np.zeros(n, dtype=bool)
    test[perm[:n_test]] = True
    return ~test, test


def _split_by_group(
    groups: np.ndarray, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(groups)
    groups = groups.copy()
    missing = np.where(groups < 0)[0]
    if len(missing):
        start = int(groups.max()) + 1 if (groups >= 0).any() else 0
        groups[missing] = np.arange(start, start + len(missing))

    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    n_test_groups = max(1, int(round(len(unique) * test_frac)))
    test_groups = set(unique[perm[:n_test_groups]].tolist())

    test = np.array([g in test_groups for g in groups], dtype=bool)
    train = ~test
    if train.sum() == 0 or test.sum() == 0:
        return _split_sentences(n, test_frac, seed)
    return train, test


def _build_offset_tensors(
    acts: list[torch.Tensor],
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
        x = a[:, :valid, :].to(torch.float32).transpose(0, 1).contiguous()
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


def _best_threshold_f1(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    if len(set(y_true.tolist())) < 2:
        return 0.5, float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.maximum(denom, 1e-12), 0.0)
    if len(f1) == 0:
        return 0.5, float("nan")
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def _train_probes_batched(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
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
        "W": None, "b": None,
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
        "W": W.detach().cpu(), "b": b.detach().cpu(),
    }


def sweep_layers_and_offsets(
    activations_path: str,
    labels_path: str,
    out_dir: str,
    max_offset: int = 10,
    test_frac: float = 0.2,
    seed: int = 0,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    prompt_filter: str | None = None,
    highlight_layers: list[int] | None = None,
) -> dict:
    data = torch.load(activations_path, weights_only=False)
    acts: list[torch.Tensor] = data["activations"]
    prompt_kinds: list[str] = data.get("prompt_kinds", ["hack"] * len(acts))
    prompt_ids_list: list[int] = data.get("prompt_ids", list(range(len(acts))))
    prompt_ids = np.asarray(prompt_ids_list, dtype=int)
    n_layer_stack = int(data["n_layer_stack"])
    with open(labels_path) as f:
        labels: list[list[int]] = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_keep = np.ones(len(acts), dtype=bool)
    n_missing = 0
    for sid, lab in enumerate(labels):
        if len(lab) == 0 or any(x == -1 for x in lab):
            sentence_keep[sid] = False
            n_missing += 1
    if n_missing:
        print(f"  dropping {n_missing} samples with missing labels (labeler failures)")
    if prompt_filter is not None:
        mask = np.array([k == prompt_filter for k in prompt_kinds], dtype=bool)
        sentence_keep = sentence_keep & mask
        if sentence_keep.sum() == 0:
            raise RuntimeError(f"no sentences match prompt_filter={prompt_filter!r}")

    n_unique_prompts = int(len(np.unique(prompt_ids[sentence_keep])))
    print(
        f"sweep [{prompt_filter or 'all'}]  device={device}  "
        f"{n_layer_stack} layers  offsets 0..{max_offset}  "
        f"n_sentences={int(sentence_keep.sum())}/{len(acts)} "
        f"across {n_unique_prompts} unique prompts  neg_per_pos={neg_per_pos}"
    )

    kept_idx = np.where(sentence_keep)[0]
    tr_kept, te_kept = _split_by_group(prompt_ids[kept_idx], test_frac, seed)
    train_mask = np.zeros(len(acts), dtype=bool)
    test_mask = np.zeros(len(acts), dtype=bool)
    train_mask[kept_idx[tr_kept]] = True
    test_mask[kept_idx[te_kept]] = True
    n_train_prompts = int(len(np.unique(prompt_ids[train_mask])))
    n_test_prompts = int(len(np.unique(prompt_ids[test_mask])))
    print(
        f"  group-aware split: {train_mask.sum()} train / {test_mask.sum()} test samples  "
        f"({n_train_prompts} train-prompts / {n_test_prompts} test-prompts)"
    )
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

    H = acts[0].shape[2] if acts else 0
    W_grid = torch.full((n_layer_stack, len(offsets), H), float("nan"))
    b_grid = torch.full((n_layer_stack, len(offsets)), float("nan"))

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
        if res["W"] is not None:
            W_grid[:, oi, :] = res["W"]
            b_grid[:, oi] = res["b"]

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
    torch.save({"W": W_grid, "b": b_grid, "layers": layers, "offsets": offsets},
               out / "probe_weights.pt")
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
    positive_kind: str = "hack",
    positive_subset_name: str = "deception_prompts",
) -> str:
    if run_name is None:
        np_tag = ("nobal" if neg_per_pos <= 0 else f"np{float(neg_per_pos):g}")
        run_name = f"{np_tag}_mo{max_offset}_ep{num_epochs}"
    print(f"probe sweep run_name = {run_name!r}")

    subsets = [("all", None), (positive_subset_name, positive_kind)]
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
    plt.title(f"Deception probe — {title} across layers × offsets")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"saved plot -> {path}")


# ---------------------------------------------------------------------
# Completion-level (aggregate) probe.
#
# Different from the per-token sweep above: features are the mean of
# activations across the first ``n_prefix`` generated tokens of each
# completion, and the label is per-completion ("does this completion
# contain ANY positive token?"). One row per completion instead of
# one row per token. The deployment question this matches: "given the
# prefix of a trajectory, will it manifest the behavior?"
# ---------------------------------------------------------------------


def _build_completion_level_features(
    acts: list[torch.Tensor],
    labels: list[list[int]],
    n_prefix: int,
    sentence_keep: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    feats, lbls, kept = [], [], []
    for sid, (a, lab) in enumerate(zip(acts, labels)):
        if not sentence_keep[sid]:
            continue
        T = a.shape[1]
        n = min(n_prefix, T)
        if n <= 0:
            continue
        prefix = a[:, :n, :].to(torch.float32)
        feats.append(prefix.mean(dim=1))
        lbls.append(1 if any(x == 1 for x in lab) else 0)
        kept.append(sid)
    if not feats:
        L = acts[0].shape[0] if acts else 0
        H = acts[0].shape[2] if acts else 0
        return torch.zeros(0, L, H), torch.zeros(0), np.array([], dtype=int)
    X = torch.stack(feats, dim=0)
    y = torch.tensor(lbls, dtype=torch.float32)
    return X, y, np.array(kept, dtype=int)


def sweep_completion_level(
    activations_path: str,
    labels_path: str,
    out_dir: str,
    prefix_lengths: tuple[int, ...] = (1, 3, 5, 10, 20),
    test_frac: float = 0.2,
    seed: int = 0,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    prompt_filter: str | None = None,
) -> None:
    data = torch.load(activations_path, weights_only=False)
    acts: list[torch.Tensor] = data["activations"]
    prompt_kinds: list[str] = data.get("prompt_kinds", ["hack"] * len(acts))
    prompt_ids = np.asarray(data.get("prompt_ids", list(range(len(acts)))), dtype=int)
    n_layer_stack = int(data["n_layer_stack"])
    with open(labels_path) as f:
        labels: list[list[int]] = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_keep = np.ones(len(acts), dtype=bool)
    n_missing = 0
    for sid, lab in enumerate(labels):
        if len(lab) == 0 or any(x == -1 for x in lab):
            sentence_keep[sid] = False
            n_missing += 1
    if n_missing:
        print(f"  dropping {n_missing} samples with missing labels")
    if prompt_filter is not None:
        mask = np.array([k == prompt_filter for k in prompt_kinds], dtype=bool)
        sentence_keep = sentence_keep & mask
        if sentence_keep.sum() == 0:
            raise RuntimeError(f"no sentences match prompt_filter={prompt_filter!r}")

    print(
        f"completion-level sweep [{prompt_filter or 'all'}]  device={device}  "
        f"{n_layer_stack} layers  prefix_lengths={list(prefix_lengths)}  "
        f"n_completions={int(sentence_keep.sum())}/{len(acts)}"
    )

    rows: list[dict] = []
    auc_grid = np.full((n_layer_stack, len(prefix_lengths)), np.nan)
    prauc_grid = np.full((n_layer_stack, len(prefix_lengths)), np.nan)
    f1_grid = np.full((n_layer_stack, len(prefix_lengths)), np.nan)

    for ni, n_prefix in enumerate(prefix_lengths):
        X, y, kept = _build_completion_level_features(acts, labels, n_prefix, sentence_keep)
        if X.numel() == 0 or y.numel() == 0:
            continue
        groups = prompt_ids[kept]
        tr_mask, te_mask = _split_by_group(groups, test_frac, seed)
        X_tr = X[tr_mask]; y_tr = y[tr_mask]
        X_te = X[te_mask]; y_te = y[te_mask]
        X_tr_b, y_tr_b = _rebalance_train(X_tr, y_tr, neg_per_pos, seed=seed + ni)
        res = _train_probes_batched(
            X_tr_b, y_tr_b, X_te, y_te,
            device=device, num_epochs=num_epochs,
        )
        for li in range(n_layer_stack):
            auc_grid[li, ni] = res["auc"][li]
            prauc_grid[li, ni] = res["pr_auc"][li]
            f1_grid[li, ni] = res["f1"][li]
            rows.append({
                "layer": li, "n_prefix": int(n_prefix),
                "accuracy": res["accuracy"][li],
                "f1": res["f1"][li],
                "auc": res["auc"][li],
                "pr_auc": res["pr_auc"][li],
                "threshold": res["threshold"][li],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "pos_rate_test": res["pos_rate_test"],
            })
        best_pr = int(np.nanargmax(prauc_grid[:, ni])) if not np.all(np.isnan(prauc_grid[:, ni])) else -1
        best_auc = int(np.nanargmax(auc_grid[:, ni])) if not np.all(np.isnan(auc_grid[:, ni])) else -1
        print(
            f"  n_prefix={n_prefix:>3}  pos_te={res['pos_rate_test']:.2%}  "
            f"best ROC-AUC={np.nanmax(auc_grid[:, ni]):.3f}@L{best_auc}  "
            f"best PR-AUC={np.nanmax(prauc_grid[:, ni]):.3f}@L{best_pr}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(
            {
                "kind": "completion_level",
                "prompt_filter": prompt_filter,
                "prefix_lengths": list(prefix_lengths),
                "layers": list(range(n_layer_stack)),
                "neg_per_pos": neg_per_pos,
                "auc": auc_grid.tolist(),
                "pr_auc": prauc_grid.tolist(),
                "f1": f1_grid.tolist(),
                "rows": rows,
            },
            f,
            indent=2,
        )
    _plot_aggregate_heatmap(
        auc_grid, list(range(n_layer_stack)), list(prefix_lengths),
        out / "heatmap_auc.png", title="ROC-AUC (completion-level)",
    )
    _plot_aggregate_heatmap(
        prauc_grid, list(range(n_layer_stack)), list(prefix_lengths),
        out / "heatmap_pr_auc.png", title="PR-AUC (completion-level)",
    )


def run_completion_level_sweep(
    activations_path: str,
    labels_path: str,
    out_dir: str,
    run_name: str | None = None,
    prefix_lengths: tuple[int, ...] = (1, 3, 5, 10, 20),
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    positive_kind: str = "hack",
    positive_subset_name: str = "deception_prompts",
) -> str:
    if run_name is None:
        pl = "_".join(str(n) for n in prefix_lengths)
        np_tag = ("nobal" if neg_per_pos <= 0 else f"np{float(neg_per_pos):g}")
        run_name = f"{np_tag}_pl{pl}_ep{num_epochs}"
    print(f"completion-level sweep run_name = {run_name!r}")

    base = f"{out_dir}/{run_name}"
    for name, pf in [("all", None), (positive_subset_name, positive_kind)]:
        print(f"\n=== completion-level sweep: {run_name}/{name} ===")
        try:
            sweep_completion_level(
                activations_path=activations_path,
                labels_path=labels_path,
                out_dir=f"{base}/{name}",
                prefix_lengths=prefix_lengths,
                num_epochs=num_epochs,
                neg_per_pos=neg_per_pos,
                prompt_filter=pf,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"subset {name!r} failed: {exc}")

    Path(base).mkdir(parents=True, exist_ok=True)
    with open(Path(base) / "config.json", "w") as f:
        json.dump(
            {
                "kind": "completion_level",
                "run_name": run_name,
                "prefix_lengths": list(prefix_lengths),
                "num_epochs": num_epochs,
                "neg_per_pos": neg_per_pos,
                "seed": seed,
            },
            f,
            indent=2,
        )
    return run_name


def _plot_aggregate_heatmap(grid, layers, prefix_lengths, path, title):
    plt.figure(figsize=(max(6.0, 0.6 * len(prefix_lengths) + 3),
                        max(4.0, 0.3 * len(layers) + 2)))
    plt.imshow(
        grid, aspect="auto", origin="lower",
        extent=(-0.5, len(prefix_lengths) - 0.5,
                layers[0] - 0.5, layers[-1] + 0.5),
        vmin=0.0, vmax=1.0, cmap="viridis",
    )
    plt.colorbar(label=title)
    plt.xticks(range(len(prefix_lengths)), [str(n) for n in prefix_lengths])
    plt.yticks(layers)
    plt.xlabel("prefix length N  (mean-pool of first N completion tokens)")
    plt.ylabel("transformer layer  (0 = embeddings)")
    plt.title(f"Deception probe — {title} (per-completion)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"saved plot -> {path}")


# ---------------------------------------------------------------------
# Per-category sweep helper.
#
# Filters the corpus by a label-stratum (claim_category for deception,
# tactic for manipulation, sandbag_mode for sandbag) and runs the
# existing token-level sweep on each stratum separately. Used to
# answer "is the L9 signal uniform across categories or category-
# specific?". Imports the corpus's metadata to enumerate strata.
# ---------------------------------------------------------------------


def sweep_per_stratum(
    activations_path: str,
    labels_path: str,
    corpus_path: str,
    stratum_field: str,
    out_dir: str,
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    positive_kind: str = "hack",
) -> None:
    with open(corpus_path) as f:
        corpus = json.load(f)
    records = corpus["records"]
    strata = sorted({r.get(stratum_field, "") for r in records if r.get(stratum_field)})
    print(
        f"per-stratum sweep: stratum_field={stratum_field!r}  "
        f"strata={strata}"
    )
    if not strata:
        raise RuntimeError(
            f"no strata found on stratum_field={stratum_field!r}; "
            f"corpus records have keys {list(records[0].keys()) if records else []}"
        )

    data = torch.load(activations_path, weights_only=False)
    prompt_ids_list = data.get("prompt_ids", list(range(len(data["activations"]))))
    prompt_ids = np.asarray(prompt_ids_list, dtype=int)
    prompt_kinds = data.get("prompt_kinds", ["hack"] * len(data["activations"]))

    record_by_prompt_id: dict[int, dict] = {}
    for r in records:
        pid = r.get("prompt_id")
        if pid is not None:
            record_by_prompt_id[int(pid)] = r

    for stratum in strata:
        keep_pids = {
            pid for pid, r in record_by_prompt_id.items()
            if r.get(stratum_field) == stratum
        }
        if not keep_pids:
            continue
        keep_mask = np.array(
            [(pid in keep_pids) and (kind == positive_kind)
             for pid, kind in zip(prompt_ids, prompt_kinds)],
            dtype=bool,
        )
        if keep_mask.sum() < 10:
            print(f"  stratum {stratum!r}: only {keep_mask.sum()} hack completions, skipping")
            continue
        sub_out = f"{out_dir}/{stratum}"
        print(f"\n--- stratum {stratum!r}: {keep_mask.sum()} hack completions ---")
        try:
            sweep_layers_and_offsets_filtered(
                activations_path=activations_path,
                labels_path=labels_path,
                out_dir=sub_out,
                max_offset=max_offset,
                num_epochs=num_epochs,
                neg_per_pos=neg_per_pos,
                seed=seed,
                sentence_mask=keep_mask,
                stratum_name=stratum,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  stratum {stratum!r} failed: {exc}")


def sweep_layers_and_offsets_filtered(
    activations_path: str,
    labels_path: str,
    out_dir: str,
    max_offset: int,
    num_epochs: int,
    neg_per_pos: float,
    seed: int,
    sentence_mask: np.ndarray,
    stratum_name: str,
) -> None:
    """Same as sweep_layers_and_offsets but takes a pre-computed sentence_keep mask."""
    data = torch.load(activations_path, weights_only=False)
    acts: list[torch.Tensor] = data["activations"]
    prompt_ids_list = data.get("prompt_ids", list(range(len(acts))))
    prompt_ids = np.asarray(prompt_ids_list, dtype=int)
    n_layer_stack = int(data["n_layer_stack"])
    with open(labels_path) as f:
        labels: list[list[int]] = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_keep = sentence_mask.copy()
    for sid, lab in enumerate(labels):
        if len(lab) == 0 or any(x == -1 for x in lab):
            sentence_keep[sid] = False
    if sentence_keep.sum() == 0:
        raise RuntimeError("sentence_mask kept nothing after dropping missing-label samples")

    kept_idx = np.where(sentence_keep)[0]
    tr_kept, te_kept = _split_by_group(prompt_ids[kept_idx], 0.2, seed)
    train_mask = np.zeros(len(acts), dtype=bool)
    test_mask = np.zeros(len(acts), dtype=bool)
    train_mask[kept_idx[tr_kept]] = True
    test_mask[kept_idx[te_kept]] = True

    offsets = list(range(max_offset + 1))
    layers = list(range(n_layer_stack))
    rows: list[dict] = []

    for k in offsets:
        X_tr, y_tr, X_te, y_te = _build_offset_tensors(
            acts, labels, k, train_mask, test_mask, sentence_keep=sentence_keep
        )
        X_tr_b, y_tr_b = _rebalance_train(X_tr, y_tr, neg_per_pos, seed=seed + k)
        res = _train_probes_batched(X_tr_b, y_tr_b, X_te, y_te, device=device, num_epochs=num_epochs)
        for li in range(n_layer_stack):
            rows.append({
                "layer": li, "offset": k, "stratum": stratum_name,
                "accuracy": res["accuracy"][li], "f1": res["f1"][li],
                "auc": res["auc"][li], "pr_auc": res["pr_auc"][li],
                "threshold": res["threshold"][li],
                "n_train": res["n_train"], "n_test": res["n_test"],
                "pos_rate_test": res["pos_rate_test"],
                "majority_baseline": res["majority_baseline"],
            })

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"stratum": stratum_name, "layers": layers, "offsets": offsets, "rows": rows},
                  f, indent=2)


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
    plt.title(f"Deception probe — {ylabel} vs offset (per layer)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"saved plot -> {path}")


# ---------------------------------------------------------------------
# MLP sanity check.
#
# Runs a small 2-layer MLP on the activations of a SINGLE (layer,
# offset) cell and reports ROC-AUC alongside the existing linear
# probe's number. Tells us whether the misalignment representation is
# linearly accessible at that cell, or whether non-linearity buys
# meaningful additional lift (which would soften the "we found the
# direction" claim).
# ---------------------------------------------------------------------


def run_mlp_sanity_check(
    activations_path: str,
    labels_path: str,
    out_path: str,
    layer: int,
    offsets: tuple[int, ...] = (0, 5),
    hidden: int = 256,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    prompt_filter: str | None = "hack",
) -> None:
    """Train a 2-layer MLP and a linear probe on the same cells; compare AUCs."""
    data = torch.load(activations_path, weights_only=False)
    acts: list[torch.Tensor] = data["activations"]
    prompt_kinds = data.get("prompt_kinds", ["hack"] * len(acts))
    prompt_ids = np.asarray(data.get("prompt_ids", list(range(len(acts)))), dtype=int)
    n_layer_stack = int(data["n_layer_stack"])
    with open(labels_path) as f:
        labels: list[list[int]] = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentence_keep = np.ones(len(acts), dtype=bool)
    for sid, lab in enumerate(labels):
        if len(lab) == 0 or any(x == -1 for x in lab):
            sentence_keep[sid] = False
    if prompt_filter is not None:
        mask = np.array([k == prompt_filter for k in prompt_kinds], dtype=bool)
        sentence_keep = sentence_keep & mask

    kept_idx = np.where(sentence_keep)[0]
    tr_kept, te_kept = _split_by_group(prompt_ids[kept_idx], 0.2, seed)
    train_mask = np.zeros(len(acts), dtype=bool)
    test_mask = np.zeros(len(acts), dtype=bool)
    train_mask[kept_idx[tr_kept]] = True
    test_mask[kept_idx[te_kept]] = True

    results = []
    for k in offsets:
        X_tr, y_tr, X_te, y_te = _build_offset_tensors(
            acts, labels, k, train_mask, test_mask, sentence_keep=sentence_keep
        )
        X_tr_b, y_tr_b = _rebalance_train(X_tr, y_tr, neg_per_pos, seed=seed + k)
        # Slice to just one layer
        X_tr_l = X_tr_b[:, layer, :].to(device)
        X_te_l = X_te[:, layer, :].to(device)
        y_tr_l = y_tr_b.to(device)
        y_te_np = y_te.numpy().astype(int)

        # Linear
        H = X_tr_l.shape[1]
        W_lin = torch.zeros(H, device=device, requires_grad=True)
        b_lin = torch.zeros(1, device=device, requires_grad=True)
        pos_rate = y_tr_l.mean().item()
        pw = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-8)], device=device)
        opt = torch.optim.AdamW([W_lin, b_lin], lr=0.05, weight_decay=1e-3)
        for _ in range(num_epochs):
            logits = X_tr_l @ W_lin + b_lin
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_l, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lin_proba = torch.sigmoid(X_te_l @ W_lin + b_lin).cpu().numpy()
        try:
            lin_auc = float(roc_auc_score(y_te_np, lin_proba))
            lin_prauc = float(average_precision_score(y_te_np, lin_proba))
        except Exception:
            lin_auc = float("nan"); lin_prauc = float("nan")

        # MLP (2-layer)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(H, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        ).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
        for _ in range(num_epochs):
            logits = mlp(X_tr_l).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_l, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            mlp_proba = torch.sigmoid(mlp(X_te_l).squeeze(-1)).cpu().numpy()
        try:
            mlp_auc = float(roc_auc_score(y_te_np, mlp_proba))
            mlp_prauc = float(average_precision_score(y_te_np, mlp_proba))
        except Exception:
            mlp_auc = float("nan"); mlp_prauc = float("nan")

        results.append({
            "layer": layer, "offset": k,
            "n_train": int(X_tr_l.shape[0]), "n_test": int(X_te_l.shape[0]),
            "linear_auc": lin_auc, "linear_pr_auc": lin_prauc,
            "mlp_auc": mlp_auc, "mlp_pr_auc": mlp_prauc,
            "mlp_minus_linear_auc": mlp_auc - lin_auc,
        })
        print(
            f"  L{layer} k={k}: linear ROC={lin_auc:.3f}  MLP ROC={mlp_auc:.3f}  "
            f"(MLP - linear = {mlp_auc - lin_auc:+.3f})"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "layer": layer, "offsets": list(offsets),
            "hidden": hidden, "num_epochs": num_epochs,
            "prompt_filter": prompt_filter, "neg_per_pos": neg_per_pos,
            "rows": results,
        }, f, indent=2)
    print(f"saved -> {out}")


