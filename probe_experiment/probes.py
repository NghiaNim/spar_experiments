"""Steps 4 & 5: GPU-vectorized linear probe sweep across (layer, offset).

For a fixed offset k the dataset is the same across layers (same positions,
same labels), so we train all L probes in parallel with a single matmul and
backward pass. This collapses ~200 sequential sklearn fits into ~11 tiny
Adam loops on GPU — roughly 15 minutes -> ~30 seconds on an L4.

For each (layer, offset) we record test-set accuracy, F1 (harm class), and
ROC-AUC. Test split is held out at the *sentence* level so token-adjacent
positions can't leak between train and test.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (N_tr, L, H) / (N_tr,) / (N_te, L, H) / (N_te,) for one offset."""
    X_tr_parts, y_tr_parts, X_te_parts, y_te_parts = [], [], [], []
    for sid, (a, lab) in enumerate(zip(acts, labels)):
        T = a.shape[1]
        if len(lab) != T or offset >= T:
            continue
        valid = T - offset
        # a is [L, T, H]; we want [valid, L, H] aligned to token position t
        x = a[:, :valid, :].to(torch.float32).transpose(0, 1).contiguous()
        y = torch.tensor(lab[offset : offset + valid], dtype=torch.float32)
        if train_mask[sid]:
            X_tr_parts.append(x); y_tr_parts.append(y)
        elif test_mask[sid]:
            X_te_parts.append(x); y_te_parts.append(y)

    def _cat(xs, default_shape):
        return torch.cat(xs, dim=0) if xs else torch.zeros(*default_shape, dtype=torch.float32)

    H = acts[0].shape[2] if acts else 0
    L = acts[0].shape[0] if acts else 0
    X_tr = _cat(X_tr_parts, (0, L, H))
    y_tr = torch.cat(y_tr_parts) if y_tr_parts else torch.zeros(0)
    X_te = _cat(X_te_parts, (0, L, H))
    y_te = torch.cat(y_te_parts) if y_te_parts else torch.zeros(0)
    return X_tr, y_tr, X_te, y_te


# ---------- probe training ----------


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
    """Train L logistic-regression probes simultaneously. Returns per-layer metrics."""
    N_tr, L, H = X_tr.shape
    y_tr_np = y_tr.numpy().astype(int)
    y_te_np = y_te.numpy().astype(int)

    pos_rate_te = float(y_te_np.mean()) if len(y_te_np) else float("nan")
    maj = max(pos_rate_te, 1 - pos_rate_te) if len(y_te_np) else float("nan")

    if N_tr == 0 or y_tr.sum().item() == 0 or y_tr.sum().item() == N_tr or len(y_te) == 0:
        return {
            "accuracy": [float("nan")] * L,
            "f1": [float("nan")] * L,
            "auc": [float("nan")] * L,
            "n_train": int(N_tr),
            "n_test": int(len(y_te)),
            "pos_rate_test": pos_rate_te,
            "majority_baseline": maj,
        }

    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    X_te = X_te.to(device)

    pos_rate = y_tr.mean().item()
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device)

    W = torch.zeros(L, H, device=device, requires_grad=True)
    b = torch.zeros(L, device=device, requires_grad=True)
    optim = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    y_tile = y_tr[:, None].expand(-1, L)  # [N_tr, L]
    for _ in range(num_epochs):
        # logits[n, l] = <X_tr[n, l, :], W[l, :]> + b[l]
        logits = torch.einsum("nlh,lh->nl", X_tr, W) + b[None, :]
        loss = F.binary_cross_entropy_with_logits(
            logits, y_tile, pos_weight=pos_weight, reduction="mean"
        )
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        logits_te = torch.einsum("nlh,lh->nl", X_te, W) + b[None, :]
        proba = torch.sigmoid(logits_te).cpu().numpy()  # [N_te, L]
        preds = (proba > 0.5).astype(int)

    acc_list, f1_list, auc_list = [], [], []
    n_classes_te = len(set(y_te_np.tolist()))
    for l in range(L):
        try: acc = float(accuracy_score(y_te_np, preds[:, l]))
        except Exception: acc = float("nan")
        try: f1 = float(f1_score(y_te_np, preds[:, l]))
        except Exception: f1 = float("nan")
        try: auc = float(roc_auc_score(y_te_np, proba[:, l])) if n_classes_te > 1 else float("nan")
        except Exception: auc = float("nan")
        acc_list.append(acc); f1_list.append(f1); auc_list.append(auc)

    return {
        "accuracy": acc_list,
        "f1": f1_list,
        "auc": auc_list,
        "n_train": int(N_tr),
        "n_test": int(len(y_te_np)),
        "pos_rate_test": pos_rate_te,
        "majority_baseline": maj,
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
    highlight_layers: list[int] | None = None,
) -> dict:
    data = torch.load(activations_path, weights_only=False)
    acts: list[torch.Tensor] = data["activations"]
    n_layer_stack = int(data["n_layer_stack"])
    with open(labels_path) as f:
        labels: list[list[int]] = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"probing on {device}, {n_layer_stack} layers, offsets 0..{max_offset}")

    train_mask, test_mask = _split_sentences(len(acts), test_frac, seed)
    offsets = list(range(max_offset + 1))
    layers = list(range(n_layer_stack))

    acc_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    f1_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    auc_grid = np.full((n_layer_stack, len(offsets)), np.nan)
    maj_by_offset = np.full(len(offsets), np.nan)
    rows: list[dict] = []

    for oi, k in enumerate(offsets):
        X_tr, y_tr, X_te, y_te = _build_offset_tensors(acts, labels, k, train_mask, test_mask)
        res = _train_probes_batched(X_tr, y_tr, X_te, y_te, device=device, num_epochs=num_epochs)
        for li in range(n_layer_stack):
            acc_grid[li, oi] = res["accuracy"][li]
            f1_grid[li, oi] = res["f1"][li]
            auc_grid[li, oi] = res["auc"][li]
            rows.append({
                "layer": li, "offset": k,
                "accuracy": res["accuracy"][li],
                "f1": res["f1"][li],
                "auc": res["auc"][li],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "pos_rate_test": res["pos_rate_test"],
                "majority_baseline": res["majority_baseline"],
            })
        maj_by_offset[oi] = res["majority_baseline"]
        print(
            f"offset {k}: n_tr={res['n_train']}, n_te={res['n_test']}, "
            f"pos={res['pos_rate_test']:.2%}  "
            f"best f1={np.nanmax(f1_grid[:, oi]):.3f} @ layer {int(np.nanargmax(f1_grid[:, oi]))}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(
            {
                "layers": layers,
                "offsets": offsets,
                "accuracy": acc_grid.tolist(),
                "f1": f1_grid.tolist(),
                "auc": auc_grid.tolist(),
                "majority_baseline_by_offset": maj_by_offset.tolist(),
                "rows": rows,
            },
            f,
            indent=2,
        )

    _plot_heatmap(f1_grid, layers, offsets, out / "heatmap_f1.png", title="F1 (harm class)")
    _plot_heatmap(acc_grid, layers, offsets, out / "heatmap_accuracy.png", title="Accuracy")
    _plot_heatmap(auc_grid, layers, offsets, out / "heatmap_auc.png", title="ROC-AUC")

    if highlight_layers is None:
        if len(layers) <= 6:
            highlight_layers = layers
        else:
            picks = np.linspace(0, len(layers) - 1, 6).round().astype(int)
            highlight_layers = [layers[i] for i in picks]
    _plot_lines(f1_grid, layers, offsets, highlight_layers, maj_by_offset,
                out / "f1_vs_offset.png", ylabel="F1 (harm class)")
    _plot_lines(auc_grid, layers, offsets, highlight_layers, maj_by_offset,
                out / "auc_vs_offset.png", ylabel="ROC-AUC")

    return {"layers": layers, "offsets": offsets}


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


def _plot_lines(grid, layers, offsets, highlight_layers, maj_baseline, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for L in highlight_layers:
        idx = layers.index(L)
        plt.plot(offsets, grid[idx], marker="o", label=f"layer {L}")
    if not np.all(np.isnan(maj_baseline)):
        plt.plot(offsets, maj_baseline, linestyle="--", color="gray", label="majority baseline")
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
