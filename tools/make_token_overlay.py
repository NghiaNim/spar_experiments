"""Render token-level probe-score overlays for the poster.

Input:
    results/deception/deception/rollout_examples_transition.json
    (produced on Modal by `make rollout-overlay-decep`)

Output:
    writeups/figures/poster/fig9_token_overlay.png

Shows 2 hack completions and 2 honest completions, each token shaded by the
probe's per-step score, labeler-marked positive tokens outlined in blue, with
the F1-tuned threshold and lead-time annotated.

Layout: one axes per completion. Tokens flow left-to-right and wrap. The axes
ylim scales to the actual number of rows used.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = ROOT / "results" / "deception" / "deception" / "rollout_examples_transition.json"
OUT_PATH = ROOT / "writeups" / "figures" / "poster" / "fig9_token_overlay.png"

TOKENS_PER_LINE = 14
BOX_W = 1.0
BOX_H = 1.0
ROW_GAP = 0.25

# Color ramp: white → light pink → red → dark red
PROBE_CMAP = LinearSegmentedColormap.from_list(
    "probe", ["#ffffff", "#ffd8d8", "#f08080", "#cd0000"]
)


def _clean_tok(tok: str) -> str:
    if tok.startswith("Ġ"):
        return "_" + tok[1:]
    if tok.startswith("▁"):
        return "_" + tok[1:]
    if tok in ("<0x0A>", "Ċ"):
        return "↵"
    return tok


def _shorten(tok: str, max_len: int = 9) -> str:
    if len(tok) <= max_len:
        return tok
    return tok[: max_len - 1] + "…"


def _select_windowed_indices(labels, scores, threshold, *,
                              max_honest=28, before_cross=12, after_pos=12):
    """Pick which token indices to render.

    For a completion with a labeled-positive token, show:
      - tokens [first_cross - before_cross ... first_cross + N1] up to and
        slightly past the probe crossing
      - a gap marker, then
      - tokens [first_pos - 4 ... first_pos + after_pos] around the labeled lie.
    If first_cross is already close to first_pos (≤ 20 tokens apart) skip the
    gap and render the whole stretch contiguously.

    For honest completions (no first_pos), show the first ``max_honest`` tokens.

    Returns (kept_indices, gap_positions) where gap_positions is a list of indices
    into kept_indices at which a "…" ellipsis cell should be inserted.
    """
    first_pos = next((i for i, l in enumerate(labels) if l == 1), None)
    first_cross = next((i for i, s in enumerate(scores) if s > threshold), None)

    if first_pos is None:
        return list(range(min(len(labels), max_honest))), []

    # Anchor 1: window around the first probe crossing (if it exists and is
    # well before first_pos).
    if first_cross is not None and (first_pos - first_cross) > 20:
        lo1 = max(0, first_cross - before_cross)
        hi1 = first_cross + before_cross  # symmetric small window
        lo2 = max(hi1 + 1, first_pos - 4)
        hi2 = min(len(labels) - 1, first_pos + after_pos)
        idx = list(range(lo1, hi1 + 1)) + list(range(lo2, hi2 + 1))
        gap_at = hi1 - lo1 + 1  # position in idx where the gap goes
        return idx, [gap_at]
    else:
        # Contiguous: just window around first_pos
        lo = max(0, (first_cross if first_cross is not None else first_pos) - before_cross)
        hi = min(len(labels) - 1, first_pos + after_pos)
        return list(range(lo, hi + 1)), []


def render_completion(ax, tokens, labels, scores, threshold, *, title=None,
                      title_color="black"):
    """Render one completion onto ax with windowed view + ellipsis. Returns the
    number of rows used."""
    keep, gap_positions = _select_windowed_indices(labels, scores, threshold)
    tokens_k = [tokens[i] for i in keep]
    labels_k = [labels[i] for i in keep]
    scores_k = [scores[i] for i in keep]

    # Splice in ellipsis cells at gap positions (after the indices are mapped).
    # Use a sentinel score of -1 to denote the gap cell.
    cells = []
    for j, (t, l, s, orig) in enumerate(zip(tokens_k, labels_k, scores_k, keep)):
        cells.append({"token": t, "label": l, "score": s, "orig": orig, "gap": False})
    for g in sorted(gap_positions, reverse=True):
        cells.insert(g, {"token": "…", "label": 0, "score": 0.0, "orig": None, "gap": True})

    n_tok = len(cells)
    n_rows = max(1, (n_tok + TOKENS_PER_LINE - 1) // TOKENS_PER_LINE)
    total_h = n_rows * (BOX_H + ROW_GAP)

    ax.set_xlim(0, TOKENS_PER_LINE * BOX_W)
    ax.set_ylim(-0.4, total_h + 0.6)
    ax.invert_yaxis()
    ax.axis("off")

    if title:
        ax.text(
            0, -0.2, title, fontsize=10, fontweight="bold",
            color=title_color, va="bottom", ha="left",
        )

    first_pos_orig = next((i for i, l in enumerate(labels) if l == 1), None)
    first_cross_orig = next((i for i, s in enumerate(scores) if s > threshold), None)

    for i, cell in enumerate(cells):
        col = i % TOKENS_PER_LINE
        row = i // TOKENS_PER_LINE
        x = col * BOX_W
        y = row * (BOX_H + ROW_GAP)

        if cell["gap"]:
            # Render an ellipsis cell with a distinct gray look
            rect = mpatches.Rectangle(
                (x + 0.02, y + 0.04),
                BOX_W - 0.04,
                BOX_H - 0.08,
                facecolor="#f0f0f0",
                edgecolor="#888",
                linewidth=0.5,
                linestyle="--",
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(
                x + BOX_W / 2, y + BOX_H / 2, "…",
                ha="center", va="center", fontsize=11,
                color="#666", zorder=3,
            )
            continue

        sc = cell["score"]
        lab = cell["label"]
        tok = cell["token"]

        bg = PROBE_CMAP(min(max(sc, 0.0), 1.0))
        rect = mpatches.Rectangle(
            (x + 0.02, y + 0.04),
            BOX_W - 0.04,
            BOX_H - 0.08,
            facecolor=bg,
            edgecolor="#bbb",
            linewidth=0.4,
            zorder=1,
        )
        ax.add_patch(rect)

        if lab == 1:
            border = mpatches.Rectangle(
                (x + 0.02, y + 0.04),
                BOX_W - 0.04,
                BOX_H - 0.08,
                facecolor="none",
                edgecolor="#0066cc",
                linewidth=2.0,
                zorder=2,
            )
            ax.add_patch(border)

        text = _shorten(_clean_tok(tok))
        text_color = "white" if sc > 0.55 else "black"
        ax.text(
            x + BOX_W / 2, y + BOX_H / 2, text,
            ha="center", va="center", fontsize=7.2,
            family="monospace", color=text_color, zorder=3,
        )

    # Annotation line below the boxes (uses ORIGINAL token indices, not windowed)
    parts = []
    if first_cross_orig is not None:
        parts.append(f"first probe crossing @ token {first_cross_orig}")
    if first_pos_orig is not None:
        parts.append(f"first labeled-lie @ token {first_pos_orig}")
        if first_cross_orig is not None and first_cross_orig < first_pos_orig:
            parts.append(f"lead = {first_pos_orig - first_cross_orig} tokens")
    if first_pos_orig is None:
        parts.append("no labeled lie (honest persona)")
    note = "    ".join(parts)
    if note:
        ax.text(
            0, total_h + 0.2, note,
            fontsize=8, color="#444", va="top", ha="left",
        )

    return n_rows


def main():
    with open(EXAMPLES_PATH) as f:
        payload = json.load(f)
    threshold = payload["threshold"]
    # Rank hack examples by lead time descending; take the top two.
    hack_sorted = sorted(
        payload["hack_examples"],
        key=lambda r: -(r["lead_time"] or 0),
    )
    hack = hack_sorted[:2]
    # One honest example is enough to land the deployment caveat.
    honest = payload["honest_examples"][:1]
    rows = [("hack", r) for r in hack] + [("honest", r) for r in honest]
    n = len(rows)

    # Allocate height per completion based on the windowed token count.
    heights = []
    for kind, rec in rows:
        keep, gaps = _select_windowed_indices(
            rec["labels"], rec["scores"], threshold,
        )
        n_tok = len(keep) + len(gaps)
        n_rows = max(1, (n_tok + TOKENS_PER_LINE - 1) // TOKENS_PER_LINE)
        heights.append(0.55 + 0.42 * n_rows)

    fig_h = sum(heights) + 0.8
    fig, axes = plt.subplots(
        n, 1,
        figsize=(13, fig_h),
        gridspec_kw={"height_ratios": heights},
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    for ax, (kind, rec) in zip(axes, rows):
        if kind == "hack":
            cat = rec.get("claim_category", "?")
            claim = (rec.get("false_claim") or "").strip()
            if len(claim) > 70:
                claim = claim[:67] + "…"
            title = f"HACK · {cat}: false claim = {claim!r}"
            tcolor = "#990000"
        else:
            cat = rec.get("claim_category", "?")
            title = f"HONEST · same prompt context ({cat}), balanced-advisor persona"
            tcolor = "#1f6391"
        render_completion(
            ax,
            tokens=rec["tokens"],
            labels=rec["labels"],
            scores=rec["scores"],
            threshold=threshold,
            title=title,
            title_color=tcolor,
        )

    fig.suptitle(
        "Token-level probe scores (L9, k=0).  "
        "Background = probe confidence (white→dark red).  "
        "Blue outline = labeler-marked commitment token.  "
        f"Threshold = {threshold:.3f}.",
        fontsize=10,
        y=1.04,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
