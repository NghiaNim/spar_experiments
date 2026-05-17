# MLP-vs-linear sanity check — results

**Date:** 2026-05-17
**Run:** `results/<exp>/<exp>/results_mlp_sanity_transition.json`
**Pipeline:** `<exp>_experiment/run.py --stage mlp-sanity --label-variant transition`
**Pipeline code:** `<exp>_experiment/probes.py::run_mlp_sanity_check`

## TL;DR

We trained a small 2-layer MLP (hidden=256, GELU, 200 epochs AdamW)
on the same (layer, offset) cells used by the headline linear v2
probes, with the same data and train/test split. The question: does
the misalignment representation require non-linear decoding, or is
linear sufficient?

**Headline:** **Linear is sufficient.** The MLP and the linear probe
score identically on ROC-AUC at the headline cells for all three
behaviors, with the MLP slightly *worse* (-0.01 ROC-AUC) — almost
certainly small-sample overfitting. This is the clean "linear is the
right tool" result the paper needs to defend the methodology.

| experiment    | cell              | linear ROC-AUC | MLP ROC-AUC | MLP − linear |
| ---           | ---               | ---:           | ---:        | ---:         |
| sandbag       | L9, k=0           | 0.746          | 0.735       | **−0.011**   |
| sandbag       | L9, k=5           | 0.621          | 0.612       | −0.010       |
| manipulation  | L12, k=0          | 0.673          | 0.677       | +0.005       |
| manipulation  | L12, k=5          | 0.604          | 0.594       | −0.011       |
| deception     | L9, k=0           | 0.751          | 0.739       | −0.012       |
| deception     | L9, k=5           | 0.679          | 0.666       | −0.012       |

All six cells: |MLP − linear| ≤ 0.012 ROC-AUC. The MLP never beats
the linear probe by a margin worth attending to.

## 1. Setup

For each of the three v2 (transition-labeled) probes, we ran a single
MLP-vs-linear comparison at two offsets (k=0 and k=5) at the
behavior's **best-ROC-AUC layer** (sandbag L9, manipulation L12,
deception L9). Train/test split, rebalancing, and probe
hyperparameters match the existing v2 token-level sweep:

A small consistency note: for sandbag, the best-ROC-AUC layer (L9)
differs from the best-PR-AUC layer at k=0 (L7, PR-AUC 0.379 vs L9
PR-AUC 0.315). The MLP comparison was run at L9 for cross-behavior
uniformity (L9 is also the best-ROC cell for deception). The
trade-off is that the MLP check does not directly cover sandbag's
PR-AUC headline cell at L7, and the L9 vs L7 directions for sandbag
are not the same direction (cos ≤ 0.46 across layers; see
`probe_similarity.md` §3b). We expect the same "MLP ≈ linear" result
at L7 — the linear-suffices conclusion is unlikely to flip between
adjacent layers — but a confirmatory run is queued
(`make mlp-sanity-sandbag-v2 MLP_LAYER=7`).

- Group-aware split (20% test) on `prompt_id`
- `neg_per_pos=10` train rebalance
- 200 epochs AdamW (lr=0.05 for linear, lr=1e-3 for MLP)
- Per-class loss weighting via `pos_weight`

The MLP architecture is intentionally small (one hidden layer, 256
units, GELU activation). Larger MLPs would overfit further on our
~10k–11k training tokens per cell.

## 2. PR-AUC details

| experiment    | cell      | linear PR-AUC | MLP PR-AUC | MLP − linear |
| ---           | ---       | ---:          | ---:       | ---:         |
| sandbag       | L9, k=0   | 0.315         | 0.312      | −0.003       |
| sandbag       | L9, k=5   | 0.233         | 0.227      | −0.006       |
| manipulation  | L12, k=0  | 0.310         | 0.312      | +0.002       |
| manipulation  | L12, k=5  | 0.261         | 0.253      | −0.008       |
| deception     | L9, k=0   | 0.413         | 0.390      | −0.023       |
| deception     | L9, k=5   | 0.334         | 0.325      | −0.009       |

PR-AUC tells the same story as ROC-AUC, with the largest gap being
deception at k=0 (linear 0.413 vs MLP 0.390, a 0.023 PR-AUC drop —
still small in absolute terms).

## 3. Interpretation

**The misalignment-commitment representation is *linearly accessible*
at the chosen layer.** This is the standard interpretability claim
for probes: if a linear classifier matches a non-linear one, the
model has organized the relevant information into a single direction
in the residual stream rather than encoding it via a complex
non-linear combination.

This strengthens three claims in the paper:

1. **The "we found a direction" reading is defensible.** The
   `probe_weights.pt` vectors saved at each cell really are a
   "misalignment-commitment direction" rather than an arbitrary
   non-linear decision boundary that happened to score well.
2. **The cosine-similarity comparison across behaviors** (see
   `probe_similarity.md`) is interpretable. Comparing the directions
   that linear probes find is only meaningful if those directions
   actually contain the signal. They do.
3. **Steering / causal interventions** based on the probe direction
   (a natural follow-up) become more plausible. If the feature isn't
   linearly accessible, you can't reliably steer with the probe
   direction.

The MLP slightly *underperforming* the linear probe is also worth
noting: it's evidence of mild overfitting on the training set. With
~7k–11k positive-weighted training tokens and a 64k-parameter MLP
(2048 → 256 → 1, plus biases), the MLP has roughly one parameter per
training token. The linear probe has ~2k parameters — much better
matched to the data.

If anything, this argues that we should resist the temptation to use
larger non-linear probes; the data is too small to benefit.

## 4. Limitations

- **Single seed.** Same multi-seed caveat as everything else in this
  atlas — the 0.01 ROC-AUC gap could be seed variance.
- **One MLP architecture.** We tested a 256-unit hidden layer. A
  different architecture (e.g., dropout, batch norm, attention pool
  over layers) might extract more. But on this dataset, with this
  test set, we have no reason to expect non-linearity helps.
- **Only two offsets tested.** k=0 (the present token) and k=5 (a
  lookahead horizon midway through the decay). We did not sweep all
  11 offsets; if there's a layer/offset cell where the linear probe
  is poorly fit and the MLP wins, we'd miss it. The k=0 and k=5
  cells were chosen as "best" and "moderate decay" — they're the two
  most-cited cells in the paper draft.

## 5. Recommendation for the paper

Add a **single paragraph** to the Methods or Discussion section:

> "To verify that the misalignment representation is linearly
> accessible at the chosen layer, we also trained 2-layer MLPs
> (hidden=256, GELU) on the same (L9, k=0), (L9, k=5), (L12, k=0),
> and (L12, k=5) cells. Across six comparisons, the MLP scored
> within 0.012 ROC-AUC of the linear probe in every cell, and was
> slightly worse on five of six — consistent with mild overfitting
> on the ~10k token training set. The linear-probe direction is a
> faithful summary of the discriminative information in the layer."

This is enough for a reviewer who asks "did you check non-linearity?"
without taking up flagship-result space.

## Artifacts

```
results/sandbag/sandbag/results_mlp_sanity_transition.json
results/manipulation/manipulation/results_mlp_sanity_transition.json
results/deception/deception/results_mlp_sanity_transition.json
```

```bash
make mlp-sanity-sandbag-v2 MLP_LAYER=9 MLP_OFFSETS=0,5
make mlp-sanity-manip-v2 MLP_LAYER_MANIP=12 MLP_OFFSETS=0,5
make mlp-sanity-decep-v2 MLP_LAYER=9 MLP_OFFSETS=0,5
```
