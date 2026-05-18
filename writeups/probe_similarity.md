# Probe-direction cosine similarity — results

**Date:** 2026-05-17
**Run:** `tools/probe_similarity.py` reading
`results/<exp>/<exp>/results_transition/np10_mo10_ep200/<subset>/probe_weights.pt`
**Pipeline code:** `tools/probe_similarity.py`

## TL;DR

We saved the linear probe weight tensors from each (layer, offset)
cell of the v2 (transition-labeled) within-hack probes and compared
them pairwise across behaviors. The question: do sandbag,
manipulation, and deception share a common "I'm misaligning"
direction in the residual stream, or do they have three independent
representations that all happen to live in the same mid-stack layer
band?

**Headline:** all three pairwise cross-experiment cosine similarities
at the headline cell (L9, k=0) are at the random-direction noise floor
for H ≈ 2048 (E[cos] = 0 ± 0.022):

- sandbag ↔ deception: **+0.023**
- sandbag ↔ manipulation: **+0.018**
- deception ↔ manipulation: **+0.020**

Across the deception ↔ manipulation (layer × offset) grid (the only
pair we report the full grid for), the **maximum cell is +0.095 at
L12, k=9** — 4.3× the noise SD, small in absolute terms but
statistically distinguishable from chance. The honest phrasing is
therefore: **near-orthogonal at the headline cell across all three
pairs, with small positive partial alignment (up to +0.095) at
upper-mid layers where both probes also read the shared hack-persona
register signature.**

**The genuinely interesting cross-experiment finding is the
within-experiment rotation rate** (see §3a below): the probe
direction at L9 rotates with offset at *essentially the same rate*
in both deception and manipulation. Different absolute directions,
shared rotation geometry. That's the real geometric regularity in
the data.

This implies a weakened version of the "mid-stack convergence"
hypothesis: the three behaviors host **co-located linear decoders**
in the L7–L13 layer band (consistent with — but not establishing — a
shared computational substrate; an orthogonal-alternative explanation
is that mid-stack has high effective rank and supports many
independent linear features). The paper's §4.1 claim should be
reduced accordingly.

---

## 1. Setup

`sweep_layers_and_offsets` was modified (commit `1635fb1`) to save
the trained weight tensor `W` (shape `(L, k, H)` = layers × offsets ×
hidden) and bias `b` (shape `(L, k)`) inside each results subset as
`probe_weights.pt`. After re-running the v2 probe sweep for each
experiment, the weights are pulled locally via `pull-{exp}-v2`.

Cosine similarity between two probe directions is
`a · b / (‖a‖ · ‖b‖)`. We compute this:

- **Cross-experiment at a fixed cell:** sim(W_decep[L,k], W_manip[L,k])
  for each (L, k). Tests whether the two behaviors point the same
  way at that probe location.
- **Within-experiment across cells:** sim(W_exp[L0,0], W_exp[L0,k])
  for k=0..10. Tests how stable a single behavior's probe direction
  is as the offset target slides forward.
- **Within-experiment across layers:** sim(W_exp[9,0], W_exp[L,0])
  for L=0..16. Tests how layer-specific the L9 direction is.

For H ≈ 2048 (the hidden dim), random unit vectors have expected
cosine similarity ~N(0, 1/√H) ≈ 0 ± 0.022. Any reported sim
within ±0.02 is at the noise floor.

## 2. Cross-experiment

### 2a. Headline cell (L9, k=0) — all 3 pairs

| pair                        | cos(W_a, W_b) | ‖W_a‖ | ‖W_b‖ |
| ---                         | ---:          | ---:  | ---:  |
| sandbag ↔ deception         | **+0.023**    | 46.52 | 45.09 |
| sandbag ↔ manipulation      | **+0.018**    | 46.52 | 39.44 |
| deception ↔ manipulation    | **+0.020**    | 45.09 | 39.44 |

All three pairs are **at the random-unit-vector noise floor**
(±0.022 for H=2048). The probes are statistically indistinguishable
from independently chosen directions.

### 2b. Full grid: deception ↔ manipulation (cos(W_decep[L,k], W_manip[L,k]))

(The full layer × offset grid was computed for deception ↔ manipulation
only; the sandbag pairs are at the headline cell only.)

### 2b. Full grid (cos(W_decep[L,k], W_manip[L,k]))

```
       k= 0    k= 1    k= 2    k= 3    k= 4    k= 5    k= 6    k= 7    k= 8    k= 9    k=10
L  0  +0.061  +0.087  +0.055  +0.048  +0.061  +0.041  +0.036  +0.046  +0.055  +0.051  +0.064
L  1  +0.077  +0.069  +0.050  +0.056  +0.046  +0.040  +0.007  +0.020  +0.018  +0.031  +0.020
L  2  +0.030  +0.019  +0.022  +0.040  +0.038  +0.039  +0.037  +0.045  +0.049  +0.047  +0.038
L  3  +0.024  +0.031  +0.042  +0.040  +0.047  +0.069  +0.064  +0.072  +0.062  +0.055  +0.033
L  4  +0.000  +0.020  -0.002  +0.036  +0.039  +0.034  +0.064  +0.062  +0.045  +0.027  +0.049
L  5  +0.010  -0.002  +0.009  +0.017  +0.014  +0.033  +0.044  +0.029  +0.024  +0.012  +0.037
L  6  +0.023  +0.015  +0.042  +0.036  +0.040  +0.049  +0.055  +0.064  +0.080  +0.079  +0.092
L  7  +0.056  +0.064  +0.049  +0.041  +0.032  +0.035  +0.056  +0.056  +0.085  +0.074  +0.064
L  8  +0.036  +0.042  +0.048  +0.043  +0.032  +0.029  +0.030  +0.060  +0.078  +0.094  +0.088
L  9  +0.020  +0.057  +0.045  +0.044  +0.036  +0.044  +0.036  +0.052  +0.067  +0.085  +0.059
L 10  +0.047  +0.050  +0.052  +0.023  +0.029  +0.024  +0.036  +0.038  +0.043  +0.063  +0.068
L 11  +0.048  +0.041  +0.046  +0.034  +0.022  +0.025  +0.024  +0.015  +0.029  +0.045  +0.051
L 12  +0.082  +0.072  +0.069  +0.057  +0.069  +0.062  +0.058  +0.072  +0.094  +0.095  +0.092
L 13  +0.032  +0.025  +0.028  +0.035  +0.041  +0.039  +0.047  +0.047  +0.050  +0.051  +0.040
L 14  +0.039  +0.053  +0.056  +0.049  +0.051  +0.061  +0.057  +0.050  +0.049  +0.054  +0.045
L 15  +0.031  +0.009  +0.026  +0.039  +0.058  +0.080  +0.055  +0.047  +0.037  +0.033  +0.025
L 16  +0.025  +0.014  +0.049  +0.050  +0.067  +0.069  +0.066  +0.036  +0.022  +0.032  +0.017
```

Maximum cosine sim across the entire grid: **+0.095** (L12, k=9).
Minimum: **−0.002** (L4, k=2 and L5, k=1). All values within
−0.005 to +0.095 — strictly inside the "no shared direction"
regime.

The slight positive bias (mean cosine sim across the grid ≈ +0.04)
is plausibly explained by both probes having a small overlap in
non-misalignment-specific tokens (e.g., the model's own register /
persona signature, which both probes are partly reading at L12).
But the magnitude is too small to claim a shared "I'm misaligning"
feature.

## 3. Within-experiment direction stability

### 3a. Across offsets at L9 (the deception headline cell)

cos(W_exp[9,0], W_exp[9,k]) for k=1..10:

| k     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    |
| ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| deception     | +0.911 | +0.801 | +0.715 | +0.642 | +0.564 | +0.515 | +0.455 | +0.415 | +0.354 | +0.292 |
| manipulation  | +0.897 | +0.779 | +0.673 | +0.561 | +0.491 | +0.425 | +0.356 | +0.304 | +0.252 | +0.198 |

**Read:** within a single behavior at a single layer, the probe
direction rotates smoothly as the lookahead horizon increases.
Neighboring cells (k=0 vs k=1) are essentially the same direction
(cos > 0.89); far cells (k=0 vs k=10) point quite differently (cos
≈ 0.2–0.3). This is consistent with the probe at each offset
specializing on the activation features predictive of that specific
lookahead horizon.

The two behaviors' rotation patterns share a qualitative shape (smooth
monotonic decay from cos≈0.9 at k=1 to cos≈0.2–0.3 at k=10), **but the
rates are not identical**. Mean cos at k=3 is 0.69 (decep 0.715, manip
0.673), at k=5 it's 0.53 (decep 0.564, manip 0.491); the gap between
the two trajectories grows from <0.03 at k=1–2 to ~0.10 at k=7–10.
**Manipulation rotates faster than deception** (cos = 0.198 at k=10 vs
0.292) — and this mirrors the lookahead-horizon gap in the ROC-AUC
table (manipulation's per-token ROC decays faster than deception's
across offsets). So behaviors don't share the misalignment direction
*or* the rotation rate, but they share the *qualitative geometry of how
the direction evolves with lookahead*, and the rotation rate tracks the
behavior's ROC-AUC decay.

### 3b. Across layers at k=0 (the headline offset)

cos(W_exp[9,0], W_exp[L,0]) for L=0..16:

| L     | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    | 16    |
| ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| dec   | +0.02 | +0.07 | +0.04 | +0.09 | +0.13 | +0.18 | +0.22 | +0.28 | +0.46 | +1.00 | +0.43 | +0.28 | +0.22 | +0.19 | +0.18 | +0.17 | +0.13 |
| manip | +0.05 | +0.04 | +0.05 | +0.12 | +0.10 | +0.13 | +0.20 | +0.25 | +0.41 | +1.00 | +0.44 | +0.30 | +0.20 | +0.18 | +0.19 | +0.15 | +0.09 |

Both behaviors show a clean "L9 is its own thing" pattern: adjacent
layers L8 and L10 are at cos ≈ 0.43, dropping smoothly to ~0.1 at
the embedding and ~0.13 at the final layer.

**This is a stronger interpretability finding than the cross-
experiment result.** It says the L9 probe direction is genuinely
layer-specific within each behavior. The "best layer" in a probe
sweep is not arbitrary — adjacent layers encode something
substantially different.

## 4. Implications for the paper

The cross-experiment near-orthogonality is the **single most
important new finding** in this round of follow-ups. It directly
constrains the mid-stack convergence story:

- The **layer band** (L7–L13) is shared across behaviors — this is
  the result from the v2 token-level sweeps.
- The **direction within that band** is not shared. All three
  pairwise cosine similarities at the headline cell L9, k=0 are at
  the noise floor (+0.018 to +0.023): the three behaviors point in
  essentially orthogonal directions.
- The right framing is therefore: **mid-stack is a shared
  compute-locus** for misalignment-commitment representations across
  behaviors, but the *features* themselves are behavior-specific.

This is a weaker finding than "there's a unified misalignment
direction in the residual stream" (which would have been a
significantly stronger paper). It is still a real, defensible
interpretability claim about *where* in the network these features
live, and the negative similarity result is a useful constraint on
overclaiming.

## 5. Recommendations

- **Paper §4.1b applies the locus-vs-feature distinction:** the
  mid-stack band is a shared *region of computation*, not a shared
  *direction*. All three pairwise cosines are reported at the
  headline cell.
- **The layer-rotation finding** (the smooth direction shift across
  offsets, with manipulation rotating faster than deception — see
  §3a) is the secondary observation, and matches the smooth decay
  shape in ROC-AUC across offsets.
- **Possible follow-up:** compute the full (layer × offset) grid for
  the two sandbag pairs (currently only the headline cell is
  reported). Not load-bearing for §4.1b but would tighten the claim.

## 6. Caveats

- **Single seed.** Each probe weight tensor is from one training
  run. Different random initializations would give different
  directions — but the *signs* of the results we report (near-zero
  cross-experiment, smooth rotation across offsets, peak at the
  trained layer) should be robust to seed variance because they
  exploit the orthogonality-prior of high-dim random vectors.
- **Probes were trained independently.** Each behavior was trained
  from a different random init seed in disjoint Modal jobs. Some
  "init-direction variance" is expected, but the noise floor for
  H=2048 is ±0.022 — our results (+0.018 to +0.023) are at noise.
- **No magnitude analysis.** The probe weight norms differ
  substantially (sandbag 46.5, deception 45.1, manipulation 39.4 at
  the headline cell). Larger norms typically mean stronger probes
  with cleaner training data. We don't yet have a story for what
  the norm difference implies.

## Artifacts

```
results/sandbag/sandbag/results_transition/np10_mo10_ep200/sandbag_prompts/probe_weights.pt
results/sandbag/sandbag/results_transition/np10_mo10_ep200/all/probe_weights.pt
results/deception/deception/results_transition/np10_mo10_ep200/deception_prompts/probe_weights.pt
results/deception/deception/results_transition/np10_mo10_ep200/all/probe_weights.pt
results/manipulation/manipulation/results_transition/np10_mo10_ep200/manipulation_prompts/probe_weights.pt
results/manipulation/manipulation/results_transition/np10_mo10_ep200/all/probe_weights.pt
results/probe_similarity_L9_k0.json   # 3-way cosine matrix at headline cell
```

```bash
make probe-similarity              # single cell (L9, k=0)
make probe-similarity-sweep         # full (layer, offset) grid
```
