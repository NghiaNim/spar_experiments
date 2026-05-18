# Multi-seed reruns — preliminary results

**Date:** 2026-05-17
**Run:** Modal apps `ap-be4k7w2cp` (deception), `ap-bay35q41x` (sandbag),
`ap-by9movij3` (manipulation). 5 seeds (0..4) each, writing to
`results_transition/np10_mo10_ep200_seed{N}/` on each volume.
**Aggregation:** `tools/multi_seed_aggregate.py`, percentile bootstrap
95% CI per (layer, offset) cell.

## TL;DR

| experiment    | seeds loaded | k=0 best L | k=0 mean ROC | k=0 95% CI | L-stability at k=0 | single-seed (was) |
| ---           | :---:        | :---:      | ---:         | ---        | ---                | ---:              |
| sandbag       | 5/5          | **L9**     | 0.735        | [0.711, 0.759] | L9 wins 2/5     | 0.746             |
| manipulation  | 5/5          | **L12**    | 0.699        | [0.681, 0.723] | L12 wins 2/5    | **0.673** (below CI) |
| deception     | 3/5*         | **L9**     | 0.747        | [0.727, 0.760] | L9 wins 2/3     | 0.751             |

\* Deception's multi-seed run hit a Modal `RemoteError` mid-loop at
seed 4 (seed 3's `deception_prompts/` subset is also missing). A
re-run for seeds 3+4 is in flight; the table will tighten when those
land but the qualitative finding will not change (seed-3 partial data
shows L9 ROC = 0.756, within the [0.727, 0.760] CI from 3 seeds).

## Headline implication: "L9 dominates" is a band claim, not a layer claim

For each behavior, the nominal single-seed "best layer" wins in only
**2/5 seeds (40%)** at k=0. Across higher offsets, best-layer stability
is also 1–3 out of 5. The honest read of the paper draft's "L9 / L12
mid-stack" claim is:

- **The L7–L13 mid-stack band is robust across seeds.** No seed
  produces a best layer outside this band for sandbag (range L6–L14
  across seeds) or deception (L7–L10 in the seeds we have); manipulation
  range is L10–L16, slightly wider.
- **The specific best-layer winner is not robust.** A different
  random init produces a different layer-2-best on every other run.
  The paper's "L9 best" / "L12 best" headlines are noise-band picks,
  not signal-strength picks.

The CI widths (0.024–0.080 ROC-AUC across cells and behaviors) are
narrow enough that the *behavior-level* numbers stay defensible. What
changes is the per-layer specificity of the claim.

## Concrete sentence updates

The paper draft's existing single-seed numbers map to multi-seed CIs as follows:

**§3.2 sandbag**
- old: "best ROC-AUC 0.821 @ L15 (v1) → 0.746 @ L9 (v2)"
- new: "v2 best ROC-AUC at the best layer per offset, mean across 5 seeds: 0.735 ± [0.711, 0.759] at L9 (winning 2/5 seeds); best-layer drifts L6–L14 across seeds within the mid-stack band."

**§3.3 deception**
- old: "best ROC-AUC 0.796 @ L7 (v1) → 0.751 @ L9 (v2)"
- new (pending seeds 3,4): "v2 best ROC-AUC: 0.747 ± [0.727, 0.760] at L9 (3 of 5 seeds loaded). Best layer at k=0 is L9 in 2/3 seeds; the third seed picks L10."

**§3.4 manipulation**
- old: "best ROC-AUC 0.813 @ L13 (v1) → 0.673 @ L12 (v2)"
- new: "v2 best ROC-AUC: **0.699** ± [0.681, 0.723] at L12 (2/5 seeds). The previously-reported single-seed 0.673 was below the lower CI bound — manipulation's headline ROC was unluckily low at seed=0."

The manipulation correction is particularly notable: the single-seed
value of 0.673 is **outside** the 95% CI of the multi-seed mean. The
"manipulation v2 lift dropped from v1's +0.226 to +0.118" framing in
the paper was partly an unlucky-seed artifact; the mean v2 ROC is
+0.026 higher than the single-seed number suggested.

Even so, the directional finding holds: v2 still costs manipulation
some lift relative to v1 (where the within-hack ROC was 0.813), and
the within-hack-vs-`all/` asymmetry that gives rise to the
"two signals at different layers" story is unchanged.

## Per-cell tables (selected offsets)

### Sandbag v2, 5/5 seeds

| k    | best L | mean ROC | 95% CI            | L stability    |
| :--: | :---:  | ---:     | ---               | ---            |
| 0    | L9     | 0.735    | [0.711, 0.759]    | L9 (2/5)       |
| 1    | L8     | 0.714    | [0.700, 0.735]    | L12 (1/5)      |
| 2    | L7     | 0.706    | [0.690, 0.723]    | L7 (2/5)       |
| 3    | L7     | 0.688    | [0.655, 0.721]    | L14 (2/5)      |
| 5    | L7     | 0.668    | [0.627, 0.707]    | L12 (1/5)      |
| 10   | L9     | 0.606    | [0.564, 0.646]    | L9 (2/5)       |

### Manipulation v2, 5/5 seeds

| k    | best L | mean ROC | 95% CI            | L stability    |
| :--: | :---:  | ---:     | ---               | ---            |
| 0    | L12    | 0.699    | [0.681, 0.723]    | L12 (2/5)      |
| 1    | L12    | 0.697    | [0.679, 0.715]    | L12 (2/5)      |
| 3    | L12    | 0.650    | [0.630, 0.665]    | L12 (3/5)      |
| 5    | L10    | 0.621    | [0.606, 0.638]    | L11 (2/5)      |
| 10   | L13    | 0.574    | [0.557, 0.587]    | L15 (1/5)      |

### Deception v2, 3/5 seeds (preliminary)

| k    | best L | mean ROC | 95% CI            | L stability    |
| :--: | :---:  | ---:     | ---               | ---            |
| 0    | L9     | 0.747    | [0.727, 0.760]    | L9 (2/3)       |
| 1    | L9     | 0.746    | [0.725, 0.763]    | L9 (2/3)       |
| 3    | L10    | 0.701    | [0.677, 0.716]    | L9 (2/3)       |
| 5    | L10    | 0.682    | [0.669, 0.695]    | L9 (2/3)       |
| 10   | L8     | 0.675    | [0.641, 0.715]    | L8 (2/3)       |

## What this tightens, what stays the same

**Tightened (more defensible after multi-seed):**

- The "mid-stack band" claim is stronger — no seed places the best
  layer outside L6–L16 for any behavior at any offset.
- The "L9 specifically" claim for sandbag and deception holds in
  ~40-67% of seeds — high enough to call L9 the modal choice but
  not "the" winner.
- The manipulation headline number gets a +0.026 ROC bump on average,
  and the L12 winner has 60% stability at k=3 (3/5 seeds).

**Softened (overclaiming under single-seed):**

- The L9-vs-L7 distinction within the mid-stack band for sandbag is
  inside the noise floor.
- The "manipulation v2 lift dropped" framing was partly seed-specific.
  The multi-seed result still shows a v1→v2 lift drop, but smaller
  than the single-seed comparison suggested.

**Unchanged:**

- Within-hack vs `all/` asymmetry for manipulation (the "two signals"
  finding) — this is a structural property of the layers, not a
  seed-noise artifact.
- Cross-experiment cosine-sim near-orthogonality at the headline cell
  (all 3 pairs at ~+0.02, verified in `probe_similarity.md`).
- The decay shape across offsets for each behavior.

## Reproduction

```bash
# Generate 5 seeds × 3 v2 experiments (~3 GPU-hours total)
make multi-seed-all-v2

# Or per-experiment
make multi-seed-sandbag-v2 multi-seed-manip-v2 multi-seed-decep-v2

# Pull per-seed result files (modal volume get is finicky with dirs, so
# this is a per-file loop)
make pull-multi-seed-all

# Local aggregation (needs numpy)
make multi-seed-aggregate
```

Outputs:
- `results/<exp>/<exp>/results_transition/np10_mo10_ep200_seed{0..4}/`
- `results/<exp>/<exp>/multi_seed_aggregate_transition.json`
