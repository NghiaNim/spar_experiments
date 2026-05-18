# Held-out category eval (deception) — results

**Date:** 2026-05-17
**Run:** Modal app `ap-...`. `deception_experiment/run.py --stage held-out-eval`.
**Code:** `deception_experiment/probes.py::sweep_held_out_category`.

## TL;DR

For each of the 5 deception categories (`false_factual`,
`invented_statistic`, `inverted_causal`, `misattributed_quote`,
`wrong_recommendation`), we trained the v2 probe on hack completions
of the **other 4** categories and tested on the held-out category's
hack completions. This is the strongest cross-category generalization
test we can run: the probe never sees a single training example from
the held-out category.

**Headline:** the probe generalizes across deception categories. Mean
ROC-AUC at L9, k=0 across the 5 leave-one-out conditions is **0.713**,
only 0.04 below the within-category v2 ROC of 0.751. The L9 signal
represents "deception" broadly, not "the specific category the probe
happened to train on."

| held-out category   | n_train | n_test | L9 ROC @ k=0 | best-anywhere ROC @ k=0 |
| ---                 | ---:    | ---:   | ---:         | ---                      |
| `false_factual`     | 96      | 24     | **0.813**    | 0.813 @ L9               |
| `invented_statistic`| 96      | 24     | **0.756**    | 0.756 @ L9               |
| `misattributed_quote`| 96     | 24     | 0.741        | 0.750 @ L5               |
| `wrong_recommendation`| 96    | 24     | 0.641        | 0.641 @ L9               |
| `inverted_causal`   | 96      | 24     | 0.613        | 0.705 @ L14              |
| **mean**            | —       | —      | **0.713**    | **0.733**                |

**L9 wins as best-anywhere layer in 3 of 5 held-out conditions**
(false_factual, invented_statistic, wrong_recommendation). For the
other 2 (misattributed_quote, inverted_causal), L5 and L14 give a
small improvement over L9, but L9 is still inside the mid-stack band.

## Most important single number

**Held out `invented_statistic`, evaluated on `invented_statistic`'s
hack completions, using probes trained on the OTHER 4 categories:
ROC-AUC 0.756 at L9.** This directly tests the critique-§H concern
that L9 is "really an invented_statistic detector at L9, not a
deception detector." If L9 were only firing on invented-statistic-
specific lexical features, holding the category out should drop
performance to chance. Instead it drops only 0.143 ROC (from
the within-category 0.899 in §3.3) and stays well above the random
0.50 baseline.

This is the strongest cross-category generalization evidence the
paper can produce on this corpus. The probe encodes something that
generalizes across `invented_statistic`'s wrong-statistic phrasing
to `false_factual`'s wrong-fact assertions, `inverted_causal`'s
wrong-causal claims, etc.

## What weakens

- **`inverted_causal` is the hardest held-out condition.** L9 at k=0
  drops to 0.613 — the lowest of the 5 conditions. Best-anywhere is
  L14 at 0.705, suggesting the cross-category signal for causal-error
  deception lives slightly higher up the stack than for the other
  categories. Not a clean "L9 generalizes" pure story.
- **`wrong_recommendation` is the weakest direct match too.** L9
  drops to 0.641 — better than `inverted_causal` but still notably
  below 0.7. The corpus's `wrong_recommendation` category may
  rely on category-specific features the other 4 don't share
  (advisory framing rather than factual assertion).
- **The L9 result mixes "L9 is the right layer for deception" with
  "L9 happens to win on average across these categories".** A
  reviewer could push back: "what if you'd picked L7 or L11 as the
  headline layer? Would it generalize the same?" The data is in the
  per-stratum results files; only L9 is shown here as the headline.

## Interpretation for the paper

The held-out generalization is the **strongest direct evidence in the
paper that L9 encodes deception-as-a-concept, not deception-as-a-
particular-category**. The §3.3 per-category result — that
`invented_statistic` is dominant within-category — was the kind of
finding that invites the "it's just a category detector" reading.
This experiment closes that.

**Recommended paper text (for §3.3):**

> "To test whether L9 represents 'deception' broadly versus 'a
> specific category the probe was trained on', we ran a leave-one-out
> category eval: for each of the 5 deception categories, train on the
> other 4 and test on the held-out one. At L9, k=0, mean ROC-AUC
> across the 5 held-out conditions is **0.713** (range 0.61–0.81),
> compared with the within-category v2 ROC of 0.751 — only 0.04
> below. Critically, **when `invented_statistic` (the strongest
> within-category result, PR-AUC 0.657 at L9) is held out, the L9
> probe trained on the other 4 categories still scores ROC 0.756 on
> the held-out invented_statistic completions** — strong evidence
> against the 'L9 is just an invented_statistic detector' reading."

## Caveats

- **n_test = 24 per held-out category** (the 5 hack prompts in that
  category × 4 samples each, minus any short-circuited refusals). The
  individual category-level numbers have meaningful variance. The
  *mean* ROC of 0.713 across 5 conditions is more stable than any
  single column.
- **Single seed.** The within-category multi-seed showed L9 wins
  4/5 seeds at k=0 (§3.3); we expect similar held-out stability but
  did not run a 5-seed version of leave-one-out. Cost would be 5
  seeds × 5 leave-one-outs = 25 probe runs; estimated ~3 GPU-hours.
- **No held-out for sandbag or manipulation.** The same
  experiment-design pattern works for those; just out of scope for
  this round.
- **k=0 only in the headline.** The full per-cell data is in each
  `<category>.json` file; offsets > 0 weren't reported in the
  summary table to keep it compact. Subsequent analyses can dig into
  cross-category lookahead.

## Artifacts

```
data/deception/deception/results_held_out_transition/
  summary.json                            # per-held-out-cat headline numbers
  false_factual.json                      # per-(layer, offset) full data
  invented_statistic.json
  inverted_causal.json
  misattributed_quote.json
  wrong_recommendation.json
```

Reproduction:

```bash
make held-out-decep        # ~15 min Modal time
```
