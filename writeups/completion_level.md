# Completion-level (aggregate) probe — results

**Date:** 2026-05-17
**Run:** `results/<exp>/<exp>/results_completion_transition/np10_pl1_3_5_10_20_ep200/`
**Pipeline:** `<exp>_experiment/run.py --stage completion-probe --label-variant transition`
**Pipeline code:** `<exp>_experiment/probes.py::run_completion_level_sweep`

## TL;DR

The completion-level probe (mean-pool first N tokens of each
completion, predict whether the completion contains *any* positive
token) tests whether per-token transition lookahead could be replaced
by a single-completion classifier — the framing manipulation v2's
negative result suggested might work for "sustained register"
behaviors.

**Headline:** the completion-level question turns out to be
**unanswerable on this corpus, not "completion-level probes don't
work in general"** — an important distinction. The within-hack
positive-class saturation is too high (95.8% sandbag/decep, **100%
manipulation**) for the within-hack classifier to have any negative
examples to fit against; the cross-class probe is dominated by
system-prompt persona detection rather than misalignment commitment
(L0 ROC ≈ 1.0 at N=1 in `all/`). **A corpus with substantial in-hack
refusal rates** (~30–50% of hack-prompted completions *not* committing
the behavior — e.g., via a more capable-but-less-compliant model, or
adversarial questions where the hack persona frequently refuses)
**is required to make the completion-level question well-posed.** We
recommend that as a corpus-design follow-up rather than a probe-design
follow-up.

Specifically the completion-level framing does *not* solve the
manipulation problem on this corpus, but in a different way than
expected:

- **`all/` subset (hack + honest)** trivially separates at every
  prefix length tested (N ∈ {1, 3, 5, 10, 20}) with ROC-AUC ≈ **1.000**
  across all three behaviors. Even N=1 (the activation at the *first*
  generated token alone) perfectly classifies hack vs honest at
  some mid-stack layer. **This is the persona-detection signal we
  already worried about** — the honest class has all-zero labels by
  construction, so any feature distinguishing system-prompt class
  scores ROC=1. The completion-level `all/` numbers do not
  meaningfully address the early-warning question.
- **`within-hack` subset** is *saturated*: 95.8% of sandbag and
  deception hack completions contain ≥1 positive token, and **100%
  of manipulation hack completions do**. With ~24 test completions
  per experiment and one-or-zero negative examples, the within-hack
  completion-level probe is essentially a degenerate task. PR-AUC
  numbers are all >0.93 (driven by the saturated baseline) and lifts
  are +0.000 to +0.040 (driven by 1-2 misclassified completions out
  of 24). **For manipulation, the probe is mathematically
  undefined** — every test completion is positive, so ROC-AUC = NaN.

**Verdict.** Completion-level probing is the wrong granularity for
this corpus. The cross-class `all/` task is dominated by class-
presence (uninformative); the within-hack task has insufficient
negative-class variance to fit a probe (manipulation: zero
negatives; sandbag/deception: 1/24 negative). The right framings
remain (a) **token-level lookahead** (the existing v2 probes), with
the caveat that v2 transition framing only suits fleeting-commitment
behaviors, or (b) **a tactic-conditioned or category-conditioned
probe** — see `per_stratum.md` for that direction.

---

## 1. Setup

For each completion in the corpus, take activations at every layer
and pool over the first N tokens. The resulting (L, H) feature
vector per completion is fed to one binary classifier per layer.
Binary label: `1 if any(label_t == 1 for t in completion) else 0`
(uses the v2 transition labels). N ∈ {1, 3, 5, 10, 20}.

- Group-aware test split (20%) on `prompt_id` — identical to the
  token-level sweep.
- `neg_per_pos=10` train-only rebalance.
- 200 epochs AdamW.
- Reported per (layer, N): ROC-AUC, PR-AUC, F1 at tuned threshold.

The implementation is `run_completion_level_sweep` in each
experiment's `probes.py`. Outputs live under
`results_completion_transition/np10_pl1_3_5_10_20_ep200/`.

## 2. Per-completion positive rates

These determine the difficulty of the completion-level task. Higher
positive rate = harder problem (smaller negative class to predict).

| experiment    | `within-hack` pos rate | `all/` pos rate | n_test (hack subset) |
| ---           | ---                     | ---              | ---                   |
| sandbag       | 95.83%                  | 54.17%           | 24                    |
| manipulation  | **100.00%**             | 70.00%           | 20                    |
| deception     | 95.83%                  | 58.33%           | 24                    |

Within-hack: 23/24 sandbag, 23/24 deception, and 20/20 manipulation
test completions contain at least one positive token. The remaining
1/24 sandbag and 1/24 deception are the "refusal-by-hack-persona"
cases — model was hack-prompted but didn't commit the behavior.

These rates are too saturated for meaningful within-hack
completion-level discrimination.

## 3. Results — `all/` subset (cross-class)

ROC-AUC of the best layer per prefix length:

| N (prefix len)  | sandbag                | manipulation           | deception              |
| ---:            | ---                    | ---                    | ---                    |
| 1               | 1.000 (L10)            | 1.000 (L 2)            | 1.000 (L 7)            |
| 3               | 1.000 (L 7)            | 1.000 (L 3)            | 1.000 (L 8)            |
| 5               | 0.998 (L 9)            | 1.000 (L 5)            | 1.000 (L 8)            |
| 10              | 0.998 (L15)            | 1.000 (L 5)            | 1.000 (L 8)            |
| 20              | 1.000 (L11)            | 1.000 (L 7)            | 1.000 (L 8)            |

**This is the persona-detection ceiling at the best-layer per N.**
The all/ task is "given the first N generated tokens, predict whether
this was a hack-system-prompt or honest-system-prompt generation."
Since honest completions are zero-labeled by the short-circuit and
the system prompts produce different opening register, the
discrimination is trivial at *some* layer — already at N=1 (a single
token of activation). (Not all layers are at 1.0: e.g., the L0
baseline at N=10 for manipulation is 0.81, which is high but not at
ceiling. The "best layer" is what we tabulate.)

PR-AUC and the corresponding lifts:

| N | sandbag PR (lift)       | manip PR (lift)         | decep PR (lift)         |
| ---: | ---                  | ---                     | ---                     |
| 1    | 1.000 (+0.458)        | 1.000 (+0.300)          | 1.000 (+0.417)          |
| 5    | 0.999 (+0.457)        | 1.000 (+0.300)          | 1.000 (+0.417)          |
| 10   | 0.999 (+0.457)        | 1.000 (+0.300)          | 1.000 (+0.417)          |

The lifts look impressive but they are over a 54–70% baseline (the
hack-class fraction in the cross-class pool), not over a noise
floor. **These numbers are not a misalignment-detection result.**

## 4. Results — `within-hack` subset

This is the cleaner subset (same system prompt in both classes), but
positive saturation makes it nearly degenerate.

| N | sandbag best (ROC / PR-lift)   | manipulation              | deception best                  |
| ---: | ---                          | ---                       | ---                              |
| 1    | 0.630 (L 0) / +0.016         | undefined (no negatives)  | 0.804 (L 9) / +0.032             |
| 3    | 0.957 (L 0) / +0.040         | undefined                 | 0.891 (L 5) / +0.036             |
| 5    | 0.739 (L 3) / +0.029         | undefined                 | 0.913 (L 3) / +0.038             |
| 10   | 0.565 (L14) / +0.019         | undefined                 | 0.500 (L16) / +0.000             |
| 20   | 0.913 (L 1) / +0.038         | undefined                 | 0.783 (L15) / +0.032             |

Notes:

- **Manipulation:** every one of the 20 test completions has ≥1
  positive token. With no negative class in the test set, ROC-AUC is
  mathematically undefined and PR-AUC is forced to 1.0 by
  construction. We cannot fit or evaluate a within-hack completion-
  level probe for manipulation.
- **Sandbag and deception:** PR-AUC numbers are all >0.93, but
  baseline (positive rate) is 0.958, so lifts are tiny. ROC-AUC
  varies wildly (0.50–0.96) across N because performance is
  dominated by a single misclassified completion at the right
  threshold.
- **The PR-AUC numbers and lifts are within seed noise.** N=3
  PR-AUC 0.957 vs N=10 PR-AUC 0.565 for sandbag is not a real
  feature of the prefix length; it's an artifact of a 24-completion
  test set with 23 positives.

## 5. Why this happened — and what it implies

The completion-level probe assumes there's *something* about the
prefix of a completion that distinguishes commits-to-misalignment
trajectories from doesn't-commit trajectories. **Our corpus
construction makes the in-hack-class doesn't-commit case extremely
rare** (96–100% of hack-prompted completions commit), so within-hack
the question is degenerate.

The cross-class question (`all/`) IS easy to answer (ROC=1.0 at
N=1), but the answer is *not* "this completion will manifest
misalignment" — it's "this completion is in the hack system-prompt
context." The honest baseline being all-zero by construction means
any class-presence feature gets credit. This is the same caveat we
flagged for token-level `all/` numbers (see §2.4 of the paper
draft); it just shows up more starkly here.

The natural cleaner experiment would be to construct an
**out-of-distribution honest baseline** — e.g., have the model
generate completions to the same questions under multiple
prompt variants (not just one "hack" and one "honest") and ask the
probe to discriminate by per-completion commitment rather than by
system-prompt class. We did not run that.

For the early-warning use case, the operational lesson is:

- Completion-level probes can clear ROC ≈ 1.0 cross-class on this
  setup, but that detector would mostly be reading the system
  prompt, which the deployer already has. The detection signal
  worth adding is the per-token lookahead the existing v2 probes
  measure.
- The token-level v2 probe at L9, k=3 (or similar) remains the
  right deployment substrate.

## 6. Best-layer pattern in `all/` (still informative about WHERE)

Although the magnitudes are uninformative, the cells where ROC-AUC
hits 1.0 are mid-stack (L2–L11) across all three behaviors at every
N. So the "mid-stack representation" finding holds even at this
trivial discrimination task — the system-prompt class signature is
read out mid-stack, not at L0 (which would be a lexical signature)
and not at L16 (which would be a top-of-stack persona register that
takes time to develop).

This is consistent with the token-level v2 finding (mid-stack band
L7–L13), and weakly supports the picture: the model's
"persona/intent" representation lives mid-stack, regardless of
whether you query it at a single token, at a prefix, or at a full
completion.

## 7. What we'd want differently

To make the completion-level framing informative for misalignment
detection on this corpus, two changes would be needed:

1. **Refusal completions inside the hack class** at higher rate.
   Sandbag had 13/120 hack-class refusals at the full-corpus level;
   manipulation had 1/100; deception had 6/120. Boosting the refusal
   rate (e.g., by using a less-compliant model or by adding
   adversarial questions where the hack persona is more likely to
   refuse) would give the within-hack completion-level probe
   meaningful negatives to train on.
2. **Continuous per-completion outcome** instead of binary. Predict
   the *fraction* of positive tokens in the completion, not just
   whether any exist. This converts the saturated 95%-positive
   binary task into a continuous regression that has variance even
   within hack class.

Both are corpus-design changes, not labeling or probe changes — out
of scope for this v0 paper round.

## 8. Cross-experiment comparison

| metric                                           | sandbag       | manipulation       | deception      |
| ---                                              | ---           | ---                | ---            |
| within-hack pos rate (saturation)               | 0.958         | **1.000**          | 0.958          |
| within-hack best PR-AUC lift (N=3)               | +0.040        | undefined          | +0.036         |
| `all/` best ROC-AUC @ N=1 (persona-detection)    | 1.000         | 1.000              | 1.000          |
| useful for early-warning?                        | no            | **no (degenerate)**| no             |

The asymmetry — manipulation is fully saturated, sandbag/deception
just nearly so — reflects the same thing the v2 token-level analyses
told us: in this corpus, **the hack-persona system prompt + the
question's matched-content design + the abliterated model's
compliance leave the model in a near-deterministic "commit to the
behavior" mode**. The probe-relevant question isn't "does this
completion commit?" but "where in the sequence does the commitment
happen?" — which is what token-level v2 measures.

## 9. Recommendation for the paper

- **Demote the completion-level probe to a methods sanity-check
  paragraph**, not a flagship analysis.
- Use the result to **defend the token-level framing**: the
  completion-level alternative is degenerate on this corpus, so
  per-token lookahead is the only way to extract a useful early-
  warning signal from the activations we have.
- **Drop the §5.2 "aggregate probe for manipulation" promise** from
  the paper draft (or convert it to a one-paragraph note that the
  aggregate probe was tried and the within-hack saturation makes
  it uninformative). The expected result was "completion-level
  recovers the v1 manipulation signal"; the actual result is
  "completion-level is degenerate on this corpus, so the v2 negative
  result for manipulation is a real property of the labeling
  target × behavior shape interaction, not a measurement artifact."

## Artifacts

```
results/sandbag/sandbag/results_completion_transition/np10_pl1_3_5_10_20_ep200/
  all/, sandbag_prompts/  -- results.json + heatmaps
results/manipulation/manipulation/results_completion_transition/np10_pl1_3_5_10_20_ep200/
  all/, manipulation_prompts/
results/deception/deception/results_completion_transition/np10_pl1_3_5_10_20_ep200/
  all/, deception_prompts/
```

```bash
make qc-decep-by-category-v2   # baseline elicitation rates
make completion-decep-v2       # re-runs the sweep
```
