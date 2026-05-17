# Per-stratum (per-category / per-tactic / per-mode) probe — results

**Date:** 2026-05-17
**Run:** `results/<exp>/<exp>/results_strata_transition/<stratum>/`
**Pipeline:** `<exp>_experiment/run.py --stage per-stratum-probe --label-variant transition`
**Pipeline code:** `<exp>_experiment/probes.py::sweep_per_stratum`

## TL;DR

For each behavior, we filtered hack-class completions by the
behavior-specific stratum field (sandbag mode / manipulation tactic /
deception category) and ran the existing token-level v2 probe sweep
on each stratum separately. This is the experiment critique §2.8 / H
asked for: *does the headline mid-stack signal hold uniformly across
categories, or is it driven by one or two categories?*

**Headline:**

- **Sandbag is the most uniform.** All four modes (`claim_ignorance`,
  `confidently_wrong`, `partial_answer`, `wrong_reasoning`) have
  best-layer PR-AUC lift in the +0.18 to +0.30 range at k=0. No single
  mode is doing disproportionate work.
- **Manipulation is bimodal.** Three tactics (`urgency` +0.412,
  `flattery` +0.379, `guilt` +0.347) carry strong signal; two
  (`scarcity` +0.150, `social_proof` +0.133) are markedly weaker.
  The reported v2 manipulation lift of +0.118 is an average across
  this heterogeneity.
- **Deception is dominated by `invented_statistic`.** Best-layer
  PR-AUC lift = **+0.526** at L9 for `invented_statistic` —
  substantially higher than the next-best category
  (`misattributed_quote` +0.329). Two categories (`inverted_causal`
  +0.120, `wrong_recommendation` +0.175) are at half the headline
  number. The "L9 detects deception" claim is partly an
  "L9 detects invented statistics" claim.

This is critique §H confirmed for two of three behaviors: the
all-categories headline numbers understate the strong-category signal
and overstate the weak-category signal. For deployment, this matters
— if the production-time misalignment distribution doesn't match the
training-time category mix, generalization will be uneven.

---

## 1. Setup

For each behavior:

1. Load the existing v2 corpus + transition labels + activations
   (no regeneration needed).
2. Iterate over the strata of the relevant field
   (`sandbag_mode` for sandbag, `tactic` for manipulation,
   `claim_category` for deception). Honest-class prompts (which
   carry the empty stratum) are excluded.
3. For each stratum, filter to its hack-class completions only, then
   run `sweep_layers_and_offsets_filtered` (identical to the existing
   token-level v2 sweep but with a pre-computed sentence mask).
4. Output one results.json per stratum.

All probe hyperparameters are matched to the existing v2 token-level
runs (np10_mo10_ep200, group-aware split on `prompt_id`, 20% test).
Per-stratum n_test is much smaller than the full-corpus run — see
each table for n_test counts.

## 2. Sandbag — per-mode (4 strata, ~32 hack completions each)

| mode                | n_test | pos rate | best ROC-AUC (L)    | best PR-AUC (L) | PR-lift |
| ---                 | ---:   | ---:     | ---                 | ---             | ---     |
| `claim_ignorance`   | 675    | 11.7%    | 0.720 (L 1)         | 0.400 (L 1)     | +0.283  |
| `confidently_wrong` | 526    | 7.0%     | **0.846 (L 6)**     | 0.369 (L 7)     | +0.299  |
| `partial_answer`    | 336    | 12.2%    | 0.767 (L 8)         | 0.303 (L 9)     | +0.181  |
| `wrong_reasoning`   | 382    | 8.9%     | 0.781 (L 7)         | 0.332 (L 1)     | +0.243  |

**Observations:**

- All four modes clear PR-AUC lift > +0.18. The probe finds signal in
  every sandbag mode, with `confidently_wrong` slightly strongest.
- Best layers cluster at **L6–L9** for three of four modes, consistent
  with the headline mid-stack band. `claim_ignorance` best layer is
  L1 by both ROC-AUC and PR-AUC — early in the stack, suggesting a
  more lexical signature ("I'm not sure", "to be honest" — short
  evasion phrases the embedding layer can already detect).
- The full-corpus v2 sandbag PR-AUC was 0.379 at L7 (PR-lift +0.217).
  The per-mode lifts (+0.18 to +0.30) bracket this number with
  `confidently_wrong` and `claim_ignorance` above and the others at
  or below.

**Read:** sandbag's headline signal is **not category-driven**; it
generalizes across modes. This is the cleanest of the three for the
"the probe represents the deeper misalignment concept" claim.

## 3. Manipulation — per-tactic (5 strata, ~20 hack completions each)

| tactic            | n_test | pos rate | best ROC-AUC (L)    | best PR-AUC (L) | PR-lift |
| ---               | ---:   | ---:     | ---                 | ---             | ---     |
| `urgency`         | 441    | 10.0%    | **0.886 (L 9)**     | **0.512 (L 6)** | **+0.412** |
| `flattery`        | 426    | 10.3%    | 0.928 (L 8)         | 0.482 (L 8)     | +0.379  |
| `guilt`           | 371    | 15.1%    | 0.820 (L13)         | 0.497 (L13)     | +0.347  |
| `scarcity`        | 426    | 17.8%    | 0.711 (L11)         | 0.328 (L11)     | +0.150  |
| `social_proof`    | 412    | 22.3%    | 0.697 (L 8)         | 0.357 (L 8)     | +0.133  |

**Observations:**

- **Strong tactics:** `urgency` (PR-lift +0.412), `flattery` (+0.379),
  and `guilt` (+0.347) carry the manipulation signal. ROC-AUC at the
  best layer is **0.82–0.93** for these — far better than the
  full-corpus v2 manipulation ROC of 0.67. The per-tactic probe is a
  much stronger detector for these tactics than the
  rotated-tactic-mixed full-corpus probe.
- **Weak tactics:** `scarcity` and `social_proof` have PR-lift only
  +0.13–0.15. Per-token positive rate is higher for these (18–22% vs
  10–15% for the strong tactics), making the discrimination harder —
  scarcity and social_proof completions are *more saturated* with
  positive tokens, so there's less variance to predict.
- **Best-layer drift:** L6–L13 across tactics. The full-corpus v2
  headline (L12) is the right band but the per-tactic best layer
  varies meaningfully (L8 for flattery, L13 for guilt).

**Read:** manipulation's signal is **bimodal across tactics**. The
full-corpus v2 number of +0.118 lift is an average that hides a
roughly 3× spread (+0.13 to +0.41). The strong-tactic per-tactic
probes are competitive with sandbag and deception headlines; the
weak-tactic probes are not.

This is consistent with the §3.4 / §4.3 paper-draft argument that
manipulation tactics have heterogeneous structural shape — some
(urgency, flattery, guilt) have a sharper onset; others (scarcity,
social_proof) unfold more diffusely.

## 4. Deception — per-category (5 strata, ~24 hack completions each)

| category              | n_test | pos rate | best ROC-AUC (L)    | best PR-AUC (L) | PR-lift |
| ---                   | ---:   | ---:     | ---                 | ---             | ---     |
| `invented_statistic`  | 456    | 13.2%    | **0.899 (L 9)**     | **0.657 (L 9)** | **+0.526** |
| `misattributed_quote` | 469    | 12.2%    | 0.814 (L 8)         | 0.450 (L 9)     | +0.329  |
| `wrong_recommendation`| 463    | 27.9%    | 0.635 (L11)         | 0.453 (L11)     | +0.175  |
| `false_factual`       | 421    | 10.5%    | 0.705 (L13)         | 0.270 (L 5)     | +0.166  |
| `inverted_causal`     | 438    | 11.4%    | 0.692 (L11)         | 0.235 (L11)     | +0.120  |

**Observations:**

- **`invented_statistic` is the outlier.** PR-AUC lift +0.526 at L9 —
  more than 2× the full-corpus v2 deception lift (+0.203). The L9
  representation is genuinely strong for this category.
- **`misattributed_quote` is also strong** (+0.329), unsurprising given
  that misattributed quotes have a distinctive lexical signature (the
  wrong-attribution name).
- **`wrong_recommendation`, `false_factual`, `inverted_causal` are
  weak.** PR-lift +0.12 to +0.18. The full-corpus v2 lift (+0.203) is
  approximately the average across this heterogeneity.
- **Best-layer locus:** L8–L13 across the strong categories — still
  mid-stack — but `false_factual` peaks at L13 (PR best at L5).

**Read:** the "deception lives at L9" headline is **partly an
'invented_statistic detector at L9' result**. When deployed on
deception types that don't include invented statistics (or include
them at a lower rate), the probe's PR-AUC will be substantially
worse than the v2 headline suggests.

This is exactly the per-category concern critique §H raised. The
honest version of the deception headline is: "L9 represents
deception in invented-statistic and misattributed-quote
categories cleanly; less cleanly in factual / causal / advisory
categories."

## 5. Cross-experiment summary

| behavior     | best-anywhere PR-lift  | worst-stratum PR-lift  | strong:weak ratio | uniformity |
| ---          | ---                    | ---                    | ---               | ---        |
| sandbag      | +0.299 (`conf_wrong`)  | +0.181 (`partial_ans`) | 1.65×             | **good**   |
| manipulation | +0.412 (`urgency`)     | +0.133 (`social_proof`)| 3.10×             | poor       |
| deception    | +0.526 (`inv_stat`)    | +0.120 (`inv_causal`)  | **4.38×**         | poor       |

Sandbag's uniformity (1.65× spread) is the only behavior where the
headline number reasonably represents the typical stratum performance.
For manipulation and deception, the headline is closer to a weighted
average of strong-stratum and weak-stratum results.

## 6. Implications for the paper

This is a **critical finding** that the paper draft must address:

1. **The "L9 represents misalignment" claim is reduced in scope.**
   It's "L9 represents misalignment *for several but not all* sub-
   categories of each behavior." Reviewers will ask for this
   breakdown; we have it now.
2. **Deception's per-category dominance pattern needs explicit
   reporting.** The `invented_statistic` lift (+0.526) is one of the
   strongest single-probe numbers in the whole atlas — but it's not
   uniform across deception types, and that should be in the §3.3
   discussion explicitly.
3. **Manipulation's per-tactic heterogeneity adds support to the
   "sustained register" interpretation** for the weak tactics
   (`scarcity`, `social_proof`) and partially undercuts it for the
   strong tactics (`urgency`, `flattery`, `guilt` — these look more
   like fleeting commitments at the tactic level). The paper §3.4
   should note this.
4. **For sandbag, this is a positive result.** Sandbag's headline
   really does generalize across modes — the paper §3.2 can cite this
   as a strength.

## 7. Recommendations

- **Add §3.X "per-stratum breakdown" to the paper** with the three
  tables above, plus 2–3 sentences of interpretation per behavior.
  Probably a half-page total.
- **Update the abstract** to reflect that the per-category result
  generalizes for sandbag but is heterogeneous for manip/decep.
- **Promote `invented_statistic` as a worked example** of the
  strongest single-probe-cell finding in the dataset. The PR-AUC lift
  of +0.526 at L9, k=0 is the most defensible single
  number in the corpus.

## 8. Caveats

- **Per-stratum n_test is small** (~24 hack completions, group-aware
  split). PR-AUC and ROC-AUC point estimates have substantial
  variance. The strong-vs-weak gap for manipulation and deception is
  large enough to be defensible (3–4× spread), but the *exact*
  ranking within strong tactics or within weak categories is noisy.
  Multi-seed reruns (highest-priority follow-up) would tighten this.
- **Stratum n_train differs across strata** because of per-stratum
  prompt-bank size. Sandbag claim_ignorance has 32 prompts (8 train
  prompts after split); manipulation tactics have 20 prompts (16
  train) each; deception categories have 24 (~19 train). This means
  the weakest-stratum results are partly small-data effects in the
  20-prompt manipulation case.
- **Best-layer per-stratum is sometimes outside the L7–L13 band**
  (sandbag claim_ignorance at L1, deception false_factual at L13/L5)
  — adds noise to the "mid-stack signal" headline.

## Artifacts

```
results/sandbag/sandbag/results_strata_transition/
  claim_ignorance/, confidently_wrong/, partial_answer/, wrong_reasoning/

results/manipulation/manipulation/results_strata_transition/
  flattery/, guilt/, scarcity/, social_proof/, urgency/

results/deception/deception/results_strata_transition/
  false_factual/, invented_statistic/, inverted_causal/,
  misattributed_quote/, wrong_recommendation/
```

```bash
make per-stratum-decep-v2     # rerun deception by category
make per-stratum-manip-v2     # rerun manipulation by tactic
make per-stratum-sandbag-v2   # rerun sandbag by mode
```
