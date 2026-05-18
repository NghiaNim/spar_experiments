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

| mode                | n_test | pos    | L0 ROC | L0 PR | ROC @ best-PR layer (L)   | best PR (L)    | PR-lift over baseline | L0→best PR gain |
| ---                 | ---:   | ---:   | ---:   | ---:  | ---            | ---            | ---:                  | ---:            |
| `claim_ignorance`   | 675    | 11.7%  | 0.638  | 0.268 | 0.720 (**L 1**)| **0.400 (L 1)**| +0.283                | +0.132          |
| `confidently_wrong` | 526    | 7.0%   | 0.553  | 0.129 | **0.827 (L 7)**| 0.369 (L 7)    | +0.299                | **+0.240**      |
| `partial_answer`    | 336    | 12.2%  | 0.523  | 0.146 | 0.755 (L 9)    | 0.303 (L 9)    | +0.181                | +0.156          |
| `wrong_reasoning`   | 382    | 8.9%   | 0.652  | 0.192 | 0.747 (**L 1**)| 0.332 (**L 1**)| +0.243                | +0.139          |

**Observations:**

- All four modes clear PR-AUC lift > +0.18 over baseline. The probe finds signal in every sandbag mode.
- **Best-layer pattern is heterogeneous.** `confidently_wrong` and `partial_answer` peak at L7 / L9 (mid-stack), consistent with the headline band. **`claim_ignorance` AND `wrong_reasoning` both peak at L1** — the first transformer block. This is the lexical-signature explanation: opening phrases like "I'm not sure", "to be honest", "let me think" are short and the embedding layer + L1 can already detect them. Their L0 ROC-AUCs (0.638 / 0.652) are the highest of any sandbag mode, confirming this.
- **Contextual gain (L0 → L_best PR-AUC) is largest for `confidently_wrong` (+0.240)** — where the wrong-value claim is genuinely a contextual judgment that embeddings can't catch — and smallest for `claim_ignorance` (+0.132), where most of the discrimination is already at L0.
- The full-corpus v2 sandbag PR-AUC was 0.379 at L7 (PR-lift +0.217). The per-mode lifts (+0.18 to +0.30) bracket this number.

**Read:** sandbag's per-mode PR-lift range is the tightest of the three behaviors (1.65×), but the *best layer* varies substantially (L1, L7, L9). Two of four modes are lexical signatures at L1, not mid-stack semantic features. The "uniform across modes" story holds for *PR-lift magnitude* but not for *layer*.

## 3. Manipulation — per-tactic (5 strata, ~20 hack completions each)

| tactic         | n_test | pos    | L0 ROC | L0 PR     | ROC @ best-PR layer (L)    | best PR (L)     | PR-lift over baseline | L0→best PR gain |
| ---            | ---:   | ---:   | ---:   | ---:      | ---             | ---             | ---:                  | ---:            |
| `urgency`      | 441    | 10.0%  | **0.750** | 0.270  | 0.869 (L 6)     | **0.512 (L 6)** | **+0.412**            | +0.241          |
| `flattery`     | 426    | 10.3%  | 0.654  | 0.199     | **0.928 (L 8)** | 0.482 (L 8)     | +0.379                | **+0.283**      |
| `guilt`        | 371    | 15.1%  | 0.643  | 0.248     | 0.820 (L13)     | 0.497 (L13)     | +0.347                | +0.250          |
| `social_proof` | 412    | 22.3%  | 0.544  | 0.265     | 0.697 (L 8)     | 0.357 (L 8)     | +0.133                | +0.092          |
| `scarcity`     | 426    | 17.8%  | 0.634  | **0.311** | 0.711 (L11)     | 0.328 (L11)     | +0.150                | **+0.017**      |

**Observations:**

- **Strong tactics:** `urgency` (PR-lift +0.412), `flattery` (+0.379), and `guilt` (+0.347) carry the manipulation signal. ROC-AUC at the best layer is **0.82–0.93**. **The contextual gain is large for `flattery` (+0.283 from L0 to L_best PR)** — the cleanest "genuinely contextual" result of any manipulation tactic. `urgency` and `guilt` also have substantial gain but with already-elevated L0 baselines (the words "today only", "now", "deserve better" carry direct lexical signature).
- **`scarcity` is essentially a pure lexical detector.** L0 PR-AUC (0.311) is *higher than the best-cell PR-AUC at any layer except L10 (0.319) and L11 (0.328)*; only 2 of 17 layers beat embeddings, by 0.008 and 0.017 respectively. The L0 → L_best gain is only +0.017. **The "scarcity probe at L11" is essentially the embedding-layer detector with a marginal contextual refinement.** This is the cleanest "the contextual layers add little" finding in the atlas.
- `social_proof` is a weaker version of the same — L0 PR=0.265, best 0.357, gain +0.092.
- **Best-layer drift:** L6–L13 across tactics. Per-tactic best layer varies meaningfully (L6 for urgency, L8 for flattery/social_proof, L13 for guilt).

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

| category              | n_test | pos    | L0 ROC    | L0 PR | ROC @ best-PR layer (L)    | best PR (L)     | PR-lift over baseline | L0→best PR gain |
| ---                   | ---:   | ---:   | ---:      | ---:  | ---             | ---             | ---:                  | ---:            |
| `invented_statistic`  | 456    | 13.2%  | **0.669** | 0.236 | **0.899 (L 9)** | **0.657 (L 9)** | **+0.526**            | **+0.421**      |
| `misattributed_quote` | 469    | 12.2%  | 0.496     | 0.136 | 0.782 (L 9)     | 0.450 (L 9)     | +0.329                | +0.314          |
| `wrong_recommendation`| 463    | 27.9%  | 0.488     | 0.265 | 0.635 (L11)     | 0.453 (L11)     | +0.175                | +0.189          |
| `false_factual`       | 421    | 10.5%  | 0.455     | 0.102 | 0.684 (L 5)     | 0.270 (L 5)     | +0.166                | +0.168          |
| `inverted_causal`     | 438    | 11.4%  | 0.507     | 0.129 | 0.692 (L11)     | 0.235 (L11)     | +0.120                | +0.105          |

**Observations:**

- **`invented_statistic` is the outlier — but it's also the most lexically distinctive deception category.** PR-AUC lift +0.526 at L9 is more than 2× the full-corpus v2 deception lift (+0.203). The L9 representation IS genuinely strong (largest L0 → L_best gain of any cell in the atlas, at +0.421). But the L0 baseline is also unusually elevated (ROC=0.669, the highest L0 ROC of any deception category) because fabricated numerical phrasing ("97% of researchers", "the most-cited finding") has a specific token-distribution signature. So the headline number sits on top of an already-informative embedding baseline; the reviewer-resistant framing is "L9 detects invented-statistic deception cleanly because (a) embeddings carry the lexical signature and (b) L9 adds a strong contextual refinement."
- **`misattributed_quote` is the cleanest contextual result.** Low L0 PR (0.136), best PR 0.450, gain +0.314 — most of the lift is genuinely contextual. The wrong-attribution name itself doesn't tip the embedding layer; the *fact* of misattribution is a compositional judgment.
- **`wrong_recommendation`, `false_factual`, `inverted_causal` are weak.** PR-lift +0.12 to +0.18 over baseline; gains over L0 +0.10 to +0.19. The full-corpus v2 lift (+0.203) is approximately the average across this heterogeneity.
- **Best-layer locus:** L8–L13 across the strong categories — still mid-stack — but `false_factual` peaks at L5 on PR-AUC (with ROC peaking at L13).

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
