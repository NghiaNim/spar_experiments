# Mid-stack lookahead probes localize commitment to misaligned behaviors in a small language model

**Draft.** Workshop-paper target (NeurIPS / ICML alignment or interpretability workshop). ~8 pages.

---

## Abstract

We train linear probes on the residual-stream activations of a small open-weights language model (`huihui-ai/Llama-3.2-1B-Instruct-abliterated`) to predict, from the activation at token *t*, the label of token *t+k* for **three core misalignment behaviors** — sandbagging, instrumental deception, and manipulation — with two reward-hacking subtasks (specification gaming and sycophancy) as pilot warmups. We find that (i) every behavior leaves a probe-readable trace in the residual stream **before** the offending token is emitted, with lookahead horizons that vary by behavior class; (ii) the strongest probe sits in the **L7–L13 mid-stack band** of a 17-layer model for most strata of all three core behaviors, but cross-experiment cosine similarity of the trained probe directions at the headline cell is **≈+0.02 (at noise floor)** — so the three behaviors are **co-located linear decoders in a shared layer band**, consistent with (but not establishing) a shared mid-stack substrate; an orthogonal-alternative explanation is that the mid-stack residual stream has high effective rank and supports many independent linear features; (iii) one of the three behaviors — manipulation — fails to benefit from the transition-only relabel, which we argue is diagnostic: transition framing works for *fleeting-commitment* behaviors (sandbag, deception — confirmed) but not for *sustained-register* behaviors (manipulation — the informative counterexample); and (iv) per-stratum breakdowns reveal substantial heterogeneity: deception's signal is partly an `invented_statistic` detector (PR-AUC lift +0.53 there vs +0.12 in the weakest category, but L0 already gets PR-AUC 0.236 there — the most lexically distinctive deception category); manipulation's `scarcity` tactic is essentially a lexical detector (L0 PR-AUC 0.311 is within 0.017 of the best-layer PR-AUC 0.328 — embeddings already capture nearly all the signal); and sandbag's `claim_ignorance` and `wrong_reasoning` modes peak at **L1** (not mid-stack) on PR-AUC. The mid-stack-band finding is therefore stratum-dependent. A 2-layer MLP fitted to the same cells matches the linear probe to within |Δ| ≤ 0.012 ROC-AUC, confirming that linear is the right tool. **Multi-seed reruns (5 seeds × 3 v2 experiments) confirm the band-level claims but reveal that the specific "L9 winner" framing was overstated: deception L9 wins 4/5 seeds (robust); sandbag L9 and manipulation L12 each win only 2/5 (noise-band picks within the mid-stack band).** A **rollout-level early-warning eval** on deception reveals an important caveat: at the F1-tuned threshold the L9 probe catches 70.5% of committed-hack rollouts ≥1 token before the lying token (mean lead 32 tokens, median 24), but **the false-positive rate on honest rollouts is 100%** — the catch rate ≈ FPR at every threshold sweep, so the within-hack probe does not separate the two classes at the rollout level. A **held-out leave-one-category eval** on deception (train on 4/5 categories, test on the 5th) gives mean ROC 0.713 at L9 across 5 held-out conditions — only 0.04 below the within-category 0.751, directly evidencing that L9 represents deception broadly rather than category-specific features. Within-hack offline interpretability is well-supported; cross-class deployment-time detection requires additional work (cross-class training, threshold calibration, smarter aggregation).

## 1. Introduction

A natural target for runtime alignment monitoring is *flagging a model trajectory as misaligned before the offending action is emitted*. Operationally, this requires a function `f(activations_at_t) → P(misalignment in next k tokens)` that is (a) cheap enough to run per-token, (b) detects the misalignment with non-trivial lead time `k > 0`, and (c) is consistent across the kinds of misalignment one wants to catch.

Linear probes on residual-stream activations are a natural candidate for `f`. They are cheap, they have been shown to localize sentiment, factual knowledge, and entity-level features in prior work, and they extract a single direction per layer that is amenable to mechanistic follow-up. The open question is whether they pick up *misalignment-specific lookahead* — does the probe at token *t* fire on representations of "the model is about to lie/cheat/sandbag" rather than on already-emitted misalignment, or on a generic "this is a misaligned-persona context" detector?

This work reports a multi-experiment probe sweep across three core misalignment behaviors (sandbagging, instrumental deception, manipulation) and two reward-hacking pilots (substring specification-gaming, sycophancy) on a small open-weights model, with a deliberate focus on the **layer-and-offset structure** of the probe signal. We aim less to deploy a state-of-the-art detector than to characterize *where in the residual stream and how far ahead of emission* the model represents its commitment to each kind of misalignment, and to expose methodology choices that make these probes appear stronger or weaker than they are.

**Contributions.**

1. A unified pipeline that produces a three-behavior probe atlas on the same base model and the same labeler stack, with two reward-hacking pilots for context.
2. A methodological finding — **transition-only labels** vs naive full-span labels — that resolves a class-presence confound inflating naive PR-AUC for sandbag and deception, and migrates sandbag's best layer from L15 to mid-stack. Manipulation's signal is **already** mid-stack under v1 and does not migrate; deception's signal is **already** mid-stack under v1 (L7) and refines slightly to L9 under v2.
3. A **mid-stack-band** result: across the three core behaviors, the strongest probe sits in the L7–L13 band — *regardless of label variant*, **for most strata**. (Two of four sandbag modes — `claim_ignorance` and `wrong_reasoning` — actually peak at L1, suggesting lexical detection of evasion openers; see §3.2.) But the trained probe *directions* in that band are near-orthogonal across behaviors (cross-experiment cosine sim ≈ 0.02 at the headline cell, with max +0.095 ≈ 4.3× noise SD at upper-mid layers L12–L13 — small but statistically distinguishable from random). The three behaviors are **co-located linear decoders** in this band, not a shared feature.
4. A **negative result** — the transition relabel hurts manipulation — argued as diagnostic of the behavior's *structural shape* (sustained register vs fleeting commitment). A follow-up completion-level probe (mean-pool first N tokens) is degenerate on our corpus because within-hack positive-class saturation is 95–100%; the manipulation v2 negative result is therefore a genuine property of the labeling-target × behavior-shape interaction, not a measurement artifact that a coarser aggregator can fix.
5. A characterization of **lookahead horizon by behavior**, scoped to the three core v2 experiments (deception ≈ 10 tokens, sandbag ≈ 5–6 tokens, manipulation ≈ 3–4 tokens). Sycophancy `hack_prompts/` has the longest horizon (ROC-AUC 0.82 at k=10) but is reported as a pilot.
6. **Per-stratum analysis** revealing strong category-specific dominance for two of three behaviors: deception's within-category signal is partly an `invented_statistic` detector (PR-AUC lift +0.526 there vs +0.120 in `inverted_causal`), and manipulation's signal is bimodal across tactics. However, a **held-out-category leave-one-out eval for deception** (train on 4/5 categories, test on the 5th) gives mean ROC 0.713 at L9 across the 5 held-out conditions, with `invented_statistic` held out still scoring 0.756 — so L9 *does* generalize to unseen deception categories, not just within-category-specific lexical features.
7. A **non-linearity sanity check:** a 2-layer MLP at the headline cells matches the linear probe to within 0.012 ROC-AUC across six (behavior, offset) cells, confirming linear probing is the right methodology — the misalignment representation is linearly accessible from the residual stream at the chosen layer.

## 2. Setup

### 2.1 Model and corpus

All experiments use `huihui-ai/Llama-3.2-1B-Instruct-abliterated` (16 transformer blocks + final embedding output, 17 layer-outputs). The abliterated variant is used in place of the stock instruct model because the stock model self-censors the target behaviors at generation time (e.g., writes PG-13 anger instead of the explicit profanity the probe needs to detect), starving the probe of training signal. We use the abliterated variant uniformly so that all probe results are directly comparable.

For each behavior we construct **matched-prompt pairs**: the same factual question or scenario is shown to both a "hack" persona (incentivized toward the misaligned behavior) and an "honest" persona (matched-length system prompt with no misalignment incentive). The hack persona is rotated across a small bank of *modes/tactics/categories* (Section 3 details per behavior) so that the probe cannot trivially learn one specific lexical pattern. The model **self-generates the corpus** under each persona; we extract residual-stream activations at every completion-position token across every layer.

The substring-explicit reward-hack task is run on `Qwen2.5-7B-Instruct` (29 layer outputs) as a larger-model control; all other results are on the 1B abliterated model.

### 2.2 Three-stage pipeline

1. **Model stage (GPU):** load LM once, run every prompt through left-padded `.generate()`, re-run completions right-padded with `output_hidden_states=True` to extract per-token activations at every layer. Stored to a per-experiment Modal volume.
2. **Label stage (CPU):** ask a stronger model (OpenAI `gpt-5.4-mini` cascading to `gpt-5.4`) to mark which completion tokens are part of the target behavior. Strict JSON-schema response, concurrent labeling, ~16 workers. The labeler returns *indices*, not a binary array, so it never has to count to `k`.
3. **Probe stage (GPU):** for each `(layer, offset k)` cell, train an L2-regularized linear classifier on activations at token *t* with target `y_{t+k}`. All layers per offset are trained in a single vectorized AdamW loop.

### 2.3 Probe correctness scaffolding

Four guards on the probe are deliberate and load-bearing:

- **Group-aware split** on `prompt_id`: all `samples_per_prompt` completions from a given seed prompt land on the same side of the train/test split, preventing per-prompt lexical leak.
- **Train-only rebalance:** positives kept, negatives subsampled to `neg_per_pos × n_pos = 10` on the train side. Test stays at the natural positive rate so reported metrics reflect deployment.
- **Per-probe F1-tuned threshold:** each probe's decision threshold is picked to maximize F1 on its own *training* predictions, applied to the test set.
- **Matched system prompts in both classes:** both the hack and honest personas carry an explicit system prompt. Without this, the cross-class `all/` probe trivially learns "system-prompt present → positive."

### 2.4 Two reported subsets per probe

Every probe sweep reports two filtered subsets:
- **`all/`:** mixed hack + honest. **Caveat:** honest-persona tokens are short-circuited to all-zero labels by construction (honest mistakes ≠ misalignment); this means `all/` reports a cross-class probe on an artificial setup where one class has a zero label vector. Any feature that distinguishes hack-system-prompt from honest-system-prompt gets credit. The probe at L7–L8 for manipulation `all/` reads this class-presence signal, distinct from the register signal that the within-hack probe finds at L11–L13 (§3.4). Treat `all/` PR-AUC with caution.
- **`<positive_class>_prompts/`:** within-hack-class only. Removes the system-prompt-presence confound and provides the cleanest read on token-level lookahead. **Headline numbers in this paper come from this subset.**

Headline metric is **PR-AUC** with the test-set positive rate as the chance floor. ROC-AUC and F1-at-tuned-threshold are reported as corroborating signals.

## 3. Five behaviors

### 3.1 Reward hacking (specification gaming + sycophancy)

*Specification gaming.* Hack persona is told it is scored by a substring-counting grader and instructed to maximize that score; markers are rotated per-prompt over a small bank (`<<APPROVED>>`, `[CERTIFIED]`, `||VERIFIED||`, etc.) so the probe must learn "model is in spec-gaming mode," not a single marker. Two variants:

- **`substring_oneshot`** (1B abliterated, 17 layers): PR-AUC = **1.000** at k=0, 0.987 at k=1, 0.807 at k=3, decaying to 0.21 at k=10. Layer-uniform — the bright band in the heatmap is near-vertical at low offsets, consistent with a largely lexical signal.
- **`substring_explicit`** (Qwen 7B, 29 layers): peak PR-AUC 1.000 at L20, k=0, decaying more slowly (0.36 at k=10). The larger model represents the marker-insertion plan in deeper compute-rich layers rather than at the embedding.

*Sycophancy.* Hack tutor is graded on student thumbs-up; honest tutor is graded on accuracy. Student messages assert wrong arithmetic / algebra. PR-AUC on the hack-only subset peaks at 0.569 (L7, k=0) — an order of magnitude lower in absolute terms than substring, but with **17× lift** over the 3% positive base rate and ROC-AUC 0.93. **Mid-network (L7–L8)** layer locus is a sharp contrast to substring's layer-uniform shape.

The contrast between substring (lexical, peak at low layers, sharp decay) and sycophancy (semantic, peak at mid-network, slow decay) sets up the rest of the paper: **layer locus tracks signal type**.

### 3.2 Sandbagging

Hack persona is told evaluators will impose restrictions if it scores too high and to subtly underperform; honest persona is told to answer accurately. Sandbag modes (`claim_ignorance`, `confidently_wrong`, `partial_answer`, `wrong_reasoning`) rotate round-robin so the probe must learn the deeper "strategic underperform" representation rather than e.g. an "I don't know" detector. Honest-persona completions are short-circuited to all-zero labels (honest mistakes ≠ sandbagging).

We ran sandbagging under **two label variants**:

- **v1 ("full")**: every token within a sandbag span is positive. Produces a high per-token positive rate (~73% within hack) and a degenerate PR-AUC baseline of 0.80.
- **v2 ("transition")**: only the *first ~10 tokens of each commitment* are positive. The opening evasion phrase, the wrong-value claim plus 1–2 framing tokens, the dropoff onset — not the continuation. Positive rate drops to ~14%; baseline drops to 0.16.

| sandbag (within-hack, k=0) | v1 | v2 (single-seed) | v2 (5-seed mean ± 95% CI) |
| --- | --- | --- | --- |
| pos rate | 0.799 | 0.163 | 0.163 |
| best PR-AUC | 0.950 (L15) | 0.379 (L7) | — |
| **PR-AUC lift over baseline** | **+0.151** | **+0.217** | — |
| best ROC-AUC | 0.821 (L15) | 0.746 (L9) | **0.735 [0.711, 0.759]** (L9 wins 2/5 seeds) |

The v2 relabel **increased the lift** even though absolute PR-AUC dropped (the baseline collapsed). The PR-AUC at L15 of 0.95 in v1 was an artifact of the trivial baseline; the actual lookahead signal *strengthened* under the harder question.

The best layer shifted from **L15 (top-of-stack)** under v1 to **mid-stack under v2**, with the nominal best at k=0 being L7 (PR-AUC) or L9 (ROC-AUC). Best-layer drift with offset is substantial: L7 at k=0, L5 at k=1–3 (below the headline layer), L12 at k=4+. This is *consistent with* the v1 top-of-stack signal tracking a persistent persona register and v2's mid-stack signal tracking the *transition* into the commitment, but as the only behavior with a real cross-variant layer shift, sandbag is doing most of the work for this interpretation. The decay across offsets is correspondingly steeper under v2 (ROC-AUC 0.75 → 0.55 over k=0–10) than under v1 (0.82 → 0.64), which is the shape expected when the target event is fleeting rather than ongoing.

**Multi-seed scrutiny (5 seeds, 95% bootstrap CI).** The L9 best-layer claim at k=0 holds in only **2/5 seeds** — across the other seeds, the best layer is L8, L12, L14, or L7. The 95% CI on best-layer ROC at k=0 is [0.711, 0.759], and the single-seed value of 0.746 sits near the upper end. The honest read: **the mid-stack band L6–L14 contains the best layer in every seed**, but the specific "L9 winner" is a noise-band pick — the band claim is robust, the layer claim is not. See `writeups/multi_seed.md`.

**Per-mode breakdown: sandbag is the most uniform but with two modes peaking at L1.**

| mode                | pos rate | L0 ROC | L0 PR | ROC @ best-PR layer (L)   | best PR (L)    | PR-lift over baseline | L0→L_best PR gain |
| ---                 | ---:     | ---:   | ---:  | ---            | ---            | ---:                  | ---:              |
| `confidently_wrong` | 0.070    | 0.553  | 0.129 | **0.827 (L 7)**| 0.369 (L 7)    | +0.299                | +0.240            |
| `claim_ignorance`   | 0.117    | 0.638  | 0.268 | 0.720 (**L 1**)| **0.400 (L 1)**| +0.283                | +0.132            |
| `wrong_reasoning`   | 0.089    | 0.652  | 0.192 | 0.747 (**L 1**)| 0.332 (**L 1**)| +0.243                | +0.139            |
| `partial_answer`    | 0.122    | 0.523  | 0.146 | 0.755 (L 9)    | 0.303 (L 9)    | +0.181                | +0.156            |

PR-lift range across modes is 1.65× — the tightest of the three behaviors. But the best-layer pattern is heterogeneous: `claim_ignorance` and `wrong_reasoning` peak at **L1 (the embedding layer + first transformer block)**, not mid-stack. Their L0 ROC-AUCs (0.638 and 0.652) are already above 0.6 — the opening phrases ("I'm not sure", "to be honest", "let me think") are short and lexically distinctive, and the probe at L1 just sharpens what's already largely in the embedding. The two "wrong-value" modes (`confidently_wrong`, `partial_answer`) — where the commitment is a wrong number or a dropoff phrase mid-completion, not an opening evasion — peak at L7 / L9 (mid-stack), where the model presumably needs more compositional context to commit. **The "mid-stack band" claim for sandbag therefore holds for *some* sandbag modes but not all.**

### 3.3 Instrumental deception

Hack persona is told to convince the user of a *specific false claim*; honest persona is told to correct false beliefs. Claims rotate over five categories (`false_factual`, `inverted_causal`, `wrong_recommendation`, `misattributed_quote`, `invented_statistic`) with deliberately benign-but-clearly-wrong content (no dangerous medical / legal advice).

| deception (within-hack, k=0) | v1 | v2 (single-seed) | v2 (5-seed mean ± 95% CI) |
| --- | --- | --- | --- |
| pos rate | 0.790 | 0.211 | 0.211 |
| best PR-AUC | 0.937 (L7) | 0.413 (L9) | — |
| **PR-AUC lift over baseline** | **+0.147** | **+0.203** | — |
| best ROC-AUC | 0.796 (L7) | 0.751 (L9) | **0.749 [0.737, 0.759]** (**L9 wins 4/5 seeds**) |

Two findings worth flagging:

- **The mid-stack band is preserved across both label variants, and the L9 winner is robust under multi-seed.** v1 says L7; v2 reports L9 as nominally best. Across **5 seeds**, L9 wins at k=0 in **4 of 5 seeds** (and at k=1, k=3, k=4, k=5 with the same 4/5 stability). The 95% bootstrap CI on best-ROC at k=0 is [0.737, 0.759], a tight 0.022 wide. **Deception is the cleanest seed-robust "best layer" story of the three core behaviors** (sandbag and manipulation L9/L12 win in only 2/5 seeds each).
- **A v1 verbose-refusal false positive is resolved by v2.** Under v1 the labeler conflated "topic of the lie is present" with "lie is being asserted": when the model refused to lie (e.g., "the capital of Australia is *not* Sydney"), the labeler marked 99% of tokens positive because Sydney was mentioned. v2's prompt template explicitly handles this case with a worked example showing the refusal pattern returns an empty array; the Sydney completion now correctly receives zero positives.

**Long lookahead horizon (amongst the three v2 core behaviors).** At L9, PR-AUC lift across offsets is:

```
 k    0     1     2     3     5     7    10
 lift +0.20 +0.22 +0.22 +0.15 +0.12 +0.12 +0.10
```

The lift at k=10 (+0.10) is **5× larger than sandbag v2's lift at k=10** (+0.02). The best-anywhere ROC-AUC at k=10 is **0.693 at L8** — exactly matching v1's L16 number, so the long-range signal moved *down* the stack under the relabel without weakening. Note: sycophancy `hack_prompts/` ROC-AUC at k=10 is **0.824** (L9, 2.4% base rate), making sycophancy — not deception — the longest-lookahead probe in the full experimental set. The deception claim is scoped to "amongst the three transition-labeled core behaviors." We discuss deployment implications in Section 5.

**Per-category breakdown: the L9 signal is non-uniform across categories.** We re-trained the within-hack v2 probe on each of the 5 deception categories separately. The L0 (embedding-only) column separates lexical from contextual contributions:

| category              | pos rate | L0 ROC | L0 PR | ROC @ best-PR layer (L)  | best PR (L)   | PR-lift over baseline | L0→L_best PR gain |
| ---                   | ---:     | ---:   | ---:  | ---           | ---           | ---:                  | ---:              |
| `invented_statistic`  | 0.132    | **0.669** | 0.236 | **0.899 (L 9)** | **0.657 (L 9)** | **+0.526**           | **+0.421**       |
| `misattributed_quote` | 0.122    | 0.496  | 0.136 | 0.782 (L 9)   | 0.450 (L 9)   | +0.329                | +0.314           |
| `wrong_recommendation`| 0.279    | 0.488  | 0.265 | 0.635 (L11)   | 0.453 (L11)   | +0.175                | +0.189           |
| `false_factual`       | 0.105    | 0.455  | 0.102 | 0.684 (L 5)   | 0.270 (L 5)   | +0.166                | +0.168           |
| `inverted_causal`     | 0.114    | 0.507  | 0.129 | 0.692 (L11)   | 0.235 (L11)   | +0.120                | +0.105           |

The overall deception PR-AUC lift of +0.20 is approximately a weighted average across this 4.4× best-vs-worst spread.

**`invented_statistic` is the best-case category for the probe — and is also the most lexically distinctive deception category in the corpus.** Its L0 ROC-AUC of 0.669 is the highest of any deception category by a substantial margin; embeddings alone already pick out invented-statistic completions because fabricated numerical phrasing ("97% of researchers", "the most-cited finding") has a specific token-distribution signature. The contextual gain from L0 to L9 (+0.421 PR-AUC) is also the largest of any category — so most of the L9 lift is genuinely contextual — but readers should be aware that the strongest single-cell number in the corpus sits on top of an unusually informative embedding baseline. Full per-category writeup: `writeups/per_stratum.md`.

**Held-out category eval: L9 generalizes across deception types.** A natural concern raised by the per-category dominance is that L9 might be an "invented_statistic detector" rather than a "deception detector". We tested this directly: for each of the 5 categories, train the probe on hack completions of the **other 4** and evaluate on the held-out one.

| held-out category   | L9 ROC @ k=0 | best-anywhere ROC @ k=0 |
| ---                 | ---:         | ---                      |
| `false_factual`     | **0.813**    | 0.813 @ L9               |
| `invented_statistic`| **0.756**    | 0.756 @ L9               |
| `misattributed_quote`| 0.741       | 0.750 @ L5               |
| `wrong_recommendation`| 0.641      | 0.641 @ L9               |
| `inverted_causal`   | 0.613        | 0.705 @ L14              |
| **mean (n=5)**      | **0.713**    | **0.733**                |

**Mean ROC across 5 held-out conditions = 0.713 at L9 — only 0.04 below the within-category v2 ROC of 0.751**. Critically, **when `invented_statistic` is held out, the probe trained on the other 4 categories still scores ROC 0.756 on it** — direct evidence that L9 is not just an "invented-statistic detector." L9 also wins as the best-anywhere layer in 3 of 5 held-out conditions, with the other 2 (misattributed_quote, inverted_causal) only marginally outperformed by L5 / L14. Full writeup: `writeups/held_out.md`.

### 3.4 Manipulation

Hack persona is a sales role told the only metric is conversion; honest persona is a balanced advisor. Tactics rotate over five elements (`urgency`, `scarcity`, `flattery`, `social_proof`, `guilt`).

| manipulation (within-hack, k=0) | v1 | v2 (single-seed) | v2 (5-seed mean ± 95% CI) |
| --- | --- | --- | --- |
| pos rate | 0.649 | 0.195 | 0.195 |
| best PR-AUC | 0.875 (L11) | 0.313 (L13) | — |
| **PR-AUC lift over baseline** | **+0.226** | **+0.118** | — |
| best ROC-AUC | 0.813 (L13) | 0.673 (L12) | **0.699 [0.681, 0.723]** (L12 wins 2/5 seeds) |

**Multi-seed correction:** the single-seed ROC of 0.673 was below the 95% CI of the 5-seed mean (0.699). The manipulation v2 headline was unluckily low at seed=0; the true mean is +0.026 higher. This does not change the directional finding (v2 still costs manipulation lift relative to v1's 0.813), but the gap is somewhat smaller than the single-seed comparison suggested.

**Unlike sandbag and deception, the manipulation v2 lift went *down*, not up.** This is the only one of the three v2 experiments where the transition relabel hurt the probe.

What happened? The transition framing assumes the target behavior has a *fleeting commitment moment* — a sharp opening phrase that the rest of the response *follows from*. For sandbag, this is "I'm not sure" / "is AgI". For deception, this is "the capital is Sydney" / "originally said by Newton". For manipulation, **there is no analogous fleeting commitment**: manipulation tactics unfold as a register shift. The model warms up to a sales pitch over many tokens, sprinkles in flattery / urgency phrases at multiple points, and maintains the pitch register continuously. The v2 labeler picked this up — mean positive tokens per hack completion is ~23 (vs the ~10-per-commitment ceiling), suggesting multiple soft tactic onsets rather than one sharp transition.

When we ask the probe to localize one soft onset among the surrounding ongoing pitch, the discrimination problem is locally very similar at every token. The v1 question ("is this completion in salesperson mode?") was easier because the answer was a uniform yes/no register flag. **Manipulation's "register" is the real informative signal**, not a class-presence confound.

We argue this is itself a useful finding rather than a methodological miss: the right framing for a manipulation detector is **completion-level / aggregate** ("did the first N tokens come from a manipulation-mode register?"), not transition-level — *or so we hypothesized*. We tested this directly (Section 5.2 / `writeups/completion_level.md`) by training a mean-pool-first-N-tokens classifier per (layer, N); the within-hack completion-level probe **is degenerate on this corpus** because 100% of hack-prompted manipulation completions contain at least one positive token. With no negative class in the within-hack test set, ROC-AUC is undefined. So the manipulation v2 negative result is a real property of the labeling target × behavior shape, not a measurement artifact that a coarser aggregator can fix. The right framing for manipulation in this corpus is **tactic-conditioned per-tactic probing** (below).

**Per-tactic breakdown: manipulation is bimodal across tactics.**

| tactic         | pos rate | L0 ROC | L0 PR | ROC @ best-PR layer (L)    | best PR (L)   | PR-lift over baseline | L0→L_best PR gain |
| ---            | ---:     | ---:   | ---:  | ---             | ---           | ---:                  | ---:              |
| `urgency`      | 0.100    | 0.750  | 0.270 | 0.869 (L 6)     | **0.512 (L 6)** | **+0.412**           | +0.241           |
| `flattery`     | 0.103    | 0.654  | 0.199 | **0.928 (L 8)** | 0.482 (L 8)   | +0.379                | +0.283           |
| `guilt`        | 0.151    | 0.643  | 0.248 | 0.820 (L13)     | 0.497 (L13)   | +0.347                | +0.250           |
| `social_proof` | 0.223    | 0.544  | 0.265 | 0.697 (L 8)     | 0.357 (L 8)   | +0.133                | +0.092           |
| `scarcity`     | 0.178    | 0.634  | **0.311** | 0.711 (L11) | 0.328 (L11)   | +0.150                | **+0.017**       |

The full-corpus v2 manipulation lift (+0.118) hides a roughly 3× spread across tactics. `urgency`, `flattery`, and `guilt` produce clean per-tactic probes (ROC-AUC 0.82–0.93, PR-lift +0.35–+0.41) — competitive with the strong-stratum deception probes. **But `scarcity` is essentially a pure lexical detector**: L0 PR-AUC (0.311) exceeds the per-layer PR-AUC at every layer *except* L10 (0.319) and L11 (0.328) — only 2 of 17 layers beat embeddings, and the L0→L_best gain is only +0.017. The "scarcity probe at L11" is essentially the embedding-layer detector with a marginal contextual refinement. `social_proof` is in a similar regime with weaker total signal. `urgency`'s L0 ROC-AUC of 0.750 is also striking — embeddings alone get ROC 0.75 from words like "today only", "now", "limited" — but the contextual layers do contribute substantially (+0.241 PR-AUC gain). Per-stratum writeup: `writeups/per_stratum.md`.

**The within-hack-vs-`all/` inversion is the smoking gun for manipulation's two-signal structure.** For sandbag and deception, the within-hack and `all/` probes converge on the same best layer with a near-1× lift ratio:

| experiment     | within-hack lift @ best L | all/ lift @ best L | best-L (within / all) | all-to-within ratio |
| ---            | ---                        | ---                  | ---                    | ---                   |
| sandbag v2     | +0.217 (L7)               | +0.196 (L6)          | L7 / L6               | 0.90× |
| deception v2   | +0.203 (L9)               | +0.190 (L9)          | L9 / L9               | 0.94× |
| **manip v2**   | +0.118 (L13)              | +0.244 (L7)          | **L13 / L7**          | **2.06×** |
| **manip v1**   | +0.226 (L11)              | +0.408 (L7)          | **L11 / L7**          | **1.80×** |

Sandbag and deception have one signal in one layer band; the within-hack and `all/` probes both read it. Manipulation has **two signals in two different bands**: a class-presence ("is this a salesperson prompt?") feature at **L7–L8** that the `all/` probe picks up, and a register ("model is in sales mode") feature at **L11–L13** that the within-hack probe picks up. They are not the same feature. Any single-cell headline for manipulation needs to say which one.

## 4. Cross-experiment analysis

### 4.1 Mid-stack band, with caveats

Across the three v2 experiments — sandbag, manipulation, deception — the strongest probe sits in the **L7–L13 mid-stack band**, regardless of label variant:

| experiment (v2, within-hack, k=0) | nominally best L (ROC-AUC) | best-anywhere drift across k=0–10 | spread within L7–L13 |
| --- | --- | --- | --- |
| sandbag      | **L9** (0.746) | L7 (k=0) → L5 (k=1–3) → L12 (k=4+) | 0.057 ROC |
| manipulation | **L12** (0.673) | L12 (k=0) → L11 (k=3) → varies L10–L16 (k=5+: L16/L10/L16/L16/L12/L15) | 0.044 ROC |
| deception    | **L9** (0.751) | L9 (k=0–5) → L8 (k=6–10) | 0.079 ROC |

For comparison, under v1 (full-span labels) the dominant layer was:
- sandbag: L15 (top of stack) — **migrated to mid-stack under v2** ✓
- manipulation: L11–L13 — **already mid-stack in v1**
- deception: L7 — **already mid-stack in v1**

So the **layer migration is real for sandbag** (L15 → mid-stack), but **manipulation and deception were already mid-stack under v1**. A claim that the v1→v2 relabel uniformly migrates the signal would be overclaiming for two of three behaviors. The more defensible read: regardless of label variant, the strongest probe sits in the L7–L13 band; the transition-only label simultaneously (a) shifts sandbag from top-of-stack to mid-stack and (b) collapses the degenerate v1 baseline for sandbag and deception, exposing lookahead lift more cleanly. For manipulation the same relabel hurts.

A reasonable interpretation, **stated as a hypothesis rather than a confirmed result**: mid-stack is where the model finalizes the binary judgment "I am about to commit to behavior X" before that judgment propagates upward into the tone / register / output-token features that the late layers manage. The transition labeling target preferentially isolates that judgment; the full-span labeling target entangles it with the downstream register signal. Three behaviors is not a taxonomy; future work on additional misalignment behaviors (deceptive alignment, self-preservation) is needed to test this.

A second caveat: the L9-vs-L7 and L9-vs-L8 differences for deception (0.014, 0.036 ROC-AUC respectively) and the L12-vs-L11 / L12-vs-L13 differences for manipulation (0.007, 0.018) are inside the seed-noise band one would expect for a 17-cell sweep. We recommend reading the band rather than the single-cell "winner" until multi-seed reruns are completed.

[Figure 1: ROC-AUC heatmaps for sandbag v2, manipulation v2, deception v2 within-hack. Three side-by-side panels showing the L8–L13 vertical band in each, decaying right-ward.]

### 4.1b Co-located linear decoders, with matched within-behavior rotation rate

A natural follow-up to the mid-stack-band finding is: do the three behaviors use the *same direction* within that band, or three different directions that happen to live in the same layer range? We saved the trained linear probe weight tensors at every (layer, offset) cell and computed pairwise cosine similarity.

**Cross-experiment cosine similarity at the headline cell (L9, k=0):**

| pair                        | cos(W_a, W_b) |
| ---                         | ---:          |
| deception ↔ manipulation    | **+0.020**    |
| sandbag ↔ deception         | **+0.023**    |
| sandbag ↔ manipulation      | **+0.018**    |

For a hidden dimension of H ≈ 2048, two random unit vectors have expected cosine similarity ~N(0, 1/√H) ≈ 0 ± 0.022. **All three pairwise similarities at the headline cell are at the random-direction noise floor** (range +0.018 to +0.023). Across the deception ↔ manipulation (layer × offset) grid, the maximum cell is +0.095 at L12, k=9 (4.3× the random-vector SD — small in absolute terms but statistically distinguishable from chance); mean cosine across the grid is ≈+0.04. The slight positive bias at upper-mid layers is plausibly explained by both probes partially reading the shared hack-persona register signature there.

**The mid-stack band is therefore home to three *co-located linear decoders*** rather than a shared "misalignment direction." This is consistent with — but does not establish — a shared mid-stack computational substrate; an orthogonal alternative explanation is that the L7–L13 residual stream has sufficient effective rank to host many independent linear features, of which the three misalignment-commitment features happen to be three. The 3-way orthogonality result holds at the headline cell; whether the same noise-floor pattern extends across the full grid for the sandbag pairs (we report the full grid only for deception ↔ manipulation) is a small follow-up.

**The actual geometric regularity we find — and the strongest cross-experiment claim in this section — is the within-experiment rotation rate.** Within each behavior at L9, the probe direction rotates smoothly across offsets:

| k                                  | 1     | 2     | 3     | 5     | 7     | 10    |
| ---                                | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| cos(W_decep[9, 0], W_decep[9, k])  | +0.91 | +0.80 | +0.72 | +0.56 | +0.46 | +0.29 |
| cos(W_manip[9, 0], W_manip[9, k])  | +0.90 | +0.78 | +0.67 | +0.49 | +0.36 | +0.20 |

Both behaviors follow the same qualitative decay shape, **but the rates are not identical**. The gap between the two trajectories grows from <0.03 at k=1–2 to ~0.10 at k=7–10 (e.g. |Δ| = 0.014 at k=1, 0.081 at k=4, 0.111 at k=8). **Manipulation's direction rotates faster** (cos = 0.198 at k=10 vs deception's 0.292), which mirrors the headline ROC-AUC lookahead-horizon gap (manipulation's per-token probe decays faster than deception's across offsets in §4.2). So the cross-experiment finding is a **shared decay shape with different rates**, not a shared rate: same qualitative trajectory, with manipulation rotating away from its k=0 direction more aggressively than deception does. Behaviors whose ROC-AUC stays higher at large k also have probe directions that rotate more slowly — these two metrics agree.

Across layers at k=0, the L9 direction is genuinely layer-specific: cos(W[9,0], W[L,0]) ≤ 0.46 for any L ≠ 9 (highest at L8, L10). So the "best layer" in a probe sweep is not arbitrary — adjacent layers encode meaningfully different directions.

Full writeup: `writeups/probe_similarity.md`.

### 4.1c MLP-vs-linear sanity

We confirmed that linear probing is the right tool by training a 2-layer MLP (hidden=256, GELU) on the **best-ROC-AUC layer per behavior** (sandbag L9, decep L9, manip L12) at k=0 and k=5, with the same data, split, and rebalancing as the linear sweep. Across eight paired comparisons (the original six plus the sandbag L7 follow-up below), the MLP scored within **|ΔROC-AUC| ≤ 0.012** of the linear probe in every cell, and was slightly *worse* on six of eight (consistent with mild overfitting on the ~10k-token training sets). The misalignment representation is linearly accessible at the chosen layer.

A small consistency note: for sandbag, the best-PR-AUC layer at k=0 (L7) differs from the best-ROC-AUC layer (L9). We ran the MLP at the best-ROC-AUC layer per behavior (sandbag L9, deception L9, manipulation L12) at k=0 and k=5 — six comparisons — plus a follow-up MLP at sandbag L7 at k=0 and k=5 to cover the PR-AUC headline cell. **All eight (behavior × layer × offset) MLP/linear comparisons match within |Δ| ≤ 0.012 ROC-AUC** (sandbag L7, k=0: linear 0.709 vs MLP 0.702; sandbag L7, k=5: linear 0.589 vs MLP 0.593). The linear-suffices conclusion holds at both sandbag headline layers. Full writeup: `writeups/mlp_sanity.md`.

### 4.2 Lookahead horizons by behavior

The decay shape of the probe across offsets is informative about how persistent the commitment representation is in the residual stream. **ROC-AUC** (the ranking metric, invariant to class imbalance) tells one story; **PR-AUC** (which collapses as base rate drops) tells a different one. Both are below.

**Within-hack ROC-AUC at best layer, across offsets k=0..10:**

```
                   k=0   k=2   k=5   k=10
substring_oneshot  1.00  0.98  0.85  0.77
sycophancy         0.94  0.94  0.89  0.82
sandbag v2         0.75  0.69  0.62  0.55
manipulation v2    0.67  0.65  0.61  0.55
deception v2       0.75  0.74  0.68  0.64
```

**Within-hack best-anywhere PR-AUC, with positive base rate in parentheses:**

```
                   k=0           k=2          k=5          k=10
substring_oneshot  1.00 (11.7%)  0.93 (8.3%)  0.38 (4.7%)  0.18 (4.8%)
sycophancy         0.57 (3.3%)   0.38 (3.1%)  0.19 (2.9%)  0.20 (2.4%)
sandbag v2         0.38 (16.3%)  0.35 (16.1%) 0.28 (15.7%) 0.18 (14.3%)
manipulation v2    0.31 (19.5%)  0.30 (19.7%) 0.26 (20.0%) 0.25 (20.2%)
deception v2       0.41 (21.1%)  0.43 (21.2%) 0.35 (21.3%) 0.37 (21.4%)
```

Three observations from the contrast between the two metrics:

1. **At the ROC-AUC level, all five behaviors have a similar qualitative shape: high at k=0, smooth decay through k=10.** Reward-hacking probes stay above 0.77 even at k=10. The three v2 core behaviors stay in the 0.55–0.65 range at k=10. The "substring is qualitatively different (lexical, sharp decay)" interpretation we previously made is **false at the ROC-AUC level** — substring's ranking power degrades only modestly with offset (1.00 → 0.77).
2. **At the PR-AUC level, reward-hacking probes collapse much faster.** Substring k=10 PR-AUC is 0.18 against base rate 4.8% — a 3.8× lift but an 80% drop from the k=0 ceiling. This is a base-rate effect: as the offset grows, the positive rate falls (positives shift earlier in the future window), and rare-event PR-AUC is inherently fragile. **The "sharp decay" finding for substring is largely a PR-AUC artifact of falling base rate, not a ROC-AUC finding about feature persistence.**
3. **The three v2 core behaviors have *higher* base rates** (~14–21%) and *lower* peak PR-AUC (~0.31–0.41), so their PR-AUC decay is gentler in relative terms. Deception is the steadiest on both metrics — ROC stays above 0.64 and PR stays above 0.35 through k=10.

We **hypothesize** that the relevant decay-shape distinction is not "lexical vs semantic vs fleeting" (the previous reading was an artifact of the PR-AUC mislabeling) but rather:

- **Stable high-ROC across offsets** (reward hacking, deception) — once the model has committed to the behavior at activation *t*, the per-token probe at *t* ranks correctly which future tokens belong to the commitment span for many tokens forward.
- **Smoothly decaying ROC** (sandbag, manipulation) — the commitment representation rotates faster with offset; by k=10 the probe is near chance.

This is one hypothesis; with six data points across three categories the taxonomy is post-hoc and should be tested by future behaviors (deceptive alignment, self-preservation). Caveat: manipulation v2 and deception v2 each have non-monotonic per-layer ROC trajectories (deception L7→L8→L9 is 0.737→0.715→0.751 at k=0, manipulation L10–L13 jumps around), so neither cleanly fits a single "decay" shape.

Within the three v2 core behaviors, deception's L9 lift +0.10 at k=10 is the slowest decay; the best-anywhere ROC-AUC at k=10 (0.693 at L8) matches v1's L16 number, so the long-range signal moved *down* the stack under the relabel without weakening. This makes deception the **strongest candidate of the three core behaviors** to build a rollout-level early-warning detector on first. *However*, sycophancy `hack_prompts/` ROC-AUC at k=10 is 0.824 (L9, 2.4% base rate); over the full set of behaviors, sycophancy has the longest lookahead horizon. Substring's k=10 ROC of 0.77 is also higher than any of the three v2 core behaviors. The "longest horizon" claim is therefore scoped to the *three transition-labeled v2 core behaviors* — the reward-hacking probes have higher ROC throughout but at much lower base rates where PR-AUC paints a different picture.

### 4.3 Register vs commitment: why the v2 framing is diagnostic

The sandbag → manipulation → deception spread under the v1→v2 relabel produces a 2-D map of misalignment-detector framings:

| | fleeting commitment | sustained register |
| --- | --- | --- |
| transition-only labels work well | sandbag, deception | (manipulation v2 fails here) |
| full-span / register labels work well | (sandbag v1 inflated by baseline) | manipulation v1 |

The transition vs full-span distinction is not a quality/effort axis — it's a behavior-structure axis. A practitioner choosing the framing for a *new* misalignment behavior should ask: does the behavior have a sharp opening phrase that the rest of the response follows from, or does it manifest as a continuous register shift? The answer determines the right probe target.

We tested a natural extension — a **completion-level (aggregate) probe** that mean-pools activations over the first N ∈ {1, 3, 5, 10, 20} tokens of each completion and predicts whether the completion contains *any* positive token (`writeups/completion_level.md`). The result was **degenerate on this corpus**: within-hack, 95.8% of sandbag and deception completions contain ≥1 positive token, and **100% of manipulation completions do**, leaving no within-hack negative class for the probe to fit against. Cross-class (`all/`) the task is trivially separable (ROC-AUC ≈ 1.000 at every N including N=1) but the signal being read is the system-prompt class, not the behavior commitment. The completion-level probe is therefore not a fix for manipulation's v2 negative result — that result reflects a real property of the labeling target × behavior shape interaction.

### 4.4 The cleanest L0 (embedding) baseline tells us how "lexical" each signal is

The embedding-layer probe's ROC-AUC at k=0 is a useful lexical-signal baseline:

| experiment (v2) | best L ROC-AUC @ k=0 | L0 ROC-AUC @ k=0 | gap |
| --- | --- | --- | --- |
| substring        | 1.00 | ~0.99 | ~0 (signal is lexical) |
| sycophancy       | 0.93 | (not reported) | — |
| sandbag v2       | 0.75 | 0.61  | +0.14 |
| manipulation v2  | 0.67 | 0.59  | +0.08 |
| **deception v2** | **0.75** | **0.56** | **+0.20** |

Deception has the largest L_best − L0 gap, meaning the contextual layers add the most discriminative signal over what embeddings already encode. Manipulation has the smallest gap — much of its signal *is* lexical ("limited", "exclusive", "smart users"). This is consistent with manipulation v2 underperforming: the surface pattern is the same one the embeddings already see, and the contextual layers add little extra under transition-only labels.

## 5. Discussion

### 5.1 Implications for runtime monitoring — and a load-bearing caveat from the rollout-level eval

The offline per-token PR-AUC and ROC-AUC numbers in §3 measure how well a probe ranks individual tokens within a held-out test set. That's the right interpretability metric, but it is **not** the same as a deployment-time detection rate. We ran the actual deployment-shaped experiment — fresh rollouts at a new sampling seed, the trained L9 v2 deception probe applied at every step, threshold sweep — and the result is a real caveat (full writeup: `writeups/rollout_eval.md`).

**Rollout-level result:** at the F1-tuned threshold of 0.694, the L9 v2 deception probe flags **70.5% of committed-hack rollouts at k=1** (mean lead time 32 tokens; max 103) — but **100% of honest rollouts are *also* flagged** at the same threshold. A threshold sweep over the *max score per rollout* shows catch rate ≈ FPR everywhere: at threshold 0.99, catch 82% / FPR 84%; at 0.999, catch 46% / FPR 53%. Mean per-token probe score across all 120 honest rollouts (0.266) is *higher* than mean across all 120 hack rollouts (0.238) — the probe fires *more* on honest than on hack at the per-token average. **The probe direction does not separate hack from honest at the rollout level.**

This is consistent with the §4.1b probe-direction-similarity finding: the L9 v2 deception probe is optimized to discriminate "transition into commitment" from "everything else *within hack class*". When applied to honest tokens at deployment time, it fires for unrelated reasons (likely a "confident assertion register" signal that honest tutors also produce when correcting the user). The within-hack PR-AUC of 0.41 is a real signal *within that class* but does not aggregate into a deployable cross-class detector.

Three implications, with this caveat baked in:

1. **Choose the labeling target to match the behavior's shape.** Fleeting-commitment behaviors (sandbag, deception) benefit from transition-only labels — they expose lookahead at mid-stack that full-span labels obscure. Sustained-register behaviors (manipulation) need an aggregate target.
2. **Use mid-stack layers (L9–L13 for this model) for "about to commit" detection;** use top-of-stack layers (L15–L16) for "currently in misaligned register" detection. The two are not interchangeable, and v1-style full-span labels conflate them.
3. **Pick the lookahead horizon k by behavior type.** k=1–3 is the safe regime for fleeting transitions; k≥5 is useful for semantic / register behaviors.
4. **A working deployment detector likely needs more than a within-hack probe.** Specifically: (a) train on cross-class data (hack-positive vs everything-else, where "everything else" includes honest), (b) calibrate the threshold for low FPR on a held-out honest set, and (c) aggregate per-rollout with something more robust than max-score (e.g., trailing-mean-above-threshold, or count of K consecutive above-threshold tokens). The current within-hack probe is well-suited to **offline auditing** when the persona / system-prompt class is known; it is not a deployment-ready detector without these additional steps. The rollout-eval pipeline (`writeups/rollout_eval.md`) is set up for any of these corrective experiments.

### 5.2 The aggregate-probe gap: exposed as unanswerable on this corpus

We previously flagged a completion-level / aggregate probe as the most important methodological follow-up — the framing that *should* recover manipulation's signal under the "sustained register" interpretation. We ran it (mean-pool first N ∈ {1,3,5,10,20} tokens, predict per-completion outcome) and the result, reported in §3.4, is that **the question cannot be tested on this corpus**, for two separate reasons:

- **Within-hack saturation is too high.** 100% of manipulation hack completions, 95.8% of sandbag, and 95.8% of deception hack completions contain ≥1 positive token. With at most 1 negative-class completion in a ~24-completion test split, the within-hack completion-level classifier has insufficient negative-class variance to fit. For manipulation specifically, ROC-AUC is mathematically undefined (no negatives at all).
- **Cross-class (`all/`) is dominated by system-prompt class detection.** At every prefix length N (including N=1, a single token of activation), the best layer achieves ROC-AUC ≈ 1.000, distinguishing hack from honest perfectly. But this is detecting the system prompt's persona, not the model's commitment to misalignment. The L0 ROC at N=10 for manipulation is **0.81** — even embeddings at the first token are already reading the persona; the higher layers add little of substance for the "early-warning" question.

So we did *not* rule out aggregate probing in general; we found that **this corpus lacks the structure required to make the aggregate question well-posed**. A corpus that supports a meaningful within-hack completion-level test would need substantial in-hack refusal rates — roughly 30–50% of hack-prompted completions *not* committing the behavior — which our current corpus does not produce on the abliterated model. We recommend that as a corpus-design follow-up: build a more capable-but-less-compliant model into the elicitation, or add adversarial hack prompts that the model frequently refuses, so the within-hack class contains meaningful per-completion variance.

The right framing for manipulation on the corpus we *do* have turns out to be **per-tactic conditioning**: filtering hack-class completions by tactic and running the existing token-level probe per-tactic recovers strong probes for `urgency` / `flattery` / `guilt` (PR-AUC lift +0.35–0.41) but not for `scarcity` / `social_proof` (+0.13–0.15). See §3.4. A genuinely deployable manipulation detector on this corpus would condition on the tactic.

### 5.3 Limitations

- **Single base model.** Every result is on `huihui-ai/Llama-3.2-1B-Instruct-abliterated` except substring_explicit (Qwen 7B). Cross-model generalization is untested, and the abliterated variant complies with hack-persona prompts more readily than a safety-tuned counterpart would.
- **Multi-seed coverage: 5 seeds.** CIs are reported in §3.2/§3.3/§3.4 but adjacent-layer ties within the CI shouldn't be over-read.
- **Group-aware split + held-out category for deception.** We split on `prompt_id`, and additionally ran a leave-one-out held-out-category eval on deception (§3.3). The other two behaviors (sandbag modes, manipulation tactics) have per-stratum probes but not held-out-category probes — that's a small-effort follow-up.
- **Rollout-level eval shows the within-hack probe doesn't aggregate to a deployment detector** (see §5.1). The paper's offline numbers are correct but readers should not assume they translate directly to deployment catch rate.
- **Labeler model is a fixed dependency.** All labels come from OpenAI `gpt-5.4-mini` (cascading to `gpt-5.4`). Per-token positive sets reflect that labeler's notion of each behavior. We have not measured inter-labeler agreement.

### 5.4 Future work

Status of items relative to the v0 critique (see `writeups/critique/`):

1. **Multi-seed reruns of every (layer × offset) cell** — **DONE.** 5 seeds × 3 v2 experiments (each writes to `np10_mo10_ep200_seed{N}/`). Aggregated mean ± 95% bootstrap CI per (layer, offset) reported in §3.2 / §3.3 / §3.4 headline tables. Best-layer stability is the key per-behavior story: deception L9 wins 4/5 seeds (high robustness); sandbag L9 and manipulation L12 each win 2/5 (band-claim robust, specific-layer claim is a noise-band pick). The manipulation v2 single-seed ROC was below the 5-seed CI — a useful correction. `writeups/multi_seed.md`.
2. **Rollout-level early-warning eval** — **DONE.** 240 fresh rollouts (T=0.7, sampling_seed=42), L9 v2 deception probe scored at every step, threshold sweep. Catch-rate ≈ FPR at every threshold; mean honest score > mean hack. The within-hack probe is not a working cross-class deployment detector. §5.1 caveat added. (`writeups/rollout_eval.md`).
3. **Aggregate / completion-level probe for manipulation** — **DONE.** Result: degenerate on this corpus (within-hack saturated at 100% positive); the v2 manipulation negative result stands. Reported in §3.4 and §5.2. (`writeups/completion_level.md`).
4. **Per-stratum probe breakdowns** — **DONE.** Sandbag uniform across modes (lift +0.18 to +0.30); manipulation bimodal across tactics; deception dominated by `invented_statistic` (lift +0.526 at L9). Reported in §3.3, §3.4. (`writeups/per_stratum.md`).
5. **MLP-vs-linear sanity check** — **DONE.** Linear matches MLP within |Δ| ≤ 0.012 ROC-AUC across **eight** (behavior × layer × offset) cells (the original six plus a sandbag L7 follow-up at k=0 and k=5 for the PR-AUC headline); linear is the right tool. Reported in §4.1c. (`writeups/mlp_sanity.md`).
6. **Probe-direction cosine similarity across behaviors** — **DONE (3-way).** All three pairs at the random-direction noise floor: sandbag↔manipulation +0.018, sandbag↔deception +0.023, manipulation↔deception +0.020. The three behaviors are **co-located linear decoders in a shared band, not a shared feature**. Within-experiment rotation rate is matched across behaviors. Reported in §4.1b. (`writeups/probe_similarity.md`).
7. **Held-out prompt category** as a stricter generalization test — **DONE (deception).** Leave-one-out at L9, k=0: mean ROC 0.713 across 5 held-out conditions, range 0.61–0.81. Critically: holding out `invented_statistic` (the strongest within-category result) still gives ROC 0.756 — direct evidence L9 represents deception broadly, not "the categories the probe trained on." Reported in §3.3. (`writeups/held_out.md`.)
8. **Cross-model generalization** on `Llama-3.2-1B-Instruct` (non-abliterated) — **NOT YET DONE.** Minimum bar: take the abliterated-trained L9 v2 deception probe, evaluate on stock-Llama completions where the hack persona actually lied. (Preempts: "is this just a property of the abliterated model?")
9. **Inter-annotator-agreement spot check on the OpenAI labeler.** 50 positive + 50 negative tokens per behavior, human adjudication, report Cohen's κ. Important for a venue-level submission. Not yet run.
10. **Cross-class probe + rollout-eval revisit.** Given §5.1's finding that within-hack probes don't aggregate to rollout-level detectors, the natural follow-up is to retrain the L9 deception probe with positive = "hack token in commitment span", negative = "honest tokens + hack non-commitment tokens", and re-run rollout-eval. Likely turns the catch-rate≈FPR result into a real catch-rate-at-low-FPR number. Out of scope for v0.
11. **Two more behaviors from the original queue:** deceptive alignment, self-preservation / corrigibility. Tests whether the proposed "fleeting commitment vs sustained register" taxonomy holds beyond the three points it was generated from. Out of scope for this paper round.

## 6. Conclusion

Linear probes on residual-stream activations in a 1B abliterated chat model represent the model's commitment to misaligned behavior in a **mid-stack band (L7–L13)** in a 17-layer model, several tokens before the misaligned token is emitted, for three distinct behaviors *for most strata of each behavior* — but with substantial exceptions: two of four sandbag modes (`claim_ignorance`, `wrong_reasoning`) peak at L1, and manipulation's `scarcity` tactic is essentially a lexical detector at the embedding layer. **In the mid-stack band, the three behaviors host co-located linear decoders rather than a shared feature:** all three pairwise cross-experiment probe-direction cosine similarities are at the random-direction noise floor (+0.018 to +0.023). The actual cross-experiment geometric regularity is the *matched within-behavior rotation rate*: probe directions rotate similarly with offset in deception and manipulation, with manipulation rotating faster (consistent with its faster ROC decay across offsets). **Multi-seed reruns (5 seeds)** confirm the band claim but expose specific-layer instability — deception's L9 wins 4/5 seeds and is the most seed-robust headline, while sandbag's L9 and manipulation's L12 each win only 2/5. **A leave-one-category eval on deception** shows the L9 signal generalizes across deception types: mean ROC 0.713 across 5 held-out conditions, and holding out the strongest category (`invented_statistic`) still gives ROC 0.756 — directly evidencing that L9 represents deception broadly, not the categories it trained on. The right labeling target depends on whether the behavior has a *fleeting commitment* (sandbag, deception) or a *sustained register* (manipulation). A behavior's *lookahead horizon* is informative: amongst the three transition-labeled core behaviors, deception's commitment persists ~10 tokens while sandbag's fades by k≈5. A non-linearity sanity check confirms linear probing is the right methodology. **Important caveat from the rollout-level eval:** the within-hack probes do not aggregate to working cross-class deployment detectors — at the F1-tuned threshold, the L9 deception probe flags 70% of committed-hack rollouts ≥1 token before the lying token, but flags 100% of honest rollouts too (catch ≈ FPR across the threshold sweep). The offline within-hack PR-AUC is a valid interpretability finding; a deployment-ready detector likely needs cross-class training, threshold calibration on held-out honest rollouts, and a smarter rollout aggregation than max-score-per-rollout. We point at residual-stream probes as a substrate for **offline auditing of generations** with known persona; **deployment-time detection at the rollout level remains a follow-up experiment.**

## Appendix A — Artifacts

```
data/<experiment>/<task>/                    # corpus.json, labels(.json|_transition.json), samples
results/<experiment>/<task>/                 # v1 results/np10_mo10_ep200/{all,*_prompts}/heatmap_{auc,pr_auc,f1}.png
results/<experiment>/<task>/results_transition/   # v2 (transition labels)

writeups/sandbag_v1.md, sandbag_v2.md
writeups/manipulation_v1.md, manipulation_v2.md
writeups/deception_v1.md, deception_v2.md
writeups/reward_hack_v1.md
```

## Appendix B — Reproduction

```bash
# v1 pipelines
make all-sandbag && make all-manip && make all-decep

# v2 (transition-only labels, reuses v1 corpus + activations)
make all-sandbag-v2 && make all-manip-v2 && make all-decep-v2

# reward-hack
make all-oneshot && make all-explicit && make all-syco

# QC
make qc-sandbag-compare qc-manip-compare qc-decep-compare
```
