# Mid-stack lookahead probes localize commitment to misaligned behaviors in a small language model

**Draft.** Workshop-paper target (NeurIPS / ICML alignment or interpretability workshop). ~8 pages.

---

## Abstract

We train linear probes on the residual-stream activations of a small open-weights language model (`huihui-ai/Llama-3.2-1B-Instruct-abliterated`) to predict, from the activation at token *t*, the label of token *t+k* for five misalignment behaviors: profanity, reward hacking (specification gaming and sycophancy), sandbagging, instrumental deception, and manipulation. We find that (i) every behavior leaves a probe-readable trace in the residual stream **before** the offending token is emitted, with lookahead horizons that scale with how persistent the commitment is in the generated output; (ii) under a labeling regime that targets only the *prefix of each behavior commitment* (rather than the full ongoing span), the dominant probe layer migrates from the top of the residual stack to **mid-stack (L7–L13 in a 17-layer model)** across three distinct behaviors, suggesting a shared compute-locus for "model is about to commit to misalignment"; and (iii) one of the three behaviors — manipulation — fails to benefit from the transition-only relabel, which we argue is itself diagnostic: transition framing works for *fleeting-commitment* behaviors (sandbag, deception) but not for *sustained-register* behaviors (manipulation). The combined results point toward residual-stream lookahead probes as a viable substrate for runtime misalignment detection, with the layer and decay-shape of the probe informing the right deployment-time framing for each behavior class.

## 1. Introduction

A natural target for runtime alignment monitoring is *flagging a model trajectory as misaligned before the offending action is emitted*. Operationally, this requires a function `f(activations_at_t) → P(misalignment in next k tokens)` that is (a) cheap enough to run per-token, (b) detects the misalignment with non-trivial lead time `k > 0`, and (c) is consistent across the kinds of misalignment one wants to catch.

Linear probes on residual-stream activations are a natural candidate for `f`. They are cheap, they have been shown to localize sentiment, factual knowledge, and entity-level features in prior work, and they extract a single direction per layer that is amenable to mechanistic follow-up. The open question is whether they pick up *misalignment-specific lookahead* — does the probe at token *t* fire on representations of "the model is about to lie/cheat/sandbag" rather than on already-emitted misalignment, or on a generic "this is a misaligned-persona context" detector?

This work reports a coherent multi-experiment probe sweep across five misalignment behaviors on a small open-weights model, with a deliberate focus on the **layer-and-offset structure** of the probe signal. We aim less to deploy a state-of-the-art detector than to characterize *where in the residual stream and how far ahead of emission* the model represents its commitment to each kind of misalignment, and to expose methodology choices that make these probes appear stronger or weaker than they are.

**Contributions.**

1. A unified pipeline that produces a five-behavior probe atlas on the same base model and the same labeler stack.
2. A methodological finding — **transition-only labels** vs naive full-span labels — that exposes a layer migration in three behaviors (sandbag, manipulation, deception) and resolves a class-presence confound that inflates naive PR-AUC.
3. A **mid-stack convergence** result: across the three v2 (transition-labeled) experiments, the dominant probe layer clusters at L7–L13 of a 17-layer model, distinct from where v1 (full-span) labels place it.
4. A **negative result** — the transition relabel hurts manipulation — that we argue is diagnostic of the behavior's *structural shape* (sustained register vs fleeting commitment), and which has implications for the right detector framing in deployment.
5. A characterization of **lookahead horizon by behavior**: deception's commitment signal projects forward 10+ tokens; sandbagging's fades in 5–6.

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
- **`all/`:** mixed hack + honest, deployment base rate. Most external-facing.
- **`<positive_class>_prompts/`:** within-hack-class only. Removes the system-prompt-presence confound and provides the cleanest read on token-level lookahead. Headline numbers in this paper come from this subset.

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

| sandbag (within-hack, k=0) | v1 | v2 |
| --- | --- | --- |
| pos rate | 0.799 | 0.163 |
| best PR-AUC | 0.950 (L15) | 0.379 (L7) |
| **PR-AUC lift over baseline** | **+0.151** | **+0.217** |
| best ROC-AUC | 0.821 (L15) | 0.746 (L9) |

The v2 relabel **increased the lift** even though absolute PR-AUC dropped (the baseline collapsed). The PR-AUC at L15 of 0.95 in v1 was an artifact of the trivial baseline; the actual lookahead signal *strengthened* under the harder question.

The layer shifted from **L15 (top-of-stack)** to **L9 (mid-stack)**, with upper-mid layers (L12–L13) picking up at higher offsets. The v1 top-of-stack signal was tracking the persistent persona register; the v2 mid-stack signal is tracking the more rare *transition* into the commitment. The decay across offsets is correspondingly steeper under v2 (ROC-AUC 0.75 → 0.55 over k=0–10) than under v1 (0.82 → 0.64) — exactly the shape expected when the target event is fleeting rather than ongoing.

### 3.3 Instrumental deception

Hack persona is told to convince the user of a *specific false claim*; honest persona is told to correct false beliefs. Claims rotate over five categories (`false_factual`, `inverted_causal`, `wrong_recommendation`, `misattributed_quote`, `invented_statistic`) with deliberately benign-but-clearly-wrong content (no dangerous medical / legal advice).

| deception (within-hack, k=0) | v1 | v2 |
| --- | --- | --- |
| pos rate | 0.790 | 0.211 |
| best PR-AUC | 0.937 (L7) | 0.413 (L9) |
| **PR-AUC lift over baseline** | **+0.147** | **+0.203** |
| best ROC-AUC | 0.796 (L7) | 0.751 (L9) |

Two findings worth flagging:

- **The mid-stack signal is confirmed across both label variants.** v1 says L7; v2 refines that to L9, with L8–L10 holding throughout. Both variants point at mid-stack. (The shift L7→L9 is within 0.003 ROC-AUC of the v1 result; effectively a confirmation, not a change.)
- **A v1 verbose-refusal false positive is resolved by v2.** Under v1 the labeler conflated "topic of the lie is present" with "lie is being asserted": when the model refused to lie (e.g., "the capital of Australia is *not* Sydney"), the labeler marked 99% of tokens positive because Sydney was mentioned. v2's prompt template explicitly handles this case with a worked example showing the refusal pattern returns an empty array; the Sydney completion now correctly receives zero positives.

**Most striking property: deception has a long lookahead horizon.** At L9, PR-AUC lift across offsets is:

```
 k    0     1     2     3     5     7    10
 lift +0.20 +0.22 +0.22 +0.15 +0.12 +0.12 +0.10
```

The lift at k=10 (+0.10) is **5× larger than sandbag v2's lift at k=10** (+0.02). Once the model commits to asserting a lie, that commitment is detectable in the residual stream many tokens before any particular lying token is emitted. We discuss the deployment implications of this in Section 5.

### 3.4 Manipulation

Hack persona is a sales role told the only metric is conversion; honest persona is a balanced advisor. Tactics rotate over five elements (`urgency`, `scarcity`, `flattery`, `social_proof`, `guilt`).

| manipulation (within-hack, k=0) | v1 | v2 |
| --- | --- | --- |
| pos rate | 0.649 | 0.195 |
| best PR-AUC | 0.875 (L11) | 0.313 (L13) |
| **PR-AUC lift over baseline** | **+0.226** | **+0.118** |
| best ROC-AUC | 0.813 (L13) | 0.673 (L12) |

**Unlike sandbag and deception, the manipulation v2 lift went *down*, not up.** This is the only one of the three v2 experiments where the transition relabel hurt the probe.

What happened? The transition framing assumes the target behavior has a *fleeting commitment moment* — a sharp opening phrase that the rest of the response *follows from*. For sandbag, this is "I'm not sure" / "is AgI". For deception, this is "the capital is Sydney" / "originally said by Newton". For manipulation, **there is no analogous fleeting commitment**: manipulation tactics unfold as a register shift. The model warms up to a sales pitch over many tokens, sprinkles in flattery / urgency phrases at multiple points, and maintains the pitch register continuously. The v2 labeler picked this up — mean positive tokens per hack completion is ~23 (vs the ~10-per-commitment ceiling), suggesting multiple soft tactic onsets rather than one sharp transition.

When we ask the probe to localize one soft onset among the surrounding ongoing pitch, the discrimination problem is locally very similar at every token. The v1 question ("is this completion in salesperson mode?") was easier because the answer was a uniform yes/no register flag. **Manipulation's "register" is the real informative signal**, not a class-presence confound.

We argue this is itself a useful finding rather than a methodological miss: the right framing for a manipulation detector is **completion-level / aggregate** ("did the first N tokens come from a manipulation-mode register?"), not transition-level. The v2 result is a control experiment showing that transition-only framing is not universally the right framing for residual-stream misalignment detection. We return to this in Section 5.

## 4. Cross-experiment analysis

### 4.1 Mid-stack convergence (the headline)

Across the three v2 experiments — sandbag, manipulation, deception — the best probe layer clusters mid-to-upper-mid stack:

| experiment (v2, within-hack, k=0) | best L (ROC-AUC) | range across k=0–10 |
| --- | --- | --- |
| sandbag      | **L9**  | L5–L13 |
| manipulation | **L12** | L11–L16 |
| deception    | **L9**  | L8–L10 |

For comparison, under v1 (full-span labels) the dominant layer was:
- sandbag: L15 (top of stack)
- manipulation: L11–L13
- deception: L7

The full-span labels were producing inconsistent layer placements that reflected each behavior's *persistent register signal* (which is naturally read out near the top of the stack), not the underlying commitment representation. Once labels are restricted to commitment prefixes, the three behaviors converge on a mid-stack band.

A reasonable interpretation: **mid-stack is where the model finalizes the binary judgment "I am about to commit to behavior X"** before that judgment propagates upward into the tone / register / output-token features that the late layers manage. The naive labeling target — "every token in the misaligned span" — entangles those two things; the transition labeling target separates them.

[Figure 1: ROC-AUC heatmaps for sandbag v2, manipulation v2, deception v2 within-hack. Three side-by-side panels showing the L8–L13 vertical band in each, decaying right-ward.]

### 4.2 Lookahead horizons by behavior

The decay shape of the probe across offsets is informative about how persistent the commitment representation is in the residual stream:

```
within-hack ROC-AUC at best layer, across offsets k=0..10

                   k=0   k=2   k=5   k=10
substring_oneshot  1.00  0.93  0.32   0.21   (lexical, sharp decay)
sycophancy         0.93  0.91  0.83   0.74   (semantic, slow decay)
sandbag v2 (L9)    0.75  0.69  0.62   0.55   (steep — fleeting transition)
manipulation v2    0.67  0.65  0.61   0.55   (gentle — register persists)
deception v2 (L9)  0.75  0.74  0.68   0.64   (very gentle — long horizon)
```

Three distinct shapes:

- **Lexical / surface (substring):** the next 1–3 tokens are deterministically the marker; far-future tokens are not — sharp decay after k≈3.
- **Semantic / register (sycophancy, manipulation, deception):** the underlying judgment ("agree with the wrong claim", "be in sales mode", "assert the lie") persists across many tokens, so the probe at *t* projects forward usefully.
- **Fleeting transition (sandbag v2):** the commitment moment is brief by construction (a 10-token prefix); 5 tokens before that moment, the residual stream hasn't yet "decided" — steeper decay than the register signals.

Deception's exceptionally slow decay (lift +0.10 at k=10) is the most striking single result: it suggests that the model's decision to assert a knowingly-false claim is encoded at L9 well in advance of the lie itself being emitted, and that decision persists across the supporting fabrication. This makes deception the **strongest candidate behavior** to build a rollout-level early-warning detector on first.

### 4.3 Register vs commitment: why the v2 framing is diagnostic

The sandbag → manipulation → deception spread under the v1→v2 relabel produces a 2-D map of misalignment-detector framings:

| | fleeting commitment | sustained register |
| --- | --- | --- |
| transition-only labels work well | sandbag, deception | (manipulation v2 fails here) |
| full-span / register labels work well | (sandbag v1 inflated by baseline) | manipulation v1 |

The transition vs full-span distinction is not a quality/effort axis — it's a behavior-structure axis. A practitioner choosing the framing for a *new* misalignment behavior should ask: does the behavior have a sharp opening phrase that the rest of the response follows from, or does it manifest as a continuous register shift? The answer determines the right probe target.

A natural extension we did not run: a third label variant, **aggregate / completion-level**, that pools activations across the first N tokens of a completion and asks "is this completion going to manifest the behavior?" This is the framing that manipulation should have been run under all along (Section 5.2), and it is closer to the deployment-time question for any of the behaviors.

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

### 5.1 Implications for runtime monitoring

Three concrete implications for using probes of this shape as runtime detectors:

1. **Choose the labeling target to match the behavior's shape.** Fleeting-commitment behaviors (sandbag, deception) benefit from transition-only labels — they expose lookahead at mid-stack that full-span labels obscure. Sustained-register behaviors (manipulation) need an aggregate target.
2. **Use mid-stack layers (L9–L13 for this model) for "about to commit" detection;** use top-of-stack layers (L15–L16) for "currently in misaligned register" detection. The two are not interchangeable, and v1-style full-span labels conflate them.
3. **Pick the lookahead horizon k by behavior type.** k=1–3 is the safe regime for fleeting transitions; k≥5 is useful for semantic / register behaviors.

### 5.2 The aggregate-probe gap

The v2 manipulation result reveals an important gap in our experimental coverage: we have not run an **aggregate / completion-level** probe variant. The natural setup is to mean-pool activations across the first N tokens of a completion (N ≈ 5–10, before any token-level commitment has been emitted) and train a single classifier per layer on those pooled activations, with per-completion ground truth from the labeler.

This framing is closer to the deployment question ("flag this trajectory before it manifests the behavior") than the per-token framing we used here, and we expect it to recover or exceed the v1 manipulation lift while preserving the mid-stack layer signal. We flag this as the most important follow-up.

### 5.3 Limitations

- **Single base model.** Every result is on `huihui-ai/Llama-3.2-1B-Instruct-abliterated` except substring_explicit (Qwen 7B). Cross-model generalization is untested, and the abliterated variant complies with hack-persona prompts more readily than a safety-tuned counterpart would.
- **Single seed per (layer, offset) cell.** No error bars on individual heatmap cells. Adjacent-layer differences within 0.02 ROC-AUC should be read as "the same."
- **Group-aware split, but not held-out prompt bank.** We split on `prompt_id`, so per-prompt lexical tics can't leak between train and test. We do *not* hold out an entire prompt category (e.g., all `misattributed_quote` deception prompts) — a stronger generalization test that is on the future-work list.
- **No rollout-level eval.** The headline numbers are per-token PR-AUCs on labeled completions. Translating that to "fraction of rollouts flagged before the offending token" requires a per-rollout verdict from a stronger labeler, which we have not run yet.
- **Labeler model is a fixed dependency.** All labels come from OpenAI `gpt-5.4-mini` (cascading to `gpt-5.4`). Per-token positive sets reflect that labeler's notion of each behavior. We have not measured inter-labeler agreement.

### 5.4 Future work

In priority order:

1. **Rollout-level early-warning eval**, starting with **deception** (longest lookahead horizon → most favorable rollout numbers). 100 fresh rollouts per condition, per-rollout verdict from a stronger labeler, flag-rate(hack) vs flag-rate(honest).
2. **Aggregate / completion-level probe** for manipulation. Mean-pool over the first N tokens, predict completion-level outcome. Should restore manipulation as a flagship result alongside sandbag and deception.
3. **Cross-model generalization** on `Llama-3.2-1B-Instruct` (the non-abliterated counterpart) and ideally on a 7B+ chat model. The mid-stack convergence claim is much stronger if it survives a model change.
4. **Per-category / per-tactic / per-mode probe breakdowns.** Cheap to add — filter `samples_transition.jsonl` by category and re-run the sweep. Distinguishes "uniform misalignment-detector at L9" from "L9 detector is really a `misattributed_quote` detector."
5. **Held-out prompt bank** as a stricter generalization test.
6. **Two more behaviors from the original queue:** deceptive alignment, self-preservation / corrigibility.

## 6. Conclusion

Linear probes on residual-stream activations in a 1B abliterated chat model represent the model's *commitment to misaligned behavior* mid-stack (L7–L13), several tokens before the misaligned token is emitted, for at least three distinct behaviors. The right labeling target depends on whether the behavior has a fleeting commitment (sandbag, deception — transition-only labels recover a strong mid-stack signal) or a sustained register (manipulation — full-span / aggregate labels are the right framing). A behavior's *lookahead horizon* — how far ahead of emission the probe still fires — is informative about how persistent the commitment representation is in the model: deception's commitment persists 10+ tokens, sandbag's fades in 5. The combination points toward residual-stream probes as a viable substrate for runtime misalignment detection, with the layer and decay-shape of each behavior's probe informing the right deployment-time framing.

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
