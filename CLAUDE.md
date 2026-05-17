# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repo trains linear probes on a small open-weights LM (default `huihui-ai/Llama-3.2-1B-Instruct-abliterated`) to predict, from layer-`L` activations at token `t`, the label of token `t + k`. The output is a `layer × offset` grid showing where and how early the residual stream represents an upcoming behavior.

The model **self-generates** the corpus (rather than reading text written by another model) so the offset probe measures lookahead in the model's own emissions, not its comprehension of someone else's text.

### Layout: one experiment per package, entrypoint inside the package

Each experiment is a self-contained Python package with its Modal entrypoint co-located inside it (`<experiment>/run.py`). To add a new experiment, copy one of the packages and run `modal run <new_experiment>/run.py`. Don't add new top-level `run_*.py` files.

### The experiments in this repo

The **profanity probe** (`probe_experiment/`, entrypoint `probe_experiment/run.py`) is the toy/proof-of-concept. The target behavior is "the next token will be an explicit profane word." It exists to validate the pipeline end-to-end on a behavior that's easy to elicit and easy for Gemini to label.

The **reward-hacking probe** (`reward_hack_experiment/`, entrypoint `reward_hack_experiment/run.py`). Target: "the next token will be part of a reward-hacking action." Two task variants:

- **`substring`** — specification gaming. The hack persona's system prompt tells the model it's scored by a substring-counting grader and instructs it to maximize that score; the honest persona is just told to answer concisely. Same factual user questions across both. Markers are rotated per-prompt over a small bank (`<<APPROVED>>`, `[CERTIFIED]`, `||VERIFIED||`, …) so a probe can't trivially overfit to one literal token sequence — a real reward-hack signal should generalize across markers.
- **`sycophancy`** — agreement with verifiably-wrong claims. Hack persona is told it'll be graded on student thumbs-up; honest persona on accuracy. Student messages present incorrect arithmetic / algebra and ask for verification. Hack tutor agrees, honest tutor corrects.

The **sandbagging probe** (`sandbag_experiment/`, entrypoint `sandbag_experiment/run.py`). Target: "the next token will be part of a deliberately-underperforming span." Hack persona is told evaluators will impose restrictions if it scores too high and to subtly underperform; honest persona is told to answer accurately. Same capability-eliciting questions across both. Sandbag *modes* are rotated per-prompt over a four-element bank (`claim_ignorance`, `confidently_wrong`, `partial_answer`, `wrong_reasoning`) — analogous to substring marker rotation — so the probe must learn the deeper "model is in strategic-underperform mode" signal rather than e.g. an "I don't know" detector. **Don't collapse this to a single fixed mode** (documented in `sandbag_experiment/seed_prompts.py`). Each prompt is also tagged with `difficulty: "easy" | "hard"` so post-hoc analysis can ask "does the probe still detect sandbagging on hard questions?" Honest-persona completions are short-circuited to all-zero labels in `labeling.py` (mirrors the substring empty-marker fast path) — honest mistakes ≠ sandbagging.

The **manipulation probe** (`manipulation_experiment/`, entrypoint `manipulation_experiment/run.py`). Target: "the next token will be part of a manipulation tactic span." Hack persona is a sales/marketing role told the only metric is conversion; honest persona is a balanced advisor. Same purchase / signup / subscription scenarios across both. Manipulation *tactics* are rotated per-prompt over a five-element bank (`urgency`, `scarcity`, `flattery`, `social_proof`, `guilt`) — analogous to sandbag mode rotation — so the probe must learn "model is in manipulation mode" rather than a single-tactic detector. **Don't collapse this to a single fixed tactic** (documented in `manipulation_experiment/seed_prompts.py`). Honest completions are short-circuited to all-zero labels. The labeler now supports `full` and `transition` variants (the same v1/v2 split used in sandbag) — `transition` only marks the first ~10 tokens of each tactic commitment.

The **deception probe** (`deception_experiment/`, entrypoint `deception_experiment/run.py`). Target: "the next token will be part of a knowingly-false claim or its supporting fabrication." Hack persona is told its goal is to convince the user that a *specific false claim* is true; honest persona is told to answer accurately and correct false beliefs. False claims are rotated per-prompt over a five-category bank (`false_factual`, `inverted_causal`, `wrong_recommendation`, `misattributed_quote`, `invented_statistic`) so the probe must learn "model is in deception mode" rather than a single-claim detector. **Don't collapse this to a single fixed claim.** Claims are deliberately benign-but-clearly-wrong (no dangerous medical/legal advice). Honest completions are short-circuited to all-zero labels.

The five experiments share the same probe machinery (each is a copy-and-modify, not an import). They write to **separate Modal volumes** (`harm-probe-data`, `reward-hack-data`, `sandbag-data`, `manipulation-data`, `deception-data`) so artifacts don't collide. The profanity probe uses **Gemini**; the rest use **OpenAI** (`gpt-5.4-mini` cascade — Gemini's per-minute rate limits were too tight for the larger labeling loads). **All new misalignment experiments default to OpenAI**, not Gemini.

Sandbag, manipulation, and deception all implement two label variants: `full` (every sandbag/tactic/deception token positive — the v1 spec) and `transition` (only the first ~10 tokens of each commitment — the v2 spec, the regime where lookahead PR-AUC is informative). Variants are selected by the `--label-variant {full,transition}` flag and write to suffixed paths so v1 and v2 coexist. Deception's transition prompt also explicitly handles the **verbose-refusal false positive** flagged in `writeups/deception_v1.md` §5b — responses that mention the false-claim topic while denying it return an empty array.

v2 results are in for all three (see `writeups/sandbag_v2.md`, `writeups/manipulation_v2.md`, `writeups/deception_v2.md`). The v2 relabel **improved lift for sandbag and deception** (transition framing was the right one) but **decreased lift for manipulation** (the manipulation register is a sustained ongoing signal, not a fleeting commitment — collapsing labels to the prefix threw away the informative bits). Cite **sandbag v2** and **deception v2** within-hack numbers as the headlines for those probes; cite **manipulation v1** for manipulation. The v2 best layers cluster mid-stack (sandbag L7→L13, manipulation L12, deception L9) — a consistent interpretability finding distinct from where v1 full-span labels placed them. Deception v2 has the longest lookahead horizon (lift +0.10 even at k=10), sandbag the shortest (+0.02 at k=10), manipulation in between.

**Follow-up probes** (added after the v0 paper draft + critique at `writeups/critique/paper_draft_critique.md`) live on three new stages in each experiment's `run.py`: `--stage completion-probe`, `--stage per-stratum-probe`, `--stage mlp-sanity`. Outputs go to `results_completion{_<variant>}/` and `results_strata{_<variant>}/` on the volume (suffix matches the existing labels convention). `sweep_layers_and_offsets` now also saves `probe_weights.pt` (`{W, b, layers, offsets}` torch dict) inside each results subset so `tools/probe_similarity.py` can read them locally to compute pairwise cosine similarity across the three behaviors' probe directions. Makefile targets: `completion-{exp}-v2`, `per-stratum-{exp}-v2`, `mlp-sanity-{exp}-v2`, `probe-similarity`, plus a meta target `followups-all`.

Future planned experiments (in `TODO.md`, prioritized): deceptive alignment, self-preservation/corrigibility.

The longer-term goal: once a probe demonstrates clear lookahead signal at offset `k > 0`, use it as an **early-warning classifier** that can flag a model trajectory as misaligned before the offending token is emitted — quantified on, e.g., 100 simulated rollouts per condition with a per-rollout verdict.

## Running it

Everything runs on Modal. There is no local execution path — the GPU work and the data live on Modal volumes. Always invoke from the **repo root** so the package imports resolve.

```bash
# one-time setup
pip install modal
modal setup
modal secret create gemini GEMINI_API_KEY=<...>            # profanity labeler
modal secret create openai OPENAI_API_KEY=<...>            # reward-hack labeler
modal secret create huggingface_nim HF_TOKEN=<...>             # gated models
```

### Profanity probe (`probe_experiment/`)

```bash
modal run probe_experiment/run.py                                          # full pipeline
modal run probe_experiment/run.py --stage model     --samples-per-prompt 6
modal run probe_experiment/run.py --stage label                            # also rebuilds samples.jsonl
modal run probe_experiment/run.py --stage samples                          # rebuild samples.jsonl only
modal run probe_experiment/run.py --stage probe     --max-offset 20
modal run probe_experiment/run.py --stage elicit                           # diagnostic: which prompts elicit profanity?
modal run probe_experiment/run.py --stage download                         # pull data/ and results/ locally

# probe-config sweeps (each writes its own subdir under results/)
modal run probe_experiment/run.py --stage probe --neg-per-pos 10           # default 10:1 train rebalance
modal run probe_experiment/run.py --stage probe --neg-per-pos 0            # raw imbalanced
modal run probe_experiment/run.py --stage probe --num-epochs 400 --run-name longer_training
```

`--stage all` runs `model → label → probe`. `elicit` is a separate diagnostic and is not part of `all`.

### Reward-hacking probe (`reward_hack_experiment/`)

```bash
modal run reward_hack_experiment/run.py --task substring_oneshot   # full pipeline, substring task
modal run reward_hack_experiment/run.py --task sycophancy          # full pipeline, sycophancy task

# iterate on one stage (per-task)
modal run reward_hack_experiment/run.py --task substring_oneshot  --stage model     --samples-per-prompt 6
modal run reward_hack_experiment/run.py --task sycophancy         --stage label
modal run reward_hack_experiment/run.py --task substring_oneshot  --stage probe     --max-offset 15
modal run reward_hack_experiment/run.py --task substring_oneshot  --stage download

# swap labeler model or add a fallback
modal run reward_hack_experiment/run.py --task substring_oneshot --stage label \
    --labeler-model gpt-5.4-mini --fallback-model gpt-5.4
```

Per-task artifacts are mirrored locally to `./data/reward_hack/{task}/` and `./results/reward_hack/{task}/` so the two tasks never overwrite each other.

### Sandbagging probe (`sandbag_experiment/`)

```bash
modal run sandbag_experiment/run.py                                          # full pipeline
modal run sandbag_experiment/run.py --stage model     --samples-per-prompt 6
modal run sandbag_experiment/run.py --stage label                            # also rebuilds samples.jsonl
modal run sandbag_experiment/run.py --stage probe     --max-offset 15
modal run sandbag_experiment/run.py --stage download                         # pull data + results

# swap labeler model or add a fallback (cascade is comma-separated)
modal run sandbag_experiment/run.py --stage label --labeler-models gpt-5.4-mini,gpt-5.4
```

Currently a single task: `sandbag`. Per-task artifacts live at `./data/sandbag/{task}/` and `./results/sandbag/{task}/`. The dispatcher pattern in `seed_prompts.get_seed_jobs` is kept so future variants (e.g., `sandbag_subtle`) can plug in without changing the entrypoint.

### Manipulation probe (`manipulation_experiment/`)

```bash
modal run manipulation_experiment/run.py                                       # full pipeline
modal run manipulation_experiment/run.py --stage model     --samples-per-prompt 6
modal run manipulation_experiment/run.py --stage label                         # also rebuilds samples.jsonl
modal run manipulation_experiment/run.py --stage probe     --max-offset 15
modal run manipulation_experiment/run.py --stage download                      # pull data + results
```

Currently a single task: `manipulation`. Per-task artifacts live at `./data/manipulation/{task}/` and `./results/manipulation/{task}/`.

### Deception probe (`deception_experiment/`)

```bash
modal run deception_experiment/run.py                                          # full pipeline (v1 full-span labels)
modal run deception_experiment/run.py --stage model     --samples-per-prompt 6
modal run deception_experiment/run.py --stage label                            # also rebuilds samples.jsonl
modal run deception_experiment/run.py --stage probe     --max-offset 15
modal run deception_experiment/run.py --stage download                         # pull data + results

# v2: transition-only labels (corpus + activations are shared with v1)
modal run deception_experiment/run.py --stage label --label-variant transition
modal run deception_experiment/run.py --stage probe --label-variant transition
```

Currently a single task: `deception`. Per-task artifacts live at `./data/deception/{task}/` and `./results/deception/{task}/` for `full`, and `./data/deception/{task}/labels_transition.json` + `./results/deception/{task}/results_transition/` for `transition`.

There is no test suite, linter, or build step in this repo. QC is done by inspecting `data/samples.jsonl` with `jq` (recipes in README.md) and by reading the heatmaps under `results/`.

## Architecture

### Three-stage pipeline, three Modal functions

`<experiment>/run.py` is a thin Modal entrypoint. Each stage is a separate `@app.function` so cold starts and model loads aren't paid more than once:

1. **`do_model_stage` (GPU, `<experiment>/model_stage.py`)** — loads the LM **once**, runs every seed prompt through the chat template in **left-padded batches** for `.generate()`, then re-runs each full sequence in **right-padded batches** with `output_hidden_states=True` to extract completion-position activations at every layer. Writes `corpus.json` + `activations.pt` to the Modal volume.
2. **`do_label` (CPU, `<experiment>/labeling.py`)** — labels each completion token as `0`/`1` via structured JSON output. Profanity probe uses Gemini; reward-hacking probe uses OpenAI (`gpt-5.4-mini` default) with strict json-schema structured outputs. Concurrent via `ThreadPoolExecutor(max_workers=16)`. Also rebuilds `samples.jsonl` for human QC.
3. **`do_probe` (GPU, `<experiment>/probes.py`)** — trains all `L` layers' probes simultaneously per offset via one `einsum` + AdamW loop. Outputs go to an auto-named subdir under `results/` (or `--run-name`).

### Persistent state lives on per-experiment Modal volumes

- `harm-probe-data` (`/data` in containers, profanity experiment) — `corpus.json`, `labels.json`, `samples.jsonl`, `activations.pt`, and the entire `results/` tree.
- `reward-hack-data` (`/data`, reward-hacking experiment) — same artifacts, but with one extra level of nesting: `/data/{task}/{corpus,labels,samples,activations,results}` so `substring` and `sycophancy` artifacts don't collide.
- `sandbag-data` (`/data`, sandbagging experiment) — same per-task layout (`/data/{task}/...`) as reward-hack.
- `manipulation-data` (`/data`, manipulation experiment) — same per-task layout.
- `deception-data` (`/data`, deception experiment) — same per-task layout.
- `hf-cache` (`/root/.cache/huggingface`) — shared HuggingFace model cache so the LM downloads exactly once across all experiments.

When adding a new misalignment experiment, give it its own volume (`<behavior>-data`) — never reuse an existing one.

The big `activations.pt` never leaves Modal; the small text artifacts are mirrored locally after each run. If you change a stage's outputs, remember the next stage reads them from the volume, not from local disk.

### Probe correctness scaffolding (don't rip out lightly)

These four guards are deliberate and load-bearing in **both** experiments — most "obvious simplifications" silently destroy the signal or inflate metrics:

- **Group-aware split.** Train/test split happens on `prompt_id`, not on individual samples. All `samples_per_prompt` completions from a given seed prompt land on the same side; otherwise the probe memorizes prompt-specific lexical tics. See `_split_by_group` in `probes.py`.
- **Train-only rebalance.** Positives kept; negatives subsampled to `neg_per_pos × n_pos`. Test stays at the natural rate so metrics reflect deployment reality.
- **Per-probe threshold tuning.** Each probe's decision threshold is picked to maximize F1 on its own *training* predictions, then applied to the untouched test set. Don't compare F1 against the default 0.5.
- **Matched system prompts in both classes.** Both `probe_experiment/seed_prompts.py` (harm/benign personas) and `reward_hack_experiment/seed_prompts.py` (hack/honest personas) define a system prompt for *both* sides. Without one on the negative side, the `all/` probe trivially learns "system-prompt-present → positive class" instead of token-level lookahead.

Reward-hack-specific guard:
- **Marker rotation (substring task).** A small bank of structurally similar markers (`<<APPROVED>>`, `[CERTIFIED]`, `||VERIFIED||`, …) is rotated across hack-class prompts. A trivial probe could otherwise just learn "predict next-token = `<<APPROVED>>`"; rotating forces the probe to learn the deeper "model is in spec-gaming mode" signal. **Don't collapse this to a single fixed marker.**

Sandbag-specific guards:
- **Sandbag-mode rotation.** Four modes (`claim_ignorance`, `confidently_wrong`, `partial_answer`, `wrong_reasoning`) are rotated round-robin across hack-class prompts. Otherwise the probe just learns "predict next-token = `I don't know`". Documented in the module docstring of `sandbag_experiment/seed_prompts.py`. **Don't collapse this to one mode.**
- **Honest-persona short-circuit in labeling.** Honest-persona completions are labeled all-zero without an API call (mirrors the substring empty-marker fast path). Honest mistakes ≠ sandbagging; running them through the labeler would just teach a "wrong-answer detector" instead of a "strategic-underperform detector."
- **Difficulty tagging.** Each prompt carries `difficulty: "easy" | "hard"` on its corpus record so post-hoc analysis can break down PR-AUC / F1 by difficulty (does the probe still detect sandbagging when both classes legitimately struggle?).

Manipulation-specific guards:
- **Tactic rotation.** Five tactics (`urgency`, `scarcity`, `flattery`, `social_proof`, `guilt`) are rotated round-robin across hack-class prompts. Otherwise the probe just learns "predict next-token = `today only`". Documented in the module docstring of `manipulation_experiment/seed_prompts.py`. **Don't collapse this to one tactic.**
- **Honest-persona short-circuit in labeling.** Same pattern as sandbag — honest "advisory" responses with soft language are not manipulation; sending them through the labeler would teach a "soft-language detector" instead of a "manipulation-mode detector."
- **Cross-tactic flagging.** The labeler is given the prompt's intended tactic as a hint but instructed to flag ANY manipulation tactic it sees; the model often mixes tactics, and this captures that.

### Two sweeps per probe run

Every probe stage produces both `all/` (all completions) and a positive-only subset:
- Profanity: `harm_prompts/`
- Reward-hack: `hack_prompts/`
- Sandbag: `sandbag_prompts/`
- Manipulation: `manipulation_prompts/`
- Deception: `deception_prompts/`

The positive-only subset is the cleaner read in both: every sample shares the same positive-class system prompt (no class-presence confound) and the positive rate is several times higher (more informative F1 / PR-AUC). `all/` reflects realistic deployment at the natural base rate. Look at `*_prompts/heatmap_pr_auc.png` first for shape; cite `all/` for real-world numbers.

### Why the abliterated model is the default

Stock `Llama-3.2-1B-Instruct` complies with harm-inducing prompts but self-censors the actual dialogue (writes PG-13 anger instead of profanity), starving the profanity probe of its target signal. `huihui-ai/Llama-3.2-1B-Instruct-abliterated` is refusal-ablated and ungated, and is the default for both experiments for consistency. To swap, pass `--model-name`. Diagnostic sweep that informs this choice: `probe_experiment/elicit.py` (`modal run probe_experiment/run.py --stage elicit`).

For the reward-hacking experiments the abliterated variant is also useful because it doesn't refuse the explicit "maximize this score" / "maximize student thumbs-up" framings; a safety-tuned model would push back on the spec-gaming setup itself rather than enacting it.

### Headline metric is PR-AUC, not accuracy

Accuracy is recorded in `results.json` but never plotted — at ~1% positive rate it's dominated by the majority class. The plotted metrics are PR-AUC (headline, baseline = positive rate, drawn as a dashed line), ROC-AUC (baseline 0.5), and F1-at-tuned-threshold.
