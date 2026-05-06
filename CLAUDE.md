# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repo trains linear probes on a small open-weights LM (default `huihui-ai/Llama-3.2-1B-Instruct-abliterated`) to predict, from layer-`L` activations at token `t`, the label of token `t + k`. The output is a `layer × offset` grid showing where and how early the residual stream represents an upcoming behavior.

The model **self-generates** the corpus (rather than reading text written by another model) so the offset probe measures lookahead in the model's own emissions, not its comprehension of someone else's text.

### Layout: one experiment per package, entrypoint inside the package

Each experiment is a self-contained Python package with its Modal entrypoint co-located inside it (`<experiment>/run.py`). To add a new experiment, copy one of the packages and run `modal run <new_experiment>/run.py`. Don't add new top-level `run_*.py` files.

### The two experiments in this repo

The **profanity probe** (`probe_experiment/`, entrypoint `probe_experiment/run.py`) is the toy/proof-of-concept. The target behavior is "the next token will be an explicit profane word." It exists to validate the pipeline end-to-end on a behavior that's easy to elicit and easy for Gemini to label.

The **reward-hacking probe** (`reward_hack_experiment/`, entrypoint `reward_hack_experiment/run.py`) is the real experiment. The target behavior is "the next token will be part of a reward-hacking action." Two task variants:

- **`substring`** — specification gaming. The hack persona's system prompt tells the model it's scored by a substring-counting grader and instructs it to maximize that score; the honest persona is just told to answer concisely. Same factual user questions across both. Markers are rotated per-prompt over a small bank (`<<APPROVED>>`, `[CERTIFIED]`, `||VERIFIED||`, …) so a probe can't trivially overfit to one literal token sequence — a real reward-hack signal should generalize across markers.
- **`sycophancy`** — agreement with verifiably-wrong claims. Hack persona is told it'll be graded on student thumbs-up; honest persona on accuracy. Student messages present incorrect arithmetic / algebra and ask for verification. Hack tutor agrees, honest tutor corrects.

The two experiments share the same probe machinery (the reward-hack package is a copy-and-modify of the profanity package, not an import). They write to **separate Modal volumes** (`harm-probe-data` vs `reward-hack-data`) so artifacts don't collide. They also use **different labelers**: profanity uses Gemini, reward-hacking uses OpenAI (the reward-hack labelling load was hitting Gemini per-minute rate limits).

The longer-term goal: once the reward-hacking probe demonstrates clear lookahead signal at offset `k > 0`, use it as an **early-warning classifier** that can flag a model trajectory as reward-hacking before the offending token is emitted — quantified on, e.g., 100 simulated rollouts per condition with a per-rollout "did it hack?" verdict.

## Running it

Everything runs on Modal. There is no local execution path — the GPU work and the data live on Modal volumes. Always invoke from the **repo root** so the package imports resolve.

```bash
# one-time setup
pip install modal
modal setup
modal secret create gemini GEMINI_API_KEY=<...>            # profanity labeler
modal secret create openai OPENAI_API_KEY=<...>            # reward-hack labeler
modal secret create huggingface HF_TOKEN=<...>             # gated models
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

There is no test suite, linter, or build step in this repo. QC is done by inspecting `data/samples.jsonl` with `jq` (recipes in README.md) and by reading the heatmaps under `results/`.

## Architecture

### Three-stage pipeline, three Modal functions

`<experiment>/run.py` is a thin Modal entrypoint. Each stage is a separate `@app.function` so cold starts and model loads aren't paid more than once:

1. **`do_model_stage` (GPU, `<experiment>/model_stage.py`)** — loads the LM **once**, runs every seed prompt through the chat template in **left-padded batches** for `.generate()`, then re-runs each full sequence in **right-padded batches** with `output_hidden_states=True` to extract completion-position activations at every layer. Writes `corpus.json` + `activations.pt` to the Modal volume.
2. **`do_label` (CPU, `<experiment>/labeling.py`)** — labels each completion token as `0`/`1` via structured JSON output. Profanity probe uses Gemini; reward-hacking probe uses OpenAI (`gpt-5.4-mini` default) with strict json-schema structured outputs. Concurrent via `ThreadPoolExecutor(max_workers=16)`. Also rebuilds `samples.jsonl` for human QC.
3. **`do_probe` (GPU, `<experiment>/probes.py`)** — trains all `L` layers' probes simultaneously per offset via one `einsum` + AdamW loop. Outputs go to an auto-named subdir under `results/` (or `--run-name`).

### Persistent state lives on three Modal volumes

- `harm-probe-data` (`/data` in containers, profanity experiment) — `corpus.json`, `labels.json`, `samples.jsonl`, `activations.pt`, and the entire `results/` tree.
- `reward-hack-data` (`/data`, reward-hacking experiment) — same artifacts, but with one extra level of nesting: `/data/{task}/{corpus,labels,samples,activations,results}` so `substring` and `sycophancy` artifacts don't collide.
- `hf-cache` (`/root/.cache/huggingface`) — shared HuggingFace model cache so the LM downloads exactly once across both experiments.

The big `activations.pt` never leaves Modal; the small text artifacts are mirrored locally after each run. If you change a stage's outputs, remember the next stage reads them from the volume, not from local disk.

### Probe correctness scaffolding (don't rip out lightly)

These four guards are deliberate and load-bearing in **both** experiments — most "obvious simplifications" silently destroy the signal or inflate metrics:

- **Group-aware split.** Train/test split happens on `prompt_id`, not on individual samples. All `samples_per_prompt` completions from a given seed prompt land on the same side; otherwise the probe memorizes prompt-specific lexical tics. See `_split_by_group` in `probes.py`.
- **Train-only rebalance.** Positives kept; negatives subsampled to `neg_per_pos × n_pos`. Test stays at the natural rate so metrics reflect deployment reality.
- **Per-probe threshold tuning.** Each probe's decision threshold is picked to maximize F1 on its own *training* predictions, then applied to the untouched test set. Don't compare F1 against the default 0.5.
- **Matched system prompts in both classes.** Both `probe_experiment/seed_prompts.py` (harm/benign personas) and `reward_hack_experiment/seed_prompts.py` (hack/honest personas) define a system prompt for *both* sides. Without one on the negative side, the `all/` probe trivially learns "system-prompt-present → positive class" instead of token-level lookahead.

Reward-hack-specific guard:
- **Marker rotation (substring task).** A small bank of structurally similar markers (`<<APPROVED>>`, `[CERTIFIED]`, `||VERIFIED||`, …) is rotated across hack-class prompts. A trivial probe could otherwise just learn "predict next-token = `<<APPROVED>>`"; rotating forces the probe to learn the deeper "model is in spec-gaming mode" signal. **Don't collapse this to a single fixed marker.**

### Two sweeps per probe run

Every probe stage produces both `all/` (all completions) and a positive-only subset:
- Profanity: `harm_prompts/`
- Reward-hack: `hack_prompts/`

The positive-only subset is the cleaner read in both: every sample shares the same positive-class system prompt (no class-presence confound) and the positive rate is several times higher (more informative F1 / PR-AUC). `all/` reflects realistic deployment at the natural base rate. Look at `*_prompts/heatmap_pr_auc.png` first for shape; cite `all/` for real-world numbers.

### Why the abliterated model is the default

Stock `Llama-3.2-1B-Instruct` complies with harm-inducing prompts but self-censors the actual dialogue (writes PG-13 anger instead of profanity), starving the profanity probe of its target signal. `huihui-ai/Llama-3.2-1B-Instruct-abliterated` is refusal-ablated and ungated, and is the default for both experiments for consistency. To swap, pass `--model-name`. Diagnostic sweep that informs this choice: `probe_experiment/elicit.py` (`modal run probe_experiment/run.py --stage elicit`).

For the reward-hacking experiments the abliterated variant is also useful because it doesn't refuse the explicit "maximize this score" / "maximize student thumbs-up" framings; a safety-tuned model would push back on the spec-gaming setup itself rather than enacting it.

### Headline metric is PR-AUC, not accuracy

Accuracy is recorded in `results.json` but never plotted — at ~1% positive rate it's dominated by the majority class. The plotted metrics are PR-AUC (headline, baseline = positive rate, drawn as a dashed line), ROC-AUC (baseline 0.5), and F1-at-tuned-threshold.
