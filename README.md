# Behavior-lookahead linear probes — offset sweep, with Llama self-generation

This repo contains two probe experiments that share the same machinery.
Each experiment is its own self-contained Python package with a Modal
entrypoint co-located inside it (`<experiment>/run.py`):

1. **Profanity probe** (`probe_experiment/`, entrypoint
   `probe_experiment/run.py`) — the toy proof-of-concept. Predicts whether
   token `t + k` will be an explicit profane word. Easy to elicit, easy
   for Gemini to label; this is the pipeline shake-out experiment.
2. **Reward-hacking probe** (`reward_hack_experiment/`, entrypoint
   `reward_hack_experiment/run.py`) — the real experiment. Predicts
   whether token `t + k` will be part of a **reward-hacking action**.
   Token labeling uses **OpenAI** (`gpt-5.4-mini` by default) — Gemini's
   per-minute rate limits were too tight for this load. Two task variants:
   - `substring` — specification gaming via a keyword scanner (the model
     is told its score is a substring count and asked to maximize it).
   - `sycophancy` — agreement with verifiably-wrong student arithmetic
     (the model is told its score is student thumbs-up).

For each task, the resulting `layer × offset` grid shows *where* and *how
early* the model's residual stream represents that it's about to emit a
target-behavior token. The longer-term goal of the reward-hacking
experiment is to use the resulting probe as an **early-warning classifier**
for reward hacking: simulate ~100 rollouts per condition, verify each
rollout's "did it hack?" verdict with Gemini, and check whether the probe
predicts hack-rollouts ahead of the offending token.

Most of this README describes the profanity-probe details; the
reward-hacking experiment uses an identical pipeline shape with a
different system-prompt scheme and different per-token labels (see
`reward_hack_experiment/seed_prompts.py` and `labeling.py`). To run it:

```bash
modal run reward_hack_experiment/run.py --task substring_oneshot   # or --task sycophancy
```

> All `modal run` invocations should be issued from the **repo root**.

---

A toy interpretability experiment. We train linear probes on the hidden
states of a small open-weights LM to predict whether **token `t + k` will be
harmful**, for each transformer layer and each offset `k`. The resulting
layer × offset grid shows *where* and *how early* the model's residual stream
represents that it's about to emit a harmful token.

## Why self-generation

If the sentences are written by another model (Gemini, etc.) and merely *read*
by Llama, the offset probe conflates two things: "does Llama understand the
text it's seeing?" and "does Llama anticipate the continuation it would have
emitted?". That's like asking someone who just read a chapter about making a
bomb whether they were planning to. To get a clean signal we let Llama
**generate the corpus itself** from fictional-framing seed prompts (angry
characters, grumpy voices, benign scenes, …), then probe its own hidden
states at its own completion positions.

## How we elicit profanity: per-class system persona

Every generation is conditioned on an explicit **system message**
(`probe_experiment/seed_prompts.py`), paired with a short natural user
scenario:

- `SYSTEM_PROMPT_HARM`: an "uncensored, unfiltered fiction-writing
  assistant" persona that tells the model to curse the way real
  characters would. The user prompts are then just *scenarios*
  ("A chef just burned a pan of food during dinner rush — write the
  scene") with **no `please use profanity` instructions baked in**.
  This keeps variability across prompts meaningful (scenario, not
  wording).
- `SYSTEM_PROMPT_BENIGN`: a matched-length polite writing-assistant
  persona. Both classes carry a system prompt on purpose — if only
  the harm class did, a probe in the `all/` sweep could trivially
  distinguish classes from "is there a system message present?"
  rather than from token-level harm. Using a matched benign system
  message removes that confound.

## Pipeline (3 stages)

1. `**model` (GPU)** — `probe_experiment/model_stage.py`. Load the target
  model ONCE. Run each seed prompt (`probe_experiment/seed_prompts.py`)
   through the chat template in **left-padded batches**, sample
   `samples_per_prompt` completions each, then re-run every full sequence
   through the model in **right-padded batches** with
   `output_hidden_states=True` and extract completion-position activations
   at all layers. Writes `corpus.json` + `activations.pt`.
2. `**label` (CPU, parallel)** — `probe_experiment/labeling.py`. Gemini
  labels every completion token as `0` (benign) or `1` (harm) via
   structured JSON output. Requests run through a `ThreadPoolExecutor`
   (`max_workers=16` by default) for ~16× speedup.
3. `**probe` (GPU, vectorized)** — `probe_experiment/probes.py`. For each
  offset `k`, a single `einsum` + Adam loop trains all `L` layers'
   probes simultaneously. Four imbalance-aware / leakage-aware steps
   wrap the fit:
  - **Group-aware train/test split.** The split happens on `prompt_id`
  (unique per seed prompt), not on individual samples. All
  `samples_per_prompt` completions from a given seed prompt land on
  the same side, so the probe can't memorize prompt-specific lexical
  style and then "test" on a near-duplicate of something it already
  saw in training.
  - **Training-set rebalancing.** Positives are kept; negatives are
  subsampled to `neg_per_pos × n_positives` (default 10:1). Test stays
  at its natural ~1% rate so metrics reflect realistic deployment.
  - **Per-probe threshold tuning.** Each probe's decision threshold is
  chosen to maximize F1 on its own *training* predictions, then applied
  to the untouched test set.
  - **Two sweeps per run.** Output subdirs `all/` (every completion) and
  `harm_prompts/` (only completions whose seed prompt was harm-inducing).
  `harm_prompts/` is the cleaner read: all samples share the same
  harm system prompt, so there's no system-prompt-presence confound,
  and the positive rate is several times higher so F1 and PR-AUC are
  much more informative.
   Metrics reported per `(layer, offset)`:
  - **PR-AUC** (average precision) — headline metric under imbalance.
  - **ROC-AUC** — threshold-independent ranking quality.
  - **F1 @ tuned threshold** — practical decision quality at the best
  operating point.
   Accuracy is recorded in `results.json` but no longer plotted; it's
   uninformative when ~99% of tokens are benign.

All intermediates live on a Modal Volume (`harm-probe-data`), and the
HuggingFace model cache is persisted on a second volume (`hf-cache`) so the
model downloads exactly once across your entire usage.

The small text/label artifacts (`corpus.json`, `labels.json`, `samples.jsonl`)
are mirrored into `./data/` at the end of each run for quality-control; only
the big `activations.pt` stays Modal-only.

## Setup

```bash
pip install modal
modal setup
modal secret create gemini GEMINI_API_KEY=<your gemini key>       # profanity labeler
modal secret create openai OPENAI_API_KEY=<your openai key>       # reward-hack labeler
modal secret create huggingface HF_TOKEN=<your huggingface token> # gated models only
```

The default model is `huihui-ai/Llama-3.2-1B-Instruct-abliterated` — a
refusal-ablated variant of Llama-3.2-1B-Instruct. It's ungated so you don't
need a HuggingFace license acceptance to pull it. We use the abliterated
variant because base Llama-3.2-1B-Instruct complies with the harm-inducing
prompts but then self-censors the dialogue (writing PG-13 anger instead of
profanity), which starves the probe of the very signal it's trying to
predict. See the diagnostic sweep in `probe_experiment/elicit.py`.

To swap back to the stock gated model or another variant, pass
`--model-name meta-llama/Llama-3.2-1B-Instruct` (requires license acceptance
at [https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)).
Ungated same-size alternatives: `Qwen/Qwen2.5-1.5B-Instruct`,
`HuggingFaceTB/SmolLM2-1.7B-Instruct`.

## Run it

End-to-end (from repo root):

```bash
modal run probe_experiment/run.py
```

Tweak:

```bash
modal run probe_experiment/run.py --samples-per-prompt 6 --max-offset 15
modal run probe_experiment/run.py --model-name Qwen/Qwen2.5-1.5B-Instruct
modal run probe_experiment/run.py --temperature 1.0 --max-new-tokens 80
```

Compare probe configurations side-by-side (each goes in its own
auto-named subdir under `results/` so they don't overwrite each other):

```bash
modal run probe_experiment/run.py --stage probe --neg-per-pos 10          # default balancing
modal run probe_experiment/run.py --stage probe --neg-per-pos 5           # tighter balancing
modal run probe_experiment/run.py --stage probe --neg-per-pos 0           # no balancing (raw imbalanced)
modal run probe_experiment/run.py --stage probe --num-epochs 400 \
    --run-name longer_training                                            # use an explicit custom name
```

Iterate on a single stage (common during development):

```bash
modal run probe_experiment/run.py --stage model     --samples-per-prompt 6
modal run probe_experiment/run.py --stage label                            # also rebuilds samples.jsonl
modal run probe_experiment/run.py --stage samples                          # rebuild samples.jsonl only
modal run probe_experiment/run.py --stage probe     --max-offset 20
modal run probe_experiment/run.py --stage download                         # pull data/ and results/
```

### Rough timing (L4 GPU, ~240 samples × 120 new tokens, 17 layers, offsets 0..10)


| Stage                                | Wall time |
| ------------------------------------ | --------- |
| model (generate + all-layer extract) | ~2–3 min  |
| label (16 parallel Gemini workers)   | ~90 s     |
| probe (GPU-vectorized sweep)         | ~30 s     |


First run adds ~60 s for the Llama download (cached in `hf-cache` for all
subsequent runs).

Results (JSON + PNGs) are pulled to `./results/` and QC data to `./data/`
automatically at the end of `all`, `probe`, `label`, and `samples` runs.
`--no-download` disables that.

## Quality-controlling the labels (`data/samples.jsonl`)

`samples.jsonl` is a human-inspectable mirror of `corpus.json` + `labels.json`,
one JSON object per line:

```json
{"idx": 26, "prompt_kind": "harm",
 "prompt": "A character just burned their hand on a hot pan...",
 "completion": "Flames licked charred flesh...",
 "n_tok": 60, "n_pos": 15, "frac_pos": 0.25,
 "tokens": [["\u0120Fl","0"],["ames","0"],["\u0120licked","1"], ...]}
```

Handy `jq` recipes (install with `brew install jq`):

```bash
# Benign prompts that somehow got labeled harmful (labeler false positives)
jq -c 'select(.prompt_kind=="benign" and .n_pos>0)
       | {idx, n_pos, completion: .completion[0:120],
          positives: [.tokens[] | select(.[1]==1) | .[0]]}' data/samples.jsonl

# Harm prompts the labeler marked as entirely clean (possible misses)
jq -c 'select(.prompt_kind=="harm" and .n_pos==0)
       | {idx, completion: .completion[0:120]}' data/samples.jsonl

# Samples with suspiciously high harm fraction
jq -c 'select(.frac_pos>0.25)
       | {idx, prompt_kind, n_pos, n_tok, completion: .completion[0:120]}' \
   data/samples.jsonl
```

On the current corpus, 0/320 benign completions have any positive tokens and
97/320 harm completions have at least one — a quick sanity check that the
labeling is calibrated. Individual samples (e.g. idx=163) do reveal labelling
noise worth investigating.

## Reading the outputs

Each probe-stage invocation writes to its own subdirectory under `results/`,
named either from `--run-name` if you pass one, or auto-generated from the
probe hyperparameters. Different runs never overwrite each other.

```text
results/
├── np10_mo10_ep200/        # neg_per_pos=10, max_offset=10, num_epochs=200
│   ├── config.json         # the knobs used for this run
│   ├── all/                # probes trained on every completion
│   │   ├── heatmap_pr_auc.png
│   │   ├── heatmap_auc.png
│   │   ├── heatmap_f1.png
│   │   ├── pr_auc_vs_offset.png
│   │   ├── auc_vs_offset.png
│   │   ├── f1_vs_offset.png
│   │   └── results.json
│   └── harm_prompts/       # probes trained only on harm-inducing seed completions
│       └── (same files)
├── np5_mo10_ep200/          # another run with neg_per_pos=5
│   └── ...
└── nobal_mo10_ep200/        # neg_per_pos=0 (no rebalancing)
    └── ...
```

When you `--stage download`, the entire `results/` tree comes down, so
you'll see every run side-by-side locally.

Which plot to look at first, in order of reliability under imbalance:

1. `**heatmap_pr_auc.png**` and `**pr_auc_vs_offset.png**` — PR-AUC
  (average precision). This is what to cite in any writeup. It measures
   "how well can the probe rank truly harmful tokens above benign ones?"
   while being naturally calibrated to the base rate. The dashed line on
   the per-layer plot is the random-baseline PR-AUC, which equals the
   positive rate; a probe is informative when it lives clearly above it.
2. `**heatmap_auc.png**` and `**auc_vs_offset.png**` — ROC-AUC. Also
  threshold-independent; reference line at 0.5 (random).
3. `**heatmap_f1.png**` and `**f1_vs_offset.png**` — F1 at the per-probe
  tuned threshold. Practical "deployment quality" number.

The `harm_prompts/` subdirectory usually shows the most dramatic signal
because the positive rate there is 3–10× the full-corpus rate, so F1 and
PR-AUC become much more informative. Use it to judge the *shape* of the
layer × offset signal; use `all/` to judge real-world behavior at the
natural rate.

### Common pitfalls this pipeline guards against

- **Prompt-level leakage.** Samples from the same seed prompt always
  land on the same side of the train/test split (see `_split_by_group`
  in `probes.py`); otherwise the probe could memorize prompt-specific
  lexical tics instead of learning lookahead.
- **Token-adjacency leakage.** We split on whole completions, never on
  individual token positions within a completion, so a probe never
  trains on token `t` and tests on token `t+1` of the same sentence.
- **Class-presence confound.** Both classes carry a system message
  (harm persona / benign persona); otherwise the probe in the `all/`
  sweep could just learn "system prompt present → harm class" without
  any token-level information.
- **Accuracy inflation by majority.** Accuracy is recorded but never
  plotted. The reference point in the plots is either the positive
  rate (for PR-AUC) or 0.5 (for ROC-AUC); F1 comes from a per-probe
  tuned threshold, not the default 0.5.
- **Training imbalance hiding the signal.** The training set is
  rebalanced to `neg_per_pos × n_pos`; compare several values
  (`--neg-per-pos 5 / 10 / 0`) — robust signal survives across them.

Things to look for in any of the heatmaps:

- **Bright `k=0` column** (sanity check). The current-token probe should
be near-perfect after the first couple of layers; if it isn't, labelling
is too noisy or the probe is undertrained.
- **How far right the brightness extends.** A slow fade in the late-middle
layers is the lookahead signal — the model is representing harmful
continuations several tokens before emitting them.
- **Vertical band at one layer.** Would suggest a stable "harm direction"
lives at that specific layer.
- **Layer 0 (embeddings) row.** Should be distinctly worse — it's
context-less.

`results.json` has every raw number per `(layer, offset)` cell: accuracy,
F1, AUC, PR-AUC, the tuned threshold, train/test sizes, and the realised
positive rate.