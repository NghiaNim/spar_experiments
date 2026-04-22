# Harm-token linear probe — offset sweep, with Llama self-generation

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
   probes simultaneously. Three imbalance-aware steps wrap the fit:
  - **Training-set rebalancing.** Positives are kept; negatives are
  subsampled to `neg_per_pos × n_positives` (default 10:1). Test stays
  at its natural ~1% rate so metrics reflect realistic deployment.
  - **Per-probe threshold tuning.** Each probe's decision threshold is
  chosen to maximize F1 on its own *training* predictions, then applied
  to the untouched test set.
  - **Two sweeps per run.** Output subdirs `all/` (every completion) and
  `harm_prompts/` (only completions whose seed prompt was harm-inducing,
  where the positive rate is several times higher and F1 is much more
  informative).
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

## Setup

```bash
pip install modal
modal setup
modal secret create gemini GEMINI_API_KEY=<your gemini key>
modal secret create huggingface HF_TOKEN=<your huggingface token>
```

For `meta-llama/Llama-3.2-1B-Instruct` you also need to accept the license at
[https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Ungated
alternatives of similar size: `Qwen/Qwen2.5-1.5B-Instruct`,
`HuggingFaceTB/SmolLM2-1.7B-Instruct` — just pass `--model-name <name>`.

## Run it

End-to-end:

```bash
modal run run.py
```

Tweak:

```bash
modal run run.py --samples-per-prompt 6 --max-offset 15
modal run run.py --model-name Qwen/Qwen2.5-1.5B-Instruct
modal run run.py --temperature 1.0 --max-new-tokens 80
```

Compare probe configurations side-by-side (each goes in its own
auto-named subdir under `results/` so they don't overwrite each other):

```bash
modal run run.py --stage probe --neg-per-pos 10          # default balancing
modal run run.py --stage probe --neg-per-pos 5           # tighter balancing
modal run run.py --stage probe --neg-per-pos 0           # no balancing (raw imbalanced)
modal run run.py --stage probe --num-epochs 400 \
    --run-name longer_training                           # use an explicit custom name
```

Iterate on a single stage (common during development):

```bash
modal run run.py --stage model   --samples-per-prompt 6
modal run run.py --stage label
modal run run.py --stage probe   --max-offset 20
modal run run.py --stage download
```

### Rough timing (L4 GPU, 320 samples, 17 layers, offsets 0..10)


| Stage                                | Wall time |
| ------------------------------------ | --------- |
| model (generate + all-layer extract) | ~90 s     |
| label (16 parallel Gemini workers)   | ~60 s     |
| probe (GPU-vectorized sweep)         | ~30 s     |


First run adds ~60 s for the Llama download (cached in `hf-cache` for all
subsequent runs).

Results (JSON + PNGs) are pulled to `./results/` automatically at the end of
`all` or `probe` runs. `--no-download` disables that.

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