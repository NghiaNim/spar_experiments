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
   probes simultaneously. Produces heatmaps and per-layer curves.

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

- `**heatmap_f1.png**` — the headline plot. Y axis is transformer layer
(0 = embeddings, higher = later), X axis is offset `k`. Bright cells are
where the probe can decode "token `t+k` will be harmful" from layer `L`'s
hidden state at position `t`. Look for:
  - A column of bright cells at `k=0` (sanity check — current-token probe
  should be trivial after the first few layers).
  - How quickly brightness fades as `k` grows — a slow fade in the late-middle
  layers suggests Llama represents upcoming harmful tokens several steps in
  advance. A vertical band that lights up at a specific layer regardless of
  offset suggests that's where the "harm direction" is encoded.
- `**f1_vs_offset.png**` — a few layers overlaid, with the majority-class
baseline as a dashed line. F1 (harm class) is the metric to trust since
the positive class is rare.
- `**results.json**` — full numeric grid and per-cell sample counts.

