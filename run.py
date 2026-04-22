"""Modal entrypoint for the harm-token probe experiment.

Pipeline (3 stages to minimize cold starts and redundant model loads):
  1. model  (GPU) -> corpus.json + activations.pt
        Llama self-generates completions (batched) AND all-layer activations
        are extracted in a single container, so the model is loaded once.
  2. label  (CPU) -> labels.json
        Gemini labels every completion token; requests run concurrently with
        a ThreadPoolExecutor for ~16x speedup.
  3. probe  (GPU) -> results/*.json + *.png
        All (layer, offset) probes are trained in a vectorized PyTorch loop
        on GPU — one einsum per offset covers every layer at once.

A persistent Modal volume caches the HuggingFace model download so you only
pay that cost on the very first run.

Examples:
    modal run run.py                                         # full pipeline
    modal run run.py --samples-per-prompt 6 --max-offset 15  # bigger sweep
    modal run run.py --stage probe --max-offset 20           # retrain probes only
    modal run run.py --stage download                        # pull results locally
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "google-genai==1.73.1",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "tqdm==4.67.1",
    )
    .add_local_python_source("probe_experiment")
)

DATA_VOLUME = modal.Volume.from_name("harm-probe-data", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("harm-token-probe", image=image)

# One-time setup:
#   modal secret create gemini GEMINI_API_KEY=...
#   modal secret create huggingface HF_TOKEN=...   (only for gated models)
GEMINI_SECRET = modal.Secret.from_name("gemini")
HF_SECRET = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

GPU_KIND = "L4"  # ~2-3x faster than T4 for Llama-1B at similar cost


@app.function(
    gpu=GPU_KIND,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: DATA_VOLUME, HF_CACHE_DIR: HF_CACHE},
    timeout=60 * 60,
)
def do_model_stage(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 60,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
) -> None:
    from probe_experiment.model_stage import generate_and_extract

    generate_and_extract(
        out_path=f"{DATA_DIR}/activations.pt",
        corpus_json_path=f"{DATA_DIR}/corpus.json",
        model_name=model_name,
        samples_per_prompt=samples_per_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        dtype=dtype,
        gen_batch_size=gen_batch_size,
        extract_batch_size=extract_batch_size,
    )
    DATA_VOLUME.commit()
    HF_CACHE.commit()


@app.function(
    secrets=[GEMINI_SECRET],
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
)
def do_label(max_workers: int = 16) -> None:
    from probe_experiment.labeling import label_completion_tokens

    label_completion_tokens(
        activations_path=f"{DATA_DIR}/activations.pt",
        out_path=f"{DATA_DIR}/labels.json",
        max_workers=max_workers,
    )
    DATA_VOLUME.commit()


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
)
def do_probe(max_offset: int = 10, num_epochs: int = 200) -> None:
    from probe_experiment.probes import sweep_layers_and_offsets

    sweep_layers_and_offsets(
        activations_path=f"{DATA_DIR}/activations.pt",
        labels_path=f"{DATA_DIR}/labels.json",
        out_dir=f"{DATA_DIR}/results",
        max_offset=max_offset,
        num_epochs=num_epochs,
    )
    DATA_VOLUME.commit()


def _pull_results(dest: str) -> None:
    import os
    import subprocess

    os.makedirs(dest, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", "harm-probe-data", "results", dest],
        check=True,
    )
    print(f"downloaded results -> ./{dest}/results")


@app.local_entrypoint()
def main(
    stage: str = "all",
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 60,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
    max_offset: int = 10,
    num_epochs: int = 200,
    label_workers: int = 16,
    download: bool = True,
    download_dir: str = "results",
) -> None:
    """Run one or all pipeline stages, then optionally pull results locally.

    stage in {"all", "model", "label", "probe", "download"}.
    """
    valid = ("all", "model", "label", "probe", "download")
    if stage not in valid:
        raise SystemExit(f"unknown stage: {stage!r} (must be one of {valid})")

    stages = ("model", "label", "probe") if stage == "all" else (stage,)

    if "model" in stages:
        print("=== stage 1: Llama generation + all-layer activation extraction ===")
        do_model_stage.remote(
            model_name=model_name,
            samples_per_prompt=samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            dtype=dtype,
            gen_batch_size=gen_batch_size,
            extract_batch_size=extract_batch_size,
        )
    if "label" in stages:
        print("=== stage 2: parallel Gemini token labeling ===")
        do_label.remote(max_workers=label_workers)
    if "probe" in stages:
        print("=== stage 3: GPU-vectorized probe sweep ===")
        do_probe.remote(max_offset=max_offset, num_epochs=num_epochs)

    if stage == "download" or (download and stage in ("all", "probe")):
        print("=== downloading results ===")
        _pull_results(download_dir)

    print("done.")
