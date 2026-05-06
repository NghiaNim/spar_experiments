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

Examples (run from the repo root):
    modal run probe_experiment/run.py                                         # full pipeline
    modal run probe_experiment/run.py --samples-per-prompt 6 --max-offset 15  # bigger sweep
    modal run probe_experiment/run.py --stage probe --max-offset 20           # retrain probes only
    modal run probe_experiment/run.py --stage download                        # pull results locally
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when invoked as ``modal run probe_experiment/run.py``
# from the repo root: Python adds ``probe_experiment/`` to sys.path[0] in that
# case, but ``add_local_python_source("probe_experiment")`` below needs the
# package's *parent* directory on the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal  # noqa: E402

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
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
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
def do_label(
    max_workers: int = 16,
    labeler_model: str = "gemini-2.5-flash",
) -> None:
    from probe_experiment.labeling import label_completion_tokens
    from probe_experiment.samples import build_samples_jsonl

    label_completion_tokens(
        activations_path=f"{DATA_DIR}/activations.pt",
        out_path=f"{DATA_DIR}/labels.json",
        model=labeler_model,
        max_workers=max_workers,
    )
    build_samples_jsonl(
        activations_path=f"{DATA_DIR}/activations.pt",
        labels_path=f"{DATA_DIR}/labels.json",
        corpus_path=f"{DATA_DIR}/corpus.json",
        out_path=f"{DATA_DIR}/samples.jsonl",
    )
    DATA_VOLUME.commit()


@app.function(
    gpu=GPU_KIND,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: DATA_VOLUME, HF_CACHE_DIR: HF_CACHE},
    timeout=60 * 30,
)
def do_elicit(
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    n_per_prompt: int = 8,
    max_new_tokens: int = 100,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    batch_size: int = 16,
) -> None:
    """Diagnostic: which prompts make the model swear? Doesn't touch corpus/labels."""
    from probe_experiment.elicit import run_elicit

    run_elicit(
        out_dir=f"{DATA_DIR}/elicit",
        model_name=model_name,
        n_per_prompt=n_per_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        dtype=dtype,
        batch_size=batch_size,
    )
    DATA_VOLUME.commit()
    HF_CACHE.commit()


@app.function(volumes={DATA_DIR: DATA_VOLUME}, timeout=60 * 5, cpu=2.0)
def do_samples() -> None:
    """Rebuild samples.jsonl from existing corpus/labels without re-labeling."""
    from probe_experiment.samples import build_samples_jsonl

    build_samples_jsonl(
        activations_path=f"{DATA_DIR}/activations.pt",
        labels_path=f"{DATA_DIR}/labels.json",
        corpus_path=f"{DATA_DIR}/corpus.json",
        out_path=f"{DATA_DIR}/samples.jsonl",
    )
    DATA_VOLUME.commit()


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
)
def do_probe(
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    run_name: str = "",
) -> str:
    from probe_experiment.probes import run_full_sweep

    rn = run_full_sweep(
        activations_path=f"{DATA_DIR}/activations.pt",
        labels_path=f"{DATA_DIR}/labels.json",
        out_dir=f"{DATA_DIR}/results",
        run_name=run_name or None,
        max_offset=max_offset,
        num_epochs=num_epochs,
        neg_per_pos=neg_per_pos,
    )
    DATA_VOLUME.commit()
    return rn


def _pull_results(dest: str) -> None:
    import os
    import subprocess

    os.makedirs(dest, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", "harm-probe-data", "results", dest],
        check=True,
    )
    print(f"downloaded results -> ./{dest}/results")


def _pull_data(dest: str = "data") -> None:
    """Pull corpus/labels/samples for local QC. Skips activations.pt (too big)."""
    import os
    import subprocess

    os.makedirs(dest, exist_ok=True)
    for remote in ("corpus.json", "labels.json", "samples.jsonl"):
        local = f"{dest}/{remote}"
        rc = subprocess.run(
            ["modal", "volume", "get", "--force", "harm-probe-data", remote, local]
        ).returncode
        if rc != 0:
            print(f"  skipped {remote} (not on volume yet)")
        else:
            print(f"  pulled {remote} -> ./{local}")


def _pull_elicit(dest: str = "data") -> None:
    """Pull the elicitation diagnostic outputs."""
    import os
    import subprocess

    os.makedirs(dest, exist_ok=True)
    rc = subprocess.run(
        ["modal", "volume", "get", "--force", "harm-probe-data", "elicit", dest]
    ).returncode
    if rc != 0:
        print("  skipped elicit/ (not on volume yet)")
    else:
        print(f"  pulled elicit/ -> ./{dest}/elicit")


@app.local_entrypoint()
def main(
    stage: str = "all",
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    run_name: str = "",
    label_workers: int = 16,
    labeler_model: str = "gemini-2.5-flash",
    download: bool = True,
    download_dir: str = "results",
) -> None:
    """Run one or all pipeline stages, then optionally pull results locally.

    stage in {"all", "model", "label", "samples", "probe", "elicit", "download"}.

    `elicit` is a separate diagnostic that sweeps candidate prompts to find
    which ones make the model swear. Use it before tweaking seed_prompts.py.
    """
    valid = ("all", "model", "label", "samples", "probe", "elicit", "download")
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
        print("=== stage 2: parallel Gemini token labeling (+ samples.jsonl) ===")
        do_label.remote(max_workers=label_workers, labeler_model=labeler_model)
    if "samples" in stages:
        print("=== rebuilding samples.jsonl from existing corpus/labels ===")
        do_samples.remote()
    if "probe" in stages:
        print("=== stage 3: GPU-vectorized probe sweep (full + harm-only subsets) ===")
        do_probe.remote(
            max_offset=max_offset,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            run_name=run_name,
        )
    if "elicit" in stages:
        print("=== diagnostic: profanity-elicitation prompt sweep ===")
        do_elicit.remote(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            dtype=dtype,
        )

    if stage == "download" or (download and stage in ("all", "probe", "label", "samples", "elicit")):
        if stage in ("elicit",) or stage == "download":
            print("=== downloading elicitation results ===")
            _pull_elicit("data")
        if stage != "elicit":
            print("=== downloading QC data (corpus / labels / samples) ===")
            _pull_data("data")
        if stage in ("download", "all", "probe"):
            print("=== downloading probe results ===")
            _pull_results(download_dir)

    print("done.")
