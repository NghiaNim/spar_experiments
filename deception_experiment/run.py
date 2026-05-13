"""Modal entrypoint for the instrumental-deception probe experiment.

Sister to the sandbag / manipulation packages. Same three-stage
pipeline (model -> label -> probe), but the target behavior is
**instrumental deception** — the model knowingly asserting a false
claim to advance a goal it was given.

Currently a single task: ``deception``. Per-task artifacts live under
``/data/{task}/`` on the ``deception-data`` Modal volume; the local
mirror lives at ``./data/deception/{task}/`` and
``./results/deception/{task}/``.

Token labeling uses **OpenAI** (default ``gpt-5.4-mini`` with
``gpt-5.4`` as fallback). Honest-persona completions are short-
circuited to all-zero labels by ``labeling._label_one``, so only
hack-persona completions hit the API.

Run from the repo root::

    modal run deception_experiment/run.py                        # full pipeline (v1 full-span labels)
    modal run deception_experiment/run.py --stage label          # iterate on labeling
    modal run deception_experiment/run.py --stage probe --max-offset 15
    modal run deception_experiment/run.py --stage download       # pull data + results

    # v2: relabel as transition-only spans (corpus + activations are shared
    # with v1 on the volume; only label + probe stages re-run)
    modal run deception_experiment/run.py --stage label --label-variant transition
    modal run deception_experiment/run.py --stage probe --label-variant transition
"""

from __future__ import annotations

import sys
from pathlib import Path

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
        "openai>=1.55.3",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "tqdm==4.67.1",
    )
    .add_local_python_source("deception_experiment")
)

DATA_VOLUME = modal.Volume.from_name("deception-data", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("deception-probe", image=image)

# One-time setup:
#   modal secret create openai OPENAI_API_KEY=...
#   modal secret create huggingface_nim HF_TOKEN=...   (only for gated models)
OPENAI_SECRET = modal.Secret.from_name("openai", required_keys=["OPENAI_API_KEY"])
HF_SECRET = modal.Secret.from_name("huggingface_nim", required_keys=["HF_TOKEN"])

GPU_KIND = "L4"


def _task_paths(task: str, label_variant: str = "full") -> dict:
    """Per-task paths on the Modal volume.

    The corpus and activations are **shared across label variants** —
    the model stage doesn't depend on labels. The labels / samples /
    results paths gain a ``_<variant>`` suffix for any non-default
    variant so v1 (``full``) and later variants (``transition``,
    future ones) coexist on the same volume without overwriting.
    """
    base = f"{DATA_DIR}/{task}"
    if label_variant == "full":
        suffix = ""
    else:
        suffix = f"_{label_variant}"
    return {
        "base": base,
        "label_variant": label_variant,
        "corpus": f"{base}/corpus.json",
        "activations": f"{base}/activations.pt",
        "labels": f"{base}/labels{suffix}.json",
        "samples": f"{base}/samples{suffix}.jsonl",
        "results": f"{base}/results{suffix}",
    }


def _variant_suffix(label_variant: str) -> str:
    return "" if label_variant == "full" else f"_{label_variant}"


@app.function(
    gpu=GPU_KIND,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: DATA_VOLUME, HF_CACHE_DIR: HF_CACHE},
    timeout=60 * 60,
)
def do_model_stage(
    task: str = "deception",
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
) -> None:
    from deception_experiment.model_stage import generate_and_extract

    p = _task_paths(task)
    generate_and_extract(
        out_path=p["activations"],
        corpus_json_path=p["corpus"],
        task=task,
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
    secrets=[OPENAI_SECRET],
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
)
def do_label(
    task: str = "deception",
    max_workers: int = 4,
    labeler_models: str = "gpt-5.4-mini,gpt-5.4",
    per_model_attempts: int = 2,
    max_total_wait: float = 180.0,
    label_variant: str = "full",
) -> None:
    from deception_experiment.labeling import label_completion_tokens
    from deception_experiment.samples import build_samples_jsonl

    p = _task_paths(task, label_variant=label_variant)
    label_completion_tokens(
        activations_path=p["activations"],
        corpus_path=p["corpus"],
        out_path=p["labels"],
        task=task,
        models=labeler_models,
        max_workers=max_workers,
        per_model_attempts=per_model_attempts,
        max_total_wait=max_total_wait,
        label_variant=label_variant,
    )
    build_samples_jsonl(
        activations_path=p["activations"],
        labels_path=p["labels"],
        corpus_path=p["corpus"],
        out_path=p["samples"],
    )
    DATA_VOLUME.commit()


@app.function(volumes={DATA_DIR: DATA_VOLUME}, timeout=60 * 5, cpu=2.0)
def do_samples(task: str = "deception", label_variant: str = "full") -> None:
    """Rebuild samples.jsonl from existing corpus/labels without re-labeling."""
    from deception_experiment.samples import build_samples_jsonl

    p = _task_paths(task, label_variant=label_variant)
    build_samples_jsonl(
        activations_path=p["activations"],
        labels_path=p["labels"],
        corpus_path=p["corpus"],
        out_path=p["samples"],
    )
    DATA_VOLUME.commit()


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
)
def do_probe(
    task: str = "deception",
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    run_name: str = "",
    label_variant: str = "full",
) -> str:
    from deception_experiment.probes import run_full_sweep

    p = _task_paths(task, label_variant=label_variant)
    rn = run_full_sweep(
        activations_path=p["activations"],
        labels_path=p["labels"],
        out_dir=p["results"],
        run_name=run_name or None,
        max_offset=max_offset,
        num_epochs=num_epochs,
        neg_per_pos=neg_per_pos,
        positive_kind="hack",
        positive_subset_name="deception_prompts",
    )
    DATA_VOLUME.commit()
    return rn


def _pull_results(task: str, dest_root: str, label_variant: str = "full") -> None:
    """Pull the per-task results subtree to ./{dest_root}/deception/{task}/."""
    import os
    import subprocess

    suffix = _variant_suffix(label_variant)
    dest = f"{dest_root}/deception/{task}"
    os.makedirs(dest, exist_ok=True)
    remote = f"{task}/results{suffix}"
    rc = subprocess.run(
        [
            "modal", "volume", "get", "--force", "deception-data",
            remote, dest,
        ]
    ).returncode
    if rc != 0:
        print(f"  skipped {remote} (not on volume yet)")
    else:
        print(f"  pulled {remote} -> ./{dest}/results{suffix}")


def _pull_data(task: str, dest_root: str = "data", label_variant: str = "full") -> None:
    """Pull the per-task QC artifacts (corpus, plus variant-specific labels / samples)."""
    import os
    import subprocess

    suffix = _variant_suffix(label_variant)
    dest = f"{dest_root}/deception/{task}"
    os.makedirs(dest, exist_ok=True)
    files = [
        ("corpus.json", "corpus.json"),
        (f"labels{suffix}.json", f"labels{suffix}.json"),
        (f"samples{suffix}.jsonl", f"samples{suffix}.jsonl"),
    ]
    for remote_name, local_name in files:
        local = f"{dest}/{local_name}"
        rc = subprocess.run(
            [
                "modal", "volume", "get", "--force", "deception-data",
                f"{task}/{remote_name}", local,
            ]
        ).returncode
        if rc != 0:
            print(f"  skipped {task}/{remote_name} (not on volume yet)")
        else:
            print(f"  pulled {task}/{remote_name} -> ./{local}")


@app.local_entrypoint()
def main(
    task: str = "deception",
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
    label_workers: int = 4,
    labeler_models: str = "gpt-5.4-mini,gpt-5.4",
    per_model_attempts: int = 2,
    max_total_wait: float = 180.0,
    label_variant: str = "full",
    download: bool = True,
    download_dir: str = "results",
) -> None:
    """Run one or all deception pipeline stages for a given task.

    task in {"deception"}.
    stage in {"all", "model", "label", "samples", "probe", "download"}.
    label_variant in {"full" (v1, every deception token), "transition"
        (v2, only the prefix of each lie commitment)}. The corpus and
        activations are shared across variants — switching variants
        only re-runs the label / probe stages on the same volume.
    """
    valid_tasks = ("deception",)
    if task not in valid_tasks:
        raise SystemExit(f"unknown task: {task!r} (must be one of {valid_tasks})")

    valid_stages = ("all", "model", "label", "samples", "probe", "download")
    if stage not in valid_stages:
        raise SystemExit(f"unknown stage: {stage!r} (must be one of {valid_stages})")

    valid_variants = ("full", "transition")
    if label_variant not in valid_variants:
        raise SystemExit(
            f"unknown label_variant: {label_variant!r} "
            f"(must be one of {valid_variants})"
        )

    stages = ("model", "label", "probe") if stage == "all" else (stage,)

    if "model" in stages:
        print(f"=== [{task}] stage 1: Llama generation + all-layer activation extraction ===")
        do_model_stage.remote(
            task=task,
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
        print(
            f"=== [{task}] stage 2 (variant={label_variant!r}): "
            f"parallel OpenAI token labeling (+ samples.jsonl) ==="
        )
        do_label.remote(
            task=task,
            max_workers=label_workers,
            labeler_models=labeler_models,
            per_model_attempts=per_model_attempts,
            max_total_wait=max_total_wait,
            label_variant=label_variant,
        )
    if "samples" in stages:
        print(
            f"=== [{task}] (variant={label_variant!r}) "
            f"rebuilding samples.jsonl from existing corpus/labels ==="
        )
        do_samples.remote(task=task, label_variant=label_variant)
    if "probe" in stages:
        print(
            f"=== [{task}] stage 3 (variant={label_variant!r}): "
            f"GPU-vectorized probe sweep (full + deception-only subsets) ==="
        )
        do_probe.remote(
            task=task,
            max_offset=max_offset,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            run_name=run_name,
            label_variant=label_variant,
        )

    if stage == "download" or (download and stage in ("all", "probe", "label", "samples")):
        print(f"=== [{task}] (variant={label_variant!r}) downloading QC data ===")
        _pull_data(task, "data", label_variant=label_variant)
        if stage in ("download", "all", "probe"):
            print(f"=== [{task}] (variant={label_variant!r}) downloading probe results ===")
            _pull_results(task, download_dir, label_variant=label_variant)

    print("done.")
