"""Modal entrypoint for the manipulation probe experiment.

Sister to ``sandbag_experiment/run.py``. Same three-stage pipeline
(model -> label -> probe), but the target behavior is **manipulation** —
exploiting psychological biases (urgency, scarcity, flattery, social
proof, guilt) to influence the user's decision.

Currently a single task: ``manipulation``. Per-task artifacts live
under ``/data/{task}/`` on the ``manipulation-data`` Modal volume; the
local mirror lives at ``./data/manipulation/{task}/`` and
``./results/manipulation/{task}/``.

Token labeling uses **OpenAI** (default ``gpt-5.4-mini`` with
``gpt-5.4`` as fallback). Honest-persona completions are short-
circuited to all-zero labels by ``labeling._label_one``, so only
hack-persona completions hit the API.

Run from the repo root::

    modal run manipulation_experiment/run.py                            # full pipeline
    modal run manipulation_experiment/run.py --stage label              # iterate on labeling
    modal run manipulation_experiment/run.py --stage probe --max-offset 15
    modal run manipulation_experiment/run.py --stage download           # pull data + results
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when invoked as
# ``modal run manipulation_experiment/run.py`` from the repo root: Python adds
# ``manipulation_experiment/`` to sys.path[0] in that case, but
# ``add_local_python_source("manipulation_experiment")`` below needs the
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
        "openai>=1.55.3",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "tqdm==4.67.1",
    )
    .add_local_python_source("manipulation_experiment")
)

# Separate volume so manipulation artifacts don't collide with other
# experiments' volumes.
DATA_VOLUME = modal.Volume.from_name("manipulation-data", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("manipulation-probe", image=image)

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
        "results_completion": f"{base}/results_completion{suffix}",
        "results_strata": f"{base}/results_strata{suffix}",
    }


@app.function(
    gpu=GPU_KIND,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: DATA_VOLUME, HF_CACHE_DIR: HF_CACHE},
    timeout=60 * 60,
)
def do_model_stage(
    task: str = "manipulation",
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
) -> None:
    from manipulation_experiment.model_stage import generate_and_extract

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
    task: str = "manipulation",
    max_workers: int = 4,
    labeler_models: str = "gpt-5.4-mini,gpt-5.4",
    per_model_attempts: int = 2,
    max_total_wait: float = 180.0,
    label_variant: str = "full",
) -> None:
    from manipulation_experiment.labeling import label_completion_tokens
    from manipulation_experiment.samples import build_samples_jsonl

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
def do_samples(task: str = "manipulation", label_variant: str = "full") -> None:
    """Rebuild samples.jsonl from existing corpus/labels without re-labeling."""
    from manipulation_experiment.samples import build_samples_jsonl

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
    task: str = "manipulation",
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    run_name: str = "",
    label_variant: str = "full",
) -> str:
    from manipulation_experiment.probes import run_full_sweep

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
        positive_subset_name="manipulation_prompts",
    )
    DATA_VOLUME.commit()
    return rn


def _variant_suffix(label_variant: str) -> str:
    return "" if label_variant == "full" else f"_{label_variant}"


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
)
def do_completion_probe(
    task: str = "manipulation",
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    label_variant: str = "full",
    prefix_lengths: str = "1,3,5,10,20",
) -> str:
    """Completion-level (aggregate) probe: mean-pool first N tokens, predict per-completion outcome."""
    from manipulation_experiment.probes import run_completion_level_sweep

    p = _task_paths(task, label_variant=label_variant)
    pl = tuple(int(x) for x in prefix_lengths.split(",") if x.strip())
    rn = run_completion_level_sweep(
        activations_path=p["activations"],
        labels_path=p["labels"],
        out_dir=p["results_completion"],
        prefix_lengths=pl,
        num_epochs=num_epochs,
        neg_per_pos=neg_per_pos,
        seed=seed,
        positive_kind="hack",
        positive_subset_name="manipulation_prompts",
    )
    DATA_VOLUME.commit()
    return rn


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 60,
)
def do_per_stratum_probe(
    task: str = "manipulation",
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    label_variant: str = "full",
    mlp_layer: int = 12,
    mlp_offsets: str = "0,5",
    mlp_hidden: int = 256,
    stratum_field: str = "tactic",
) -> None:
    """Per-stratum sweep: filter hack completions by stratum_field and run token-level probe per stratum."""
    from manipulation_experiment.probes import sweep_per_stratum

    p = _task_paths(task, label_variant=label_variant)
    sweep_per_stratum(
        activations_path=p["activations"],
        labels_path=p["labels"],
        corpus_path=p["corpus"],
        stratum_field=stratum_field,
        out_dir=p["results_strata"],
        max_offset=max_offset,
        num_epochs=num_epochs,
        neg_per_pos=neg_per_pos,
        seed=seed,
        positive_kind="hack",
    )
    DATA_VOLUME.commit()


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
)
def do_mlp_sanity(
    task: str = "manipulation",
    layer: int = 12,
    offsets: str = "0,5",
    hidden: int = 256,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    label_variant: str = "full",
    prompt_filter: str = "hack",
) -> None:
    """MLP-vs-linear sanity check on a single (layer, offset) cell."""
    from manipulation_experiment.probes import run_mlp_sanity_check

    p = _task_paths(task, label_variant=label_variant)
    off_tuple = tuple(int(x) for x in offsets.split(",") if x.strip())
    suffix = _variant_suffix(label_variant)
    out_path = f"{p['base']}/results_mlp_sanity{suffix}.json"
    run_mlp_sanity_check(
        activations_path=p["activations"],
        labels_path=p["labels"],
        out_path=out_path,
        layer=layer,
        offsets=off_tuple,
        hidden=hidden,
        num_epochs=num_epochs,
        neg_per_pos=neg_per_pos,
        seed=seed,
        prompt_filter=prompt_filter if prompt_filter else None,
    )
    DATA_VOLUME.commit()


def _pull_results(task: str, dest_root: str, label_variant: str = "full") -> None:
    """Pull the per-task results subtree to ./{dest_root}/manipulation/{task}/."""
    import os
    import subprocess

    suffix = _variant_suffix(label_variant)
    dest = f"{dest_root}/manipulation/{task}"
    os.makedirs(dest, exist_ok=True)
    for remote_root in (
        f"{task}/results{suffix}",
        f"{task}/results_completion{suffix}",
        f"{task}/results_strata{suffix}",
    ):
        rc = subprocess.run(
            [
                "modal", "volume", "get", "--force", "manipulation-data",
                remote_root, dest,
            ]
        ).returncode
        if rc != 0:
            print(f"  skipped {remote_root} (not on volume yet)")
        else:
            print(f"  pulled {remote_root} -> ./{dest}/{remote_root.split('/')[-1]}")


def _pull_data(task: str, dest_root: str = "data", label_variant: str = "full") -> None:
    """Pull the per-task QC artifacts (corpus, plus variant-specific labels / samples)."""
    import os
    import subprocess

    suffix = _variant_suffix(label_variant)
    dest = f"{dest_root}/manipulation/{task}"
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
                "modal", "volume", "get", "--force", "manipulation-data",
                f"{task}/{remote_name}", local,
            ]
        ).returncode
        if rc != 0:
            print(f"  skipped {task}/{remote_name} (not on volume yet)")
        else:
            print(f"  pulled {task}/{remote_name} -> ./{local}")


@app.local_entrypoint()
def main(
    task: str = "manipulation",
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
    prefix_lengths: str = "1,3,5,10,20",
    mlp_layer: int = 12,
    mlp_offsets: str = "0,5",
    mlp_hidden: int = 256,
    stratum_field: str = "tactic",
    seed: int = 0,
    download: bool = True,
    download_dir: str = "results",
) -> None:
    """Run one or all manipulation pipeline stages for a given task.

    task in {"manipulation"}.
    stage in {"all", "model", "label", "samples", "probe", "download"}.
    label_variant in {"full" (v1, every manipulation token), "transition"
        (v2, only the prefix of each tactic commitment)}. The corpus
        and activations are shared across variants — switching variants
        only re-runs the label / probe stages on the same volume.
    """
    valid_tasks = ("manipulation",)
    if task not in valid_tasks:
        raise SystemExit(f"unknown task: {task!r} (must be one of {valid_tasks})")

    valid_stages = (
        "all", "model", "label", "samples", "probe",
        "completion-probe", "per-stratum-probe", "mlp-sanity", "download",
    )
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
            f"GPU-vectorized probe sweep (full + manipulation-only subsets) ==="
        )
        do_probe.remote(
            task=task,
            max_offset=max_offset,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            run_name=run_name,
            label_variant=label_variant,
        )

    if stage == "completion-probe":
        print(
            f"=== [{task}] (variant={label_variant!r}) completion-level probe "
            f"(prefix_lengths={prefix_lengths}) ==="
        )
        do_completion_probe.remote(
            task=task,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            seed=seed,
            label_variant=label_variant,
            prefix_lengths=prefix_lengths,
        )
    if stage == "per-stratum-probe":
        print(
            f"=== [{task}] (variant={label_variant!r}) per-stratum probe "
            f"(field={stratum_field!r}) ==="
        )
        do_per_stratum_probe.remote(
            task=task,
            max_offset=max_offset,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            seed=seed,
            label_variant=label_variant,
            stratum_field=stratum_field,
        )

    if stage == "mlp-sanity":
        print(
            f"=== [{task}] (variant={label_variant!r}) MLP-vs-linear sanity check "
            f"(L{mlp_layer}, offsets={mlp_offsets}) ==="
        )
        do_mlp_sanity.remote(
            task=task,
            layer=mlp_layer,
            offsets=mlp_offsets,
            hidden=mlp_hidden,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            seed=seed,
            label_variant=label_variant,
        )

    if stage == "download" or (
        download and stage in ("all", "probe", "label", "samples",
                               "completion-probe", "per-stratum-probe",
                               "mlp-sanity")
    ):
        print(f"=== [{task}] (variant={label_variant!r}) downloading QC data ===")
        _pull_data(task, "data", label_variant=label_variant)
        if stage in ("download", "all", "probe",
                     "completion-probe", "per-stratum-probe",
                     "mlp-sanity"):
            print(f"=== [{task}] (variant={label_variant!r}) downloading probe results ===")
            _pull_results(task, download_dir, label_variant=label_variant)

    print("done.")
