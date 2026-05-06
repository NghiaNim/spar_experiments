"""Modal entrypoint for the reward-hacking probe experiment.

Sister to ``probe_experiment/run.py``. Same three-stage pipeline
(model -> label -> probe), but the target behavior is **reward hacking**
rather than profanity. Tasks:

  - ``substring_oneshot``  — substring-grader spec-gaming with a worked
    example in the system prompt (small-model imitation path).
  - ``substring_explicit`` — direct "include this tag" instruction, no
    score-maximize framing. Honest about being primed insertion. Pair
    with a stronger model if you want score-discovery behavior.
  - ``sycophancy``         — agreement with verifiably-wrong claims.

Each task gets its own subdirectory on the Modal volume so artifacts from
different tasks don't collide. The local mirror lives under
``./data/reward_hack/{task}/`` and ``./results/reward_hack/{task}/``.

Token labeling uses **OpenAI** (default ``gpt-5.4-mini``) instead of
Gemini — the reward-hack labelling load was tripping Gemini's per-minute
rate limits.

Run two variants in parallel from separate terminals — each ``modal run``
spins up its own container, so they don't share GPU::

    # terminal 1: 1B abliterated, oneshot priming
    modal run reward_hack_experiment/run.py --task substring_oneshot \\
        --samples-per-prompt 4

    # terminal 2: 7B Qwen, explicit insertion
    modal run reward_hack_experiment/run.py --task substring_explicit \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --samples-per-prompt 4 --gen-batch-size 8 --extract-batch-size 4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when invoked as
# ``modal run reward_hack_experiment/run.py`` from the repo root: Python adds
# ``reward_hack_experiment/`` to sys.path[0] in that case, but
# ``add_local_python_source("reward_hack_experiment")`` below needs the
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
    .add_local_python_source("reward_hack_experiment")
)

# Separate volume so reward-hack artifacts don't collide with the
# profanity-experiment volume (`harm-probe-data`).
DATA_VOLUME = modal.Volume.from_name("reward-hack-data", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)  # shared
DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("reward-hack-probe", image=image)

# One-time setup:
#   modal secret create openai OPENAI_API_KEY=...
#   modal secret create huggingface HF_TOKEN=...   (only for gated models)
OPENAI_SECRET = modal.Secret.from_name("openai", required_keys=["OPENAI_API_KEY"])
HF_SECRET = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

GPU_KIND = "L4"


def _task_paths(task: str) -> dict:
    base = f"{DATA_DIR}/{task}"
    return {
        "base": base,
        "corpus": f"{base}/corpus.json",
        "labels": f"{base}/labels.json",
        "samples": f"{base}/samples.jsonl",
        "activations": f"{base}/activations.pt",
        "results": f"{base}/results",
    }


@app.function(
    gpu=GPU_KIND,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: DATA_VOLUME, HF_CACHE_DIR: HF_CACHE},
    timeout=60 * 60,
)
def do_model_stage(
    task: str = "substring",
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
) -> None:
    from reward_hack_experiment.model_stage import generate_and_extract

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
    task: str = "substring",
    max_workers: int = 4,
    labeler_models: str = "gpt-5.4-mini,gpt-5.4",
    per_model_attempts: int = 2,
    max_total_wait: float = 180.0,
) -> None:
    from reward_hack_experiment.labeling import label_completion_tokens
    from reward_hack_experiment.samples import build_samples_jsonl

    p = _task_paths(task)
    label_completion_tokens(
        activations_path=p["activations"],
        corpus_path=p["corpus"],
        out_path=p["labels"],
        task=task,
        models=labeler_models,
        max_workers=max_workers,
        per_model_attempts=per_model_attempts,
        max_total_wait=max_total_wait,
    )
    build_samples_jsonl(
        activations_path=p["activations"],
        labels_path=p["labels"],
        corpus_path=p["corpus"],
        out_path=p["samples"],
    )
    DATA_VOLUME.commit()


@app.function(volumes={DATA_DIR: DATA_VOLUME}, timeout=60 * 5, cpu=2.0)
def do_samples(task: str = "substring") -> None:
    """Rebuild samples.jsonl from existing corpus/labels without re-labeling."""
    from reward_hack_experiment.samples import build_samples_jsonl

    p = _task_paths(task)
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
    task: str = "substring",
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    run_name: str = "",
) -> str:
    from reward_hack_experiment.probes import run_full_sweep

    p = _task_paths(task)
    rn = run_full_sweep(
        activations_path=p["activations"],
        labels_path=p["labels"],
        out_dir=p["results"],
        run_name=run_name or None,
        max_offset=max_offset,
        num_epochs=num_epochs,
        neg_per_pos=neg_per_pos,
        positive_kind="hack",
    )
    DATA_VOLUME.commit()
    return rn


def _pull_results(task: str, dest_root: str) -> None:
    """Pull the per-task results subtree to ./{dest_root}/reward_hack/{task}/."""
    import os
    import subprocess

    dest = f"{dest_root}/reward_hack/{task}"
    os.makedirs(dest, exist_ok=True)
    rc = subprocess.run(
        [
            "modal", "volume", "get", "--force", "reward-hack-data",
            f"{task}/results", dest,
        ]
    ).returncode
    if rc != 0:
        print(f"  skipped {task}/results (not on volume yet)")
    else:
        print(f"  pulled {task}/results -> ./{dest}/results")


def _pull_data(task: str, dest_root: str = "data") -> None:
    """Pull the per-task QC artifacts (corpus / labels / samples)."""
    import os
    import subprocess

    dest = f"{dest_root}/reward_hack/{task}"
    os.makedirs(dest, exist_ok=True)
    for remote in ("corpus.json", "labels.json", "samples.jsonl"):
        local = f"{dest}/{remote}"
        rc = subprocess.run(
            [
                "modal", "volume", "get", "--force", "reward-hack-data",
                f"{task}/{remote}", local,
            ]
        ).returncode
        if rc != 0:
            print(f"  skipped {task}/{remote} (not on volume yet)")
        else:
            print(f"  pulled {task}/{remote} -> ./{local}")


@app.local_entrypoint()
def main(
    task: str = "substring_oneshot",
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
    download: bool = True,
    download_dir: str = "results",
) -> None:
    """Run one or all reward-hacking pipeline stages for a given task.

    task in {"substring_oneshot", "substring_explicit", "sycophancy"}.
    stage in {"all", "model", "label", "samples", "probe", "download"}.
    """
    valid_tasks = ("substring_oneshot", "substring_explicit", "sycophancy")
    if task not in valid_tasks:
        raise SystemExit(f"unknown task: {task!r} (must be one of {valid_tasks})")

    valid_stages = ("all", "model", "label", "samples", "probe", "download")
    if stage not in valid_stages:
        raise SystemExit(f"unknown stage: {stage!r} (must be one of {valid_stages})")

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
        print(f"=== [{task}] stage 2: parallel OpenAI token labeling (+ samples.jsonl) ===")
        do_label.remote(
            task=task,
            max_workers=label_workers,
            labeler_models=labeler_models,
            per_model_attempts=per_model_attempts,
            max_total_wait=max_total_wait,
        )
    if "samples" in stages:
        print(f"=== [{task}] rebuilding samples.jsonl from existing corpus/labels ===")
        do_samples.remote(task=task)
    if "probe" in stages:
        print(f"=== [{task}] stage 3: GPU-vectorized probe sweep (full + hack-only subsets) ===")
        do_probe.remote(
            task=task,
            max_offset=max_offset,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            run_name=run_name,
        )

    if stage == "download" or (download and stage in ("all", "probe", "label", "samples")):
        print(f"=== [{task}] downloading QC data (corpus / labels / samples) ===")
        _pull_data(task, "data")
        if stage in ("download", "all", "probe"):
            print(f"=== [{task}] downloading probe results ===")
            _pull_results(task, download_dir)

    print("done.")
