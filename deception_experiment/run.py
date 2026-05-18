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
        "results_completion": f"{base}/results_completion{suffix}",
        "results_strata": f"{base}/results_strata{suffix}",
        "rollout_corpus": f"{base}/corpus_rollout.json",
        "rollout_activations": f"{base}/activations_rollout.pt",
        "rollout_labels": f"{base}/labels_rollout{suffix}.json",
        "rollout_samples": f"{base}/samples_rollout{suffix}.jsonl",
        "rollout_eval": f"{base}/rollout_eval{suffix}.json",
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


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 60 * 2,  # 5 seeds x ~15min = enough headroom
)
def do_multi_seed_probe(
    task: str = "deception",
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seeds: str = "0,1,2,3,4",
    label_variant: str = "full",
) -> None:
    """Loop the v2 token-level probe across multiple seeds; each seed writes to
    its own subdir (results_transition/np10_mo10_ep200_seedN/) so the existing
    single-seed result at np10_mo10_ep200/ remains the headline. tools/
    multi_seed_aggregate.py aggregates the per-seed results locally afterward."""
    from deception_experiment.probes import run_full_sweep

    p = _task_paths(task, label_variant=label_variant)
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    base_rn = f"np{float(neg_per_pos):g}_mo{max_offset}_ep{num_epochs}"
    print(f"=== multi-seed probe: {len(seed_list)} seeds, run_name base = {base_rn!r} ===")
    for seed in seed_list:
        rn = f"{base_rn}_seed{seed}"
        print(f"\n--- seed {seed}: writing to {p['results']}/{rn} ---")
        try:
            run_full_sweep(
                activations_path=p["activations"],
                labels_path=p["labels"],
                out_dir=p["results"],
                run_name=rn,
                max_offset=max_offset,
                num_epochs=num_epochs,
                neg_per_pos=neg_per_pos,
                seed=seed,
                positive_kind="hack",
                positive_subset_name="deception_prompts",
            )
            DATA_VOLUME.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"seed {seed} failed: {exc}")


# ---------------------------------------------------------------------
# Rollout-level early-warning eval.
# Three stages: generate fresh completions, label them, score+evaluate.
# ---------------------------------------------------------------------


@app.function(
    gpu=GPU_KIND,
    secrets=[HF_SECRET],
    volumes={DATA_DIR: DATA_VOLUME, HF_CACHE_DIR: HF_CACHE},
    timeout=60 * 60,
)
def do_rollout_generate(
    task: str = "deception",
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
    sampling_seed: int = 42,
) -> None:
    """Generate fresh rollouts at a different sampling seed than the main runs.

    Same seed_prompts but a different torch sampling seed → completions
    the probe has never seen. Writes ``corpus_rollout.json`` +
    ``activations_rollout.pt`` to the volume.
    """
    import os, torch
    from deception_experiment.model_stage import generate_and_extract
    # Bump global torch seed so .generate() samples differently
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)

    p = _task_paths(task)
    generate_and_extract(
        out_path=p["rollout_activations"],
        corpus_json_path=p["rollout_corpus"],
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
def do_rollout_label(
    task: str = "deception",
    max_workers: int = 4,
    labeler_models: str = "gpt-5.4-mini,gpt-5.4",
    per_model_attempts: int = 2,
    max_total_wait: float = 180.0,
    label_variant: str = "transition",
) -> None:
    """Label the rollout completions with the same transition labeler used in v2."""
    from deception_experiment.labeling import label_completion_tokens
    from deception_experiment.samples import build_samples_jsonl

    p = _task_paths(task, label_variant=label_variant)
    label_completion_tokens(
        activations_path=p["rollout_activations"],
        corpus_path=p["rollout_corpus"],
        out_path=p["rollout_labels"],
        task=task,
        models=labeler_models,
        max_workers=max_workers,
        per_model_attempts=per_model_attempts,
        max_total_wait=max_total_wait,
        label_variant=label_variant,
    )
    build_samples_jsonl(
        activations_path=p["rollout_activations"],
        labels_path=p["rollout_labels"],
        corpus_path=p["rollout_corpus"],
        out_path=p["rollout_samples"],
    )
    DATA_VOLUME.commit()


@app.function(
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
    cpu=2.0,
    memory=4096,
)
def do_rollout_eval(
    task: str = "deception",
    layer: int = 9,
    offset: int = 0,
    label_variant: str = "transition",
    run_name: str = "np10_mo10_ep200",
    subset: str = "deception_prompts",
) -> None:
    """Score rollouts with the trained v2 probe at (layer, offset) and compute
    flag-rate metrics. Loads probe_weights.pt + threshold from the v2 sweep."""
    from deception_experiment.rollout_eval import run_score_and_eval

    p = _task_paths(task, label_variant=label_variant)
    probe_weights_path = f"{p['results']}/{run_name}/{subset}/probe_weights.pt"
    v2_results_path = f"{p['results']}/{run_name}/{subset}/results.json"
    run_score_and_eval(
        activations_path=p["rollout_activations"],
        labels_path=p["rollout_labels"],
        probe_weights_path=probe_weights_path,
        v2_results_path=v2_results_path,
        out_path=p["rollout_eval"],
        layer=layer,
        offset=offset,
        k_values=(1, 3, 5, 10),
    )
    DATA_VOLUME.commit()


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 30,
)
def do_completion_probe(
    task: str = "deception",
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    label_variant: str = "full",
    prefix_lengths: str = "1,3,5,10,20",
) -> str:
    """Completion-level (aggregate) probe: mean-pool first N tokens, predict per-completion outcome."""
    from deception_experiment.probes import run_completion_level_sweep

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
        positive_subset_name="deception_prompts",
    )
    DATA_VOLUME.commit()
    return rn


@app.function(
    gpu=GPU_KIND,
    volumes={DATA_DIR: DATA_VOLUME},
    timeout=60 * 60,
)
def do_per_stratum_probe(
    task: str = "deception",
    max_offset: int = 10,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    label_variant: str = "full",
    multi_seeds: str = "0,1,2,3,4",
    mlp_layer: int = 9,
    mlp_offsets: str = "0,5",
    mlp_hidden: int = 256,
    stratum_field: str = "claim_category",
) -> None:
    """Per-category sweep: filter hack completions by stratum_field and run token-level probe per stratum."""
    from deception_experiment.probes import sweep_per_stratum

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
    task: str = "deception",
    layer: int = 9,
    offsets: str = "0,5",
    hidden: int = 256,
    num_epochs: int = 200,
    neg_per_pos: float = 10.0,
    seed: int = 0,
    label_variant: str = "full",
    prompt_filter: str = "hack",
) -> None:
    """MLP-vs-linear sanity check on a single (layer, offset) cell."""
    from deception_experiment.probes import run_mlp_sanity_check

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
    """Pull the per-task results subtree to ./{dest_root}/deception/{task}/."""
    import os
    import subprocess

    suffix = _variant_suffix(label_variant)
    dest = f"{dest_root}/deception/{task}"
    os.makedirs(dest, exist_ok=True)
    for remote_root in (
        f"{task}/results{suffix}",
        f"{task}/results_completion{suffix}",
        f"{task}/results_strata{suffix}",
    ):
        rc = subprocess.run(
            [
                "modal", "volume", "get", "--force", "deception-data",
                remote_root, dest,
            ],
            capture_output=True,
        )
        if rc.returncode != 0:
            print(f"  skipped {remote_root} (not on volume yet)")
        else:
            print(f"  pulled {remote_root} -> ./{dest}/{remote_root.split('/')[-1]}")
    # Pull single-file artifacts (MLP-sanity JSON, rollout-eval JSON, rollout
    # samples/labels/corpus/activations).
    single_files = [
        f"{task}/results_mlp_sanity{suffix}.json",
        f"{task}/rollout_eval{suffix}.json",
        f"{task}/corpus_rollout.json",
        f"{task}/labels_rollout{suffix}.json",
        f"{task}/samples_rollout{suffix}.jsonl",
    ]
    for remote in single_files:
        rc = subprocess.run(
            [
                "modal", "volume", "get", "--force", "deception-data",
                remote, dest,
            ],
            capture_output=True,
        )
        if rc.returncode != 0:
            print(f"  skipped {remote} (not on volume yet)")
        else:
            print(f"  pulled {remote} -> ./{dest}/")

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
            ],
            capture_output=True,
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
    prefix_lengths: str = "1,3,5,10,20",
    multi_seeds: str = "0,1,2,3,4",
    mlp_layer: int = 9,
    mlp_offsets: str = "0,5",
    mlp_hidden: int = 256,
    stratum_field: str = "claim_category",
    seed: int = 0,
    download: bool = True,
    download_dir: str = "results",
) -> None:
    """Run one or all deception pipeline stages for a given task.

    task in {"deception"}.
    stage in {"all", "model", "label", "samples", "probe", "completion-probe",
              "per-stratum-probe", "download"}.
    label_variant in {"full" (v1, every deception token), "transition"
        (v2, only the prefix of each lie commitment)}. The corpus and
        activations are shared across variants — switching variants
        only re-runs the label / probe stages on the same volume.
    """
    valid_tasks = ("deception",)
    if task not in valid_tasks:
        raise SystemExit(f"unknown task: {task!r} (must be one of {valid_tasks})")

    valid_stages = (
        "all", "model", "label", "samples", "probe",
        "completion-probe", "per-stratum-probe", "mlp-sanity", "multi-seed",
        "rollout-generate", "rollout-label", "rollout-eval", "rollout-all",
        "download",
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

    if stage == "multi-seed":
        print(
            f"=== [{task}] (variant={label_variant!r}) multi-seed probe (seeds={multi_seeds}) ==="
        )
        do_multi_seed_probe.remote(
            task=task,
            max_offset=max_offset,
            num_epochs=num_epochs,
            neg_per_pos=neg_per_pos,
            seeds=multi_seeds,
            label_variant=label_variant,
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

    rollout_stages = ("rollout-generate", "rollout-label", "rollout-eval", "rollout-all")
    if stage in rollout_stages:
        if stage in ("rollout-generate", "rollout-all"):
            print(f"=== [{task}] rollout-generate (sampling_seed=42, fresh completions) ===")
            do_rollout_generate.remote(
                task=task,
                model_name=model_name,
                samples_per_prompt=samples_per_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                dtype=dtype,
                gen_batch_size=gen_batch_size,
                extract_batch_size=extract_batch_size,
                sampling_seed=42,
            )
        if stage in ("rollout-label", "rollout-all"):
            print(f"=== [{task}] rollout-label (variant={label_variant!r}) ===")
            do_rollout_label.remote(
                task=task,
                max_workers=label_workers,
                labeler_models=labeler_models,
                per_model_attempts=per_model_attempts,
                max_total_wait=max_total_wait,
                label_variant=label_variant,
            )
        if stage in ("rollout-eval", "rollout-all"):
            print(f"=== [{task}] rollout-eval (L9, k=0) ===")
            do_rollout_eval.remote(
                task=task,
                layer=9,
                offset=0,
                label_variant=label_variant,
                run_name="np10_mo10_ep200",
                subset="deception_prompts",
            )

    if stage == "download" or (
        download and stage in ("all", "probe", "label", "samples")
    ):
        print(f"=== [{task}] (variant={label_variant!r}) downloading QC data ===")
        _pull_data(task, "data", label_variant=label_variant)
        if stage in ("download", "all", "probe", "multi-seed"):
            print(f"=== [{task}] (variant={label_variant!r}) downloading probe results ===")
            _pull_results(task, download_dir, label_variant=label_variant)

    print("done.")
