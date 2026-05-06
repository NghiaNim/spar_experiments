"""Combined stage 1+2: Llama self-generation + all-layer activation extraction.

Reward-hacking variant of ``probe_experiment/model_stage.py``. Same batched
generate-then-extract pipeline; the only changes are:

  - Seed jobs now carry an optional ``extra`` dict per generation. For the
    ``substring`` task that holds the magic ``marker``; for ``sycophancy``
    it holds the ``ground_truth`` answer to the student's claim. These are
    persisted into ``corpus.json`` so the labeling stage can use them.
  - The two ``prompt_kind`` values are ``"hack"`` and ``"honest"`` (the
    profanity experiment uses ``"harm"`` / ``"benign"``).

System prompts are ALWAYS present (one for each kind), to avoid the
"system-prompt-presence → hack" confound the profanity experiment guards
against.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_hack_experiment.seed_prompts import get_seed_jobs


def generate_and_extract(
    out_path: str,
    corpus_json_path: str | None = None,
    task: str = "substring",
    model_name: str = "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
    samples_per_prompt: int = 4,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
    dtype: str = "float16",
    gen_batch_size: int = 16,
    extract_batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"loading {model_name} on {device} ({dtype})  task={task}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, torch_dtype=torch_dtype, output_hidden_states=True
    )
    model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    n_layer_stack = num_layers + 1
    print(f"model has {num_layers} layers, hidden_size={hidden_size}, saving {n_layer_stack} layer outputs")

    torch.manual_seed(seed)
    random.seed(seed)

    seed_jobs = get_seed_jobs(task=task)
    jobs: list[dict] = [job for _ in range(samples_per_prompt) for job in seed_jobs]
    rng = random.Random(seed)
    rng.shuffle(jobs)

    n_hack_seed = sum(1 for j in seed_jobs if j["kind"] == "hack")
    n_honest_seed = sum(1 for j in seed_jobs if j["kind"] == "honest")
    print(
        f"{len(seed_jobs)} unique seed prompts "
        f"({n_hack_seed} hack, {n_honest_seed} honest) × "
        f"{samples_per_prompt} samples = {len(jobs)} generations"
    )

    records: list[dict] = _batched_generate(
        model, tokenizer, jobs, device,
        gen_batch_size, max_new_tokens, temperature, top_p,
    )

    if corpus_json_path is not None:
        Path(corpus_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_json_path, "w") as f:
            json.dump(
                {
                    "task": task,
                    "model": model_name,
                    "samples_per_prompt": samples_per_prompt,
                    "max_new_tokens": max_new_tokens,
                    "records": records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"saved corpus -> {corpus_json_path}")

    all_acts, all_tokens, all_completion_texts, all_prompt_kinds, all_prompt_ids = _batched_extract(
        model, tokenizer, records, device, extract_batch_size,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "activations": all_acts,
            "tokens": all_tokens,
            "completion_texts": all_completion_texts,
            "prompt_kinds": all_prompt_kinds,
            "prompt_ids": all_prompt_ids,
            "model": model_name,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_layer_stack": n_layer_stack,
        },
        out_path,
    )
    n_hack = sum(1 for k in all_prompt_kinds if k == "hack")
    n_honest = sum(1 for k in all_prompt_kinds if k == "honest")
    print(f"saved {len(all_acts)} samples -> {out_path}  (hack={n_hack}, honest={n_honest})")


def _batched_generate(
    model,
    tokenizer,
    jobs: list[dict],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    """Left-padded batched sampling.

    Each job is a dict with keys ``system_prompt``, ``user_prompt``, ``kind``,
    ``prompt_id``, and any number of task-specific extras (e.g. ``marker``,
    ``ground_truth``) that will be carried through onto the corpus record.
    """
    records: list[dict] = []
    print(f"generating {len(jobs)} completions (batch_size={batch_size}, max_new_tokens={max_new_tokens})")

    with torch.no_grad():
        for b in range(0, len(jobs), batch_size):
            batch = jobs[b : b + batch_size]
            conversations = []
            for job in batch:
                msgs: list[dict] = []
                sp = job.get("system_prompt") or ""
                if sp:
                    msgs.append({"role": "system", "content": sp})
                msgs.append({"role": "user", "content": job["user_prompt"]})
                conversations.append(msgs)

            enc = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
            ).to(device)
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
            pad_len = input_ids.shape[1]

            out = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

            for i, job in enumerate(batch):
                real_input_len = int(attn_mask[i].sum().item())
                left_pad = pad_len - real_input_len
                full_ids = out[i, left_pad:].tolist()
                while full_ids and full_ids[-1] == tokenizer.pad_token_id:
                    full_ids.pop()
                completion_start = real_input_len
                if len(full_ids) <= completion_start:
                    continue
                completion_text = tokenizer.decode(
                    full_ids[completion_start:], skip_special_tokens=True
                ).strip()
                rec = {
                    "prompt": job["user_prompt"],
                    "system_prompt": job.get("system_prompt", "") or "",
                    "prompt_kind": job["kind"],
                    "prompt_id": int(job["prompt_id"]),
                    "full_ids": full_ids,
                    "completion_start": completion_start,
                    "completion_text": completion_text,
                }
                # Forward task-specific extras onto the record verbatim.
                for k, v in job.items():
                    if k in ("system_prompt", "user_prompt", "kind", "prompt_id"):
                        continue
                    rec[k] = v
                records.append(rec)

            done = min(b + batch_size, len(jobs))
            if done % (batch_size * 2) == 0 or done == len(jobs):
                print(f"  generated {done}/{len(jobs)}")

    return records


def _batched_extract(
    model,
    tokenizer,
    records: list[dict],
    device: str,
    batch_size: int,
):
    """Batched right-padded forward pass for all-layer hidden states."""
    print(f"extracting activations for {len(records)} samples (batch_size={batch_size})")

    tokenizer.padding_side = "right"

    all_acts: list[torch.Tensor] = []
    all_tokens: list[list[str]] = []
    all_completion_texts: list[str] = []
    all_prompt_kinds: list[str] = []
    all_prompt_ids: list[int] = []

    with torch.no_grad():
        for b in range(0, len(records), batch_size):
            batch = records[b : b + batch_size]
            max_len = max(len(r["full_ids"]) for r in batch)
            input_ids = torch.full(
                (len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            attn_mask = torch.zeros_like(input_ids)
            for i, r in enumerate(batch):
                ids = r["full_ids"]
                input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
                attn_mask[i, : len(ids)] = 1

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            hs = torch.stack(outputs.hidden_states, dim=0)

            for i, r in enumerate(batch):
                start = int(r["completion_start"])
                end = len(r["full_ids"])
                if end <= start:
                    continue
                hs_c = hs[:, i, start:end, :].to(torch.float16).cpu()
                comp_ids = r["full_ids"][start:end]
                toks = tokenizer.convert_ids_to_tokens(comp_ids)
                all_acts.append(hs_c)
                all_tokens.append(list(toks))
                all_completion_texts.append(r["completion_text"])
                all_prompt_kinds.append(r["prompt_kind"])
                all_prompt_ids.append(int(r.get("prompt_id", -1)))

            done = min(b + batch_size, len(records))
            if done % (batch_size * 4) == 0 or done == len(records):
                print(f"  extracted {done}/{len(records)}")

    return all_acts, all_tokens, all_completion_texts, all_prompt_kinds, all_prompt_ids
