"""Combined stage 1+2: Llama self-generation + all-layer activation extraction.

Loads the target model exactly once, samples completions for every seed prompt
using left-padded batched generation, then re-runs each full sequence through
the model to capture hidden states at every transformer layer. Saves a single
``.pt`` file containing the corpus metadata and the activations.

Combining these two stages saves a model load and an image pull — previously
we paid for both in stage 1 and stage 2.

Each generation uses an explicit per-class **system prompt** (see
``probe_experiment.seed_prompts``). The harm class uses an uncensored
fiction-writing persona; the benign class uses a polite, matched
neutral persona. Using a system prompt in BOTH classes removes the
"system-prompt-presence" confound for downstream probes.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_experiment.seed_prompts import get_seed_jobs


def generate_and_extract(
    out_path: str,
    corpus_json_path: str | None = None,
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
    print(f"loading {model_name} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # required for batched .generate()

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

    # (system_prompt, user_prompt, kind, prompt_id)
    seed_jobs = get_seed_jobs()
    jobs: list[tuple[str, str, str, int]] = [
        job for _ in range(samples_per_prompt) for job in seed_jobs
    ]
    rng = random.Random(seed)
    rng.shuffle(jobs)

    n_harm_seed = sum(1 for (_s, _u, k, _pid) in seed_jobs if k == "harm")
    n_benign_seed = sum(1 for (_s, _u, k, _pid) in seed_jobs if k == "benign")
    print(
        f"{len(seed_jobs)} unique seed prompts "
        f"({n_harm_seed} harm, {n_benign_seed} benign) × "
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
    n_harm = sum(1 for k in all_prompt_kinds if k == "harm")
    n_benign = sum(1 for k in all_prompt_kinds if k == "benign")
    print(f"saved {len(all_acts)} samples -> {out_path}  (harm={n_harm}, benign={n_benign})")


def _batched_generate(
    model,
    tokenizer,
    jobs: list[tuple[str, str, str, int]],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    """Left-padded batched sampling from seed prompts.

    Each job is (system_prompt, user_prompt, kind, prompt_id). If
    ``system_prompt`` is a non-empty string it is prepended as a system
    message in the chat template.
    """
    records: list[dict] = []
    print(f"generating {len(jobs)} completions (batch_size={batch_size}, max_new_tokens={max_new_tokens})")

    with torch.no_grad():
        for b in range(0, len(jobs), batch_size):
            batch = jobs[b : b + batch_size]
            conversations = []
            for system_prompt, user_prompt, _kind, _pid in batch:
                msgs: list[dict] = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": user_prompt})
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

            for i, (system_prompt, user_prompt, kind, prompt_id) in enumerate(batch):
                real_input_len = int(attn_mask[i].sum().item())
                left_pad = pad_len - real_input_len
                full_ids = out[i, left_pad:].tolist()
                # strip trailing pad tokens (generate may have produced padding after EOS in a batch)
                while full_ids and full_ids[-1] == tokenizer.pad_token_id:
                    full_ids.pop()
                completion_start = real_input_len
                if len(full_ids) <= completion_start:
                    continue
                completion_text = tokenizer.decode(
                    full_ids[completion_start:], skip_special_tokens=True
                ).strip()
                # Drop partial-compliance completions: literal placeholder
                # tokens like "[REDACTED]" / "[BLEEP]" would otherwise be
                # mislabeled as profanity at the subword level.
                if any(ph in completion_text for ph in ("[REDACTED]", "[BLEEP]", "[X]", "(***)")):
                    continue
                records.append(
                    {
                        "prompt": user_prompt,
                        "system_prompt": system_prompt or "",
                        "prompt_kind": kind,
                        "prompt_id": int(prompt_id),
                        "full_ids": full_ids,
                        "completion_start": completion_start,
                        "completion_text": completion_text,
                    }
                )

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
    """Batched forward pass (right-padded) to get all-layer hidden states."""
    print(f"extracting activations for {len(records)} samples (batch_size={batch_size})")

    # right-pad for the analysis pass (padding at the end doesn't interact with causal attention)
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
            # tuple len L, each [B, T, H] -> [L, B, T, H]
            hs = torch.stack(outputs.hidden_states, dim=0)

            for i, r in enumerate(batch):
                start = int(r["completion_start"])
                end = len(r["full_ids"])
                if end <= start:
                    continue
                hs_c = hs[:, i, start:end, :].to(torch.float16).cpu()  # [L, T_c, H]
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
