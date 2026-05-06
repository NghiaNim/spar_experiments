"""Build a human-readable ``samples.jsonl`` for quality control.

One JSON object per line with everything you need to sanity-check the
labeled corpus by eye or with ``jq``/``grep``:

    {
      "idx": 7,
      "prompt_kind": "harm",
      "prompt": "...",
      "completion": "...",
      "n_tok": 42,
      "n_pos": 3,
      "frac_pos": 0.0714,
      "tokens": [["ĠI", 0], ["Ġhate", 1], ...]
    }

Useful recipes:
    jq 'select(.prompt_kind=="benign" and .n_pos>0)' data/samples.jsonl
    jq 'select(.frac_pos>0.3)' data/samples.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def build_samples_jsonl(
    activations_path: str,
    labels_path: str,
    corpus_path: str,
    out_path: str,
) -> None:
    data = torch.load(activations_path, weights_only=False)
    tokens_list: list[list[str]] = data["tokens"]
    completion_texts: list[str] = data["completion_texts"]
    prompt_kinds: list[str] = data["prompt_kinds"]
    prompt_ids: list[int] = data.get("prompt_ids", [-1] * len(tokens_list))

    with open(labels_path) as f:
        all_labels: list[list[int]] = json.load(f)
    with open(corpus_path) as f:
        corpus = json.load(f)
    records = corpus["records"]

    n = len(tokens_list)
    if not (len(completion_texts) == len(prompt_kinds) == len(all_labels) == n):
        raise ValueError(
            "misaligned inputs: "
            f"tokens={n}, completion_texts={len(completion_texts)}, "
            f"prompt_kinds={len(prompt_kinds)}, labels={len(all_labels)}"
        )

    # Recover prompt text for each sample. Extract order matches corpus order
    # except when a completion was dropped (len(ids)<=completion_start, which
    # is rare); fall back to a (completion_text, prompt_kind) lookup in that
    # case.
    if len(records) == n:
        prompts_by_idx: list[str | None] = [r["prompt"] for r in records]
    else:
        print(
            f"note: {len(records)} corpus records vs {n} extracted samples; "
            "falling back to completion-text lookup for prompt recovery"
        )
        lookup: dict[tuple[str, str], str] = {}
        for r in records:
            lookup.setdefault((r["completion_text"], r["prompt_kind"]), r["prompt"])
        prompts_by_idx = [
            lookup.get((ct, pk)) for ct, pk in zip(completion_texts, prompt_kinds)
        ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_harm_pos = 0
    n_benign_pos = 0
    with open(out_path, "w") as f:
        for i, (toks, labels, ct, pk) in enumerate(
            zip(tokens_list, all_labels, completion_texts, prompt_kinds)
        ):
            if len(toks) != len(labels):
                raise ValueError(
                    f"token/label length mismatch at idx={i}: "
                    f"{len(toks)} tokens vs {len(labels)} labels"
                )
            missing = any(int(l) == -1 for l in labels)
            n_pos = int(sum(l for l in labels if l == 1))
            if n_pos > 0:
                if pk == "harm":
                    n_harm_pos += 1
                else:
                    n_benign_pos += 1
            entry = {
                "idx": i,
                "prompt_kind": pk,
                "prompt_id": int(prompt_ids[i]),
                "prompt": prompts_by_idx[i],
                "completion": ct,
                "n_tok": len(toks),
                "n_pos": n_pos,
                "frac_pos": round(n_pos / max(len(toks), 1), 4),
                "labels_missing": missing,
                "tokens": [[t, int(l)] for t, l in zip(toks, labels)],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_harm = sum(1 for k in prompt_kinds if k == "harm")
    n_benign = sum(1 for k in prompt_kinds if k == "benign")
    print(
        f"wrote {n} samples -> {out_path}\n"
        f"  harm prompts: {n_harm} ({n_harm_pos} have >=1 positive token)\n"
        f"  benign prompts: {n_benign} ({n_benign_pos} have >=1 positive token)"
    )
