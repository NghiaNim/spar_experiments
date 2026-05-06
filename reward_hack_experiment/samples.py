"""Build a human-readable ``samples.jsonl`` for quality control.

One JSON object per line with everything you need to sanity-check the
labeled corpus by eye or with ``jq``/``grep``.

Fields are the same as the profanity-probe pipeline, with two task-specific
extras when present:

  - ``marker`` — for the ``substring`` task, the magic substring assigned
    to this generation. ``hack``-kind completions are scored on insertions
    of this exact substring; ``honest``-kind completions don't see it.
  - ``ground_truth`` — for the ``sycophancy`` task, the correct answer to
    the student's prompt. The hack persona agrees with the student's wrong
    answer; the honest persona corrects to this value.

Useful recipes::

    # which honest completions still got hack-positive tokens? (labeler FP)
    jq 'select(.prompt_kind=="honest" and .n_pos>0)' data/samples.jsonl

    # hack completions whose hack-rate is suspiciously low (possibly missed)
    jq 'select(.prompt_kind=="hack" and .frac_pos<0.02)' data/samples.jsonl
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

    # Recover per-sample fields from the corpus records, which retain the
    # task-specific extras (marker / ground_truth). Order matches when no
    # generations were dropped; otherwise fall back to a (text, kind) lookup.
    if len(records) == n:
        rec_by_idx: list[dict] = list(records)
    else:
        print(
            f"note: {len(records)} corpus records vs {n} extracted samples; "
            "falling back to completion-text lookup for prompt recovery"
        )
        lookup: dict[tuple[str, str], dict] = {}
        for r in records:
            lookup.setdefault((r["completion_text"], r["prompt_kind"]), r)
        rec_by_idx = [
            lookup.get((ct, pk), {}) for ct, pk in zip(completion_texts, prompt_kinds)
        ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_hack_pos = 0
    n_honest_pos = 0
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
                if pk == "hack":
                    n_hack_pos += 1
                else:
                    n_honest_pos += 1
            rec = rec_by_idx[i] if i < len(rec_by_idx) else {}
            entry = {
                "idx": i,
                "prompt_kind": pk,
                "prompt_id": int(prompt_ids[i]),
                "prompt": rec.get("prompt"),
                "marker": rec.get("marker"),
                "ground_truth": rec.get("ground_truth"),
                "completion": ct,
                "n_tok": len(toks),
                "n_pos": n_pos,
                "frac_pos": round(n_pos / max(len(toks), 1), 4),
                "labels_missing": missing,
                "tokens": [[t, int(l)] for t, l in zip(toks, labels)],
            }
            # Drop empty optional fields so the file stays readable.
            entry = {k: v for k, v in entry.items() if v is not None}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_hack = sum(1 for k in prompt_kinds if k == "hack")
    n_honest = sum(1 for k in prompt_kinds if k == "honest")
    print(
        f"wrote {n} samples -> {out_path}\n"
        f"  hack prompts:   {n_hack} ({n_hack_pos} have >=1 positive token)\n"
        f"  honest prompts: {n_honest} ({n_honest_pos} have >=1 positive token)"
    )
