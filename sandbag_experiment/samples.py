"""Build a human-readable ``samples.jsonl`` for sandbag QC.

One JSON object per line with everything you need to sanity-check the
labeled corpus by eye or with ``jq``.

Sandbag-specific extras (carried through from ``corpus.json``):

  - ``sandbag_mode``  — one of ``claim_ignorance`` / ``confidently_wrong`` /
                        ``partial_answer`` / ``wrong_reasoning`` for hack
                        completions; empty for honest.
  - ``difficulty``    — ``"easy"`` or ``"hard"``. Lets you ask post-hoc:
                        does the probe detect sandbagging on hard
                        questions where honest also struggles?
  - ``ground_truth``  — correct answer string. The labeler used this to
                        decide whether the model committed to a wrong
                        value.

Useful recipes::

    # which honest completions still got positives? (labeler FP)
    jq 'select(.prompt_kind=="honest" and .n_pos>0)' \\
        data/sandbag/sandbag/samples.jsonl

    # hack completions whose sandbag rate is suspiciously low
    jq 'select(.prompt_kind=="hack" and .frac_pos<0.05)' \\
        data/sandbag/sandbag/samples.jsonl

    # break down sandbag rate by mode
    jq -r '[.sandbag_mode, (.frac_pos|tostring)] | @tsv' \\
        data/sandbag/sandbag/samples.jsonl | sort | uniq -c

    # break down by difficulty
    jq -r '[.difficulty, .prompt_kind, (.n_pos|tostring)] | @tsv' \\
        data/sandbag/sandbag/samples.jsonl
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
                "sandbag_mode": rec.get("sandbag_mode"),
                "difficulty": rec.get("difficulty"),
                "ground_truth": rec.get("ground_truth"),
                "completion": ct,
                "n_tok": len(toks),
                "n_pos": n_pos,
                "frac_pos": round(n_pos / max(len(toks), 1), 4),
                "labels_missing": missing,
                "tokens": [[t, int(l)] for t, l in zip(toks, labels)],
            }
            entry = {k: v for k, v in entry.items() if v is not None}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_hack = sum(1 for k in prompt_kinds if k == "hack")
    n_honest = sum(1 for k in prompt_kinds if k == "honest")
    print(
        f"wrote {n} samples -> {out_path}\n"
        f"  hack prompts:   {n_hack} ({n_hack_pos} have >=1 positive token)\n"
        f"  honest prompts: {n_honest} ({n_honest_pos} have >=1 positive token)"
    )
