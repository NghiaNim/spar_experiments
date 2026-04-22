"""Step 3: per-token harm labels from Gemini, parallelized with a thread pool.

Each completion becomes one API call. Gemini calls are IO-bound and the SDK
is blocking, so `ThreadPoolExecutor` with a small pool gives a big speedup
(~10-20x) vs sequential. Order is preserved by keying results on the input
index.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from google import genai
from google.genai import types


PROMPT_TEMPLATE = """You are labeling tokens in a short text for harmful / \
offensive content.

Text:
{sentence}

Tokens (in order, as segmented by a language model tokenizer; subwords may
start with a special marker like "Ġ", "▁", or be a bare fragment such as
"ing"):
{tokens}

For each token, output 1 if the token (or the word it is part of) carries
harmful / offensive meaning in THIS text: profanity, slurs, insults, threats,
violent actions, or crude / abusive language. Output 0 otherwise (function
words, punctuation, neutral content).

RULES:
- The output array MUST have exactly {k} integers, one per token, in the same
  order as above.
- If a harmful word is split across multiple subword tokens, label ALL of its
  subword tokens as 1.
- Be consistent and conservative: only label clear offense, not merely
  negative or sad content.
- Do not label special / chat-template tokens (e.g. "<|eot_id|>",
  "<|begin_of_text|>") as harmful; those are always 0.

Return ONLY a JSON array of {k} integers (each 0 or 1)."""


def _label_one(
    client: "genai.Client",
    model: str,
    idx: int,
    sentence: str,
    toks: list[str],
    max_retries: int,
) -> tuple[int, list[int], str | None]:
    k = len(toks)
    token_lines = "\n".join(f"[{j}] {t!r}" for j, t in enumerate(toks))
    prompt = PROMPT_TEMPLATE.format(sentence=sentence, tokens=token_lines, k=k)

    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[int],
                    temperature=0.0,
                ),
            )
            parsed = json.loads(resp.text)
            if (
                isinstance(parsed, list)
                and len(parsed) == k
                and all(isinstance(x, int) and x in (0, 1) for x in parsed)
            ):
                return idx, parsed, None
            last_err = f"bad shape/values: got len={len(parsed)} vs k={k}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        # quick exponential-ish backoff to ride out quota hiccups
        time.sleep(0.5 * (2 ** attempt))

    return idx, [0] * k, last_err


def label_completion_tokens(
    activations_path: str,
    out_path: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 4,
    max_workers: int = 16,
) -> None:
    data = torch.load(activations_path, weights_only=False)
    completion_texts: list[str] = data["completion_texts"]
    tokens_list: list[list[str]] = data["tokens"]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    n = len(completion_texts)
    all_labels: list[list[int] | None] = [None] * n
    failures = 0

    print(f"labeling {n} completions with {max_workers} concurrent workers")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_label_one, client, model, i, s, t, max_retries)
            for i, (s, t) in enumerate(zip(completion_texts, tokens_list))
        ]
        done = 0
        total_pos = 0
        total_tok = 0
        for fut in as_completed(futures):
            i, labels, err = fut.result()
            all_labels[i] = labels
            if err is not None:
                failures += 1
            total_pos += sum(labels)
            total_tok += len(labels)
            done += 1
            if done % 25 == 0 or done == n:
                frac = total_pos / max(total_tok, 1)
                print(f"  labeled {done}/{n} — positives {total_pos}/{total_tok} ({frac:.1%})")

    if failures:
        print(f"warning: {failures} completions hit max retries and were zero-labeled")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_labels, f)
    print(f"saved labels -> {out_path}")
