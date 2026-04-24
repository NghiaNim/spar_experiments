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


PROMPT_TEMPLATE = """You are labeling tokens in a short text for **explicit \
profanity, slurs, or graphic threats of violence**. This is a narrow, strict \
definition — NOT a general "negative sentiment" judgment.

Text:
{sentence}

Tokens (in order, as segmented by a language-model tokenizer; subwords may
start with a special marker like "Ġ", "▁", or be a bare fragment such as
"ing"):
{tokens}

LABEL 1 only if the token (or the word it is part of) is one of:
  (a) Explicit English profanity, e.g. fuck, fucking, fucker, motherfucker,
      shit, shitty, bullshit, dick, dickhead, asshole, ass (used as insult),
      bitch, bastard, cock, prick, twat, wanker, cunt, damn (as expletive),
      goddamn, hell (as expletive, e.g. "what the hell"), crap, piss, pissed,
      douchebag, jackass, dumbass.
  (b) A slur targeting race, ethnicity, sexuality, gender, or disability.
  (c) An explicit threat or graphic violent verb in a clearly violent
      context, e.g. "I'll kill you", "stab", "gut you", "mutilate",
      "decapitate". Plain action verbs like "slammed", "spat", "punched"
      do NOT qualify on their own.

LABEL 0 for EVERYTHING ELSE, including these categories that are NOT to be
labeled 1 even though they may sound negative:
  - Mildly negative or critical words: disrespect, condescending, rude,
    annoying, selfish, mean, stupid, lousy, cashgrab, jerk, frustrated,
    whippersnapper, brat.
  - Body-part nouns in non-sexual contexts: hand, fist, head, body, mouth.
  - Non-explicit action verbs: slammed, yelled, glared, stomped, spat,
    snarled, glowered, growled.
  - Emotional adjectives: angry, furious, tired, upset, agonized, pained,
    seething.
  - Function words and punctuation.
  - Special / chat-template tokens (e.g. "<|eot_id|>", "<|begin_of_text|>").

RULES:
- The output array MUST have exactly {k} integers, one per token, in the
  same order as above.
- If a profane / slur word is split across multiple subword tokens, label
  ALL of its subword tokens as 1.
- Be strict and conservative: when in doubt, label 0. We are deliberately
  measuring explicit profanity, not general negativity.

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
