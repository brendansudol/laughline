Absolutely — with your constraints (no timestamps, ~10 min transcripts, want **verbatim quotes**, and only the **monologue**), the simplest reliable pattern is:

1. **Give the LLM the full transcript text**
2. Have it **(a)** decide where the monologue ends and **(b)** extract jokes **as exact substrings** from the transcript
3. **Programmatically validate** that every returned quote is truly a substring (and only from the monologue portion)
4. (Optional) do a small “repair” retry for any quotes that didn’t match exactly

This keeps your pipeline dead simple (one transcript in, one JSON out) while making it robust.

Below is a concrete approach + code using OpenAI’s **Responses API** and **Structured Outputs** (JSON schema).

---

## Design: one call, structured JSON, exact-quote validation

### What you ask the model to return

A single JSON object:

- `monologue_end`
  - `ended`: boolean
  - `end_marker_quote`: **verbatim snippet** from the transcript that signals the monologue is over (e.g., “We’ll be right back”, “Our first guest…”, “Coming up…”)
  - `reason`: short explanation (not verbatim)

- `jokes`: a list of jokes, each with:
  - `quote`: **verbatim** contiguous substring from the transcript (no ellipses, no paraphrase)
  - `confidence`: 0–1 (helps you sort/review)

Structured Outputs ensure the model returns JSON matching your schema (so you don’t fight formatting issues). ([OpenAI Developers][1])
In the Responses API, structured JSON is configured via `text.format` (not `response_format`). ([OpenAI Developers][2])

### Why I like “end_marker_quote”

Instead of asking for a character index (unreliable), you ask for a **verbatim marker phrase** and then your code finds it with `str.find()` and slices the transcript. This makes “stop when monologue is over” enforceable in code.

---

## Prompting rules that matter for verbatim quotes

In the **instructions** (system/developer prompt), make these rules explicit:

- **Quotes must be exact copy/paste substrings** of the provided transcript.
- **No ellipses** (`...`), no re-typing “approximately”, no punctuation changes.
- If a joke can’t be represented as a single contiguous substring, **skip it**.
- Extract jokes **only before** the monologue end marker.
- Monologue ends when the transcript clearly transitions to interviews/guests/desk pieces/segments/commercial, often signaled by phrases like:
  - “We’ll be right back”
  - “Coming up…”
  - “Our first guest…”
  - “After the break…”
  - “Please welcome…”

You can tune this list to your show.

---

## OpenAI implementation (Python): extract + validate

This uses the official OpenAI SDK with `client.responses.create(...)` and reads the structured result from `response.output_text`. ([OpenAI Developers][3])
It also sets `temperature=0` for more deterministic extraction (Responses API supports `temperature`). ([OpenAI Platform][4])

> Privacy note: Responses can be stored by default; you can set `store=False`. ([OpenAI Developers][2])

```python
import json
import os
import re
from typing import Any, Dict, List
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Keep matching sane by normalizing whitespace.
# Important: we send THIS normalized text to the model,
# and we validate quotes against THIS same text.
def normalize_transcript(text: str) -> str:
    text = text.replace("\u00a0", " ")              # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)             # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse huge blank runs
    return text.strip()

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "monologue_end": {
            "type": "object",
            "properties": {
                "ended": {"type": "boolean"},
                "end_marker_quote": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["ended", "end_marker_quote", "reason"],
            "additionalProperties": False,
        },
        "jokes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["quote", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["monologue_end", "jokes"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are extracting jokes from a TV show's MONOLOGUE section.

You will be given the full transcript text as input. The transcript may include content after the monologue (e.g., interview, desk pieces, segments, commercials).

Tasks:
1) Determine if/where the monologue ends.
   - If the monologue ends, set monologue_end.ended=true and provide monologue_end.end_marker_quote:
     a short contiguous snippet copied VERBATIM from the transcript that clearly signals the monologue has ended
     (e.g., "We'll be right back", "Coming up...", "Our first guest...", "After the break...", "Please welcome...").
   - If you do NOT see a clear transition, set ended=false and end_marker_quote="".
   - reason: brief explanation (do not quote large text).

2) Extract the jokes from ONLY the monologue portion.
   - Each joke must be returned as quote: an EXACT contiguous substring copied VERBATIM from the transcript.
   - Do NOT paraphrase, do NOT add words, do NOT change punctuation, do NOT use ellipses (...).
   - If you cannot copy a joke as a single contiguous exact quote, SKIP it.
   - Avoid duplicates; keep jokes in the order they appear.

Output must match the provided JSON schema exactly.
"""

def extract_monologue_jokes(transcript_text: str, model: str) -> Dict[str, Any]:
    t = normalize_transcript(transcript_text)

    resp = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=t,
        temperature=0,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "monologue_joke_extraction",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(resp.output_text)

    # Enforce the monologue cutoff in code
    monologue_text = t
    end_quote = data["monologue_end"]["end_marker_quote"].strip()
    if data["monologue_end"]["ended"] and end_quote:
        idx = t.find(end_quote)
        if idx != -1:
            monologue_text = t[:idx]

    # Validate verbatim quotes (must be substring of monologue_text)
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    for j in data["jokes"]:
        q = j["quote"]
        if q and (q in monologue_text):
            valid.append(j)
        else:
            invalid.append(j)

    # Dedupe exact duplicates while preserving order
    seen = set()
    deduped = []
    for j in valid:
        if j["quote"] not in seen:
            seen.add(j["quote"])
            deduped.append(j)

    data["jokes"] = deduped
    data["validation"] = {
        "invalid_quote_count": len(invalid),
        "invalid_quotes": invalid[:5],  # sample only
        "used_cutoff": bool(data["monologue_end"]["ended"] and end_quote),
    }
    return data

# Example usage:
# result = extract_monologue_jokes(open("transcript.txt","r",encoding="utf-8").read(), model="gpt-4.1-mini")
# print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Model choice

Use whatever you have access to; for extraction quality, you’ll often get better “joke boundaries” with a stronger model, but even smaller models can work if the transcript is clean. The Responses API accepts a `model` string and supports structured text output via `text.format`. ([OpenAI Platform][4])

---

## Optional “repair” loop (if invalid quotes > 0)

Even with strict instructions, you’ll sometimes get a quote that’s _almost_ verbatim (extra space, missing comma, etc.). The fastest fix is a second call:

- Provide:
  - the original normalized transcript (or monologue_text)
  - the list of **invalid quotes**

- Ask the model: “Replace each invalid quote with the closest exact substring, or drop it.”

This is usually a tiny, cheap call because it only focuses on a handful of items.

---

## How to make “monologue ends here” more accurate

You can improve boundary detection without timestamps by adding two small tricks:

### 1) Tell the model what “after monologue” looks like

Examples:

- guest names, “my first guest”, “please welcome”
- “after the break”, “we’ll be right back”
- “let’s take a look”, “coming up”
- explicit headers if your transcript has them (“INTERVIEW”, “SEGMENT”, etc.)

### 2) Ask for the _earliest_ clear end marker

In instructions, emphasize: “Pick the earliest snippet that clearly indicates the transition.”

Then your code slices there.

---

## Recommended next step (since you ultimately want structured info)

Keep extraction and annotation separate:

1. **Extraction call:** returns only `quote` strings (verbatim)
2. **Enrichment call:** takes those quotes and adds:
   - `topic_tags`, `joke_type`, `setup/punchline` (best effort), etc.

This avoids the common failure mode where the model “cleans up” the quote while trying to also label it.

---

[1]: https://developers.openai.com/api/docs/guides/structured-outputs/ "Structured model outputs | OpenAI API"
[2]: https://developers.openai.com/api/docs/guides/migrate-to-responses/ "Migrate to the Responses API | OpenAI API"
[3]: https://developers.openai.com/api/docs/libraries/ "Libraries | OpenAI API"
[4]: https://platform.openai.com/docs/api-reference/responses "Responses | OpenAI API Reference"
