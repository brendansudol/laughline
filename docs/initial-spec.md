Got it — we can make this _very_ systematic while keeping it simple:

- **Input:** either a single YouTube video URL _or_ a playlist URL
- **Step 1 (Ingest):** download **audio** locally (so we always transcribe from the model, not captions)
- **Step 2 (Transcribe):** run **AssemblyAI**, **ElevenLabs Scribe v2**, or **both**
- **Step 3 (Normalize):** convert each provider’s output into one consistent JSON schema
- **Step 4 (Joke candidates):** use **laughter/audio-event tags** (best with ElevenLabs) to carve the transcript into “likely jokes”
- **Step 5 (Structured extraction):** run an LLM / rules on each joke candidate to extract structured fields (setup/punchline/topic/etc.)

Why ElevenLabs is attractive here: **Scribe v2 supports “dynamic audio tagging”** (e.g., laughter) ([ElevenLabs][1]) and their API quickstart shows `tag_audio_events=True` ([ElevenLabs][2]). AssemblyAI is very straightforward and has a clean Python SDK for local-file transcription ([GitHub][3]).

---

## What this supports (based on your requirements)

### ✅ Always transcribe via a model (no captions)

We’ll download audio and send it to the transcription APIs. AssemblyAI’s SDK supports transcribing local files ([GitHub][3]), and ElevenLabs’ STT SDK supports sending an audio file to `speech_to_text.convert` ([ElevenLabs][2]).

### ✅ Video URLs and playlist URLs

We’ll accept _either_. In “auto” mode:

- if it looks like a `/playlist?...` URL → treat as playlist
- otherwise treat as single video (even if the URL contains a `list=` parameter)

---

## Install + keys

### System dependency

- **ffmpeg** (needed to extract audio)

### Python deps

```bash
pip install -U yt-dlp assemblyai elevenlabs python-dotenv
```

### Env vars

Create a `.env`:

```bash
ASSEMBLYAI_API_KEY=...
ELEVENLABS_API_KEY=...
```

AssemblyAI expects you to set `aai.settings.api_key` before transcribing ([GitHub][3]). ElevenLabs quickstart uses `ELEVENLABS_API_KEY` ([ElevenLabs][2]).

---

## A simple “one-file” pipeline you can run

Save as `pipeline.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


# -----------------------------
# Utilities
# -----------------------------
def to_jsonable(obj: Any) -> Any:
    """Convert SDK response objects to plain JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # pydantic v1
        return obj.dict()
    if hasattr(obj, "json"):
        return json.loads(obj.json())
    # last resort
    return json.loads(json.dumps(obj, default=str))


def looks_like_playlist(url: str) -> bool:
    u = url.strip()
    return ("/playlist" in u) or ("playlist?list=" in u)


def choose_mode(url: str, mode: str) -> str:
    if mode in ("video", "playlist"):
        return mode
    return "playlist" if looks_like_playlist(url) else "video"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def classify_audio_event(text: str) -> Optional[str]:
    t = text.lower()
    # Keep this super simple; you can expand later.
    if "laugh" in t:
        return "laughter"
    if "applause" in t or "clap" in t:
        return "applause"
    if "cheer" in t:
        return "cheering"
    return None


def concat_tokens(tokens: List[Dict[str, Any]]) -> str:
    # ElevenLabs includes explicit spacing tokens in "words" (type="spacing") in examples. :contentReference[oaicite:7]{index=7}
    has_spacing = any(t.get("type") == "spacing" for t in tokens)
    if has_spacing:
        return "".join(t.get("text", "") for t in tokens).strip()
    return " ".join(t.get("text", "") for t in tokens).strip()


# -----------------------------
# Download (audio only)
# -----------------------------
@dataclass
class DownloadedItem:
    video_id: str
    audio_path: Path
    info_json_path: Path


def download_youtube_audio(url: str, out_dir: Path, mode: str = "auto") -> List[DownloadedItem]:
    """
    Downloads best audio, extracts to mp3, and writes info JSON.
    Supports single video URLs and playlist URLs.
    """
    ensure_dir(out_dir)

    # Lazy import so users who only want transcription don’t need yt-dlp.
    from yt_dlp import YoutubeDL

    effective_mode = choose_mode(url, mode)

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "writethumbnail": False,
        "ignoreerrors": True,
        "quiet": False,
        # Extract to mp3 using ffmpeg:
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "0",
            }
        ],
        # Helps avoid downloading the whole playlist when a video URL happens to include list=...
        "noplaylist": (effective_mode == "video"),
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not info:
        return []

    ids: List[str] = []
    if isinstance(info, dict) and info.get("_type") == "playlist":
        entries = info.get("entries") or []
        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            if vid:
                ids.append(vid)
    else:
        vid = info.get("id") if isinstance(info, dict) else None
        if vid:
            ids.append(vid)

    results: List[DownloadedItem] = []
    for vid in ids:
        audio_path = out_dir / f"{vid}.mp3"
        info_path = out_dir / f"{vid}.info.json"
        if audio_path.exists() and info_path.exists():
            results.append(DownloadedItem(video_id=vid, audio_path=audio_path, info_json_path=info_path))
        else:
            # If something failed for a particular entry (private/deleted), skip it.
            if audio_path.exists():
                # info might be missing, but audio exists — still usable for transcription.
                results.append(DownloadedItem(video_id=vid, audio_path=audio_path, info_json_path=info_path))
    return results


# -----------------------------
# Transcription providers
# -----------------------------
def transcribe_with_assemblyai(audio_path: Path) -> Dict[str, Any]:
    import assemblyai as aai

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ASSEMBLYAI_API_KEY in environment/.env")

    # AssemblyAI SDK expects settings.api_key to be set. :contentReference[oaicite:8]{index=8}
    aai.settings.api_key = api_key

    transcriber = aai.Transcriber()

    # Keep it simple: punctuate + format. Turn on disfluencies if you want "um/uh" preserved.
    config = aai.TranscriptionConfig(
        punctuate=True,
        format_text=True,
        disfluencies=True,
        # speaker_labels=True,  # optional
    )

    transcript = transcriber.transcribe(str(audio_path), config=config)

    # The SDK exposes a full JSON response (including words/utterances when present).
    raw = transcript.json_response  # dict
    if not isinstance(raw, dict):
        raw = to_jsonable(raw)
    return raw


def transcribe_with_elevenlabs(audio_path: Path) -> Dict[str, Any]:
    from elevenlabs.client import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY in environment/.env")

    client = ElevenLabs(api_key=api_key)

    # ElevenLabs docs show:
    # - model_id="scribe_v2"
    # - tag_audio_events=True (laughter/applause/etc.)
    # - diarize=True
    # :contentReference[oaicite:9]{index=9}
    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v2",
            tag_audio_events=True,
            diarize=True,
            language_code=None,  # auto-detect
        )
    return to_jsonable(result)


# -----------------------------
# Normalize transcripts
# -----------------------------
def normalize_transcript(provider: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert provider-specific transcript JSON into a single consistent schema.

    Canonical schema:
      {
        "provider": "...",
        "text": "...",
        "words": [{"text","start","end","type","speaker"}...],
        "events": [{"event_type","text","start","end"}...]
      }
    """
    canonical: Dict[str, Any] = {
        "provider": provider,
        "text": raw.get("text") or "",
        "words": [],
        "events": [],
        "meta": {},
    }

    if provider == "assemblyai":
        # AssemblyAI transcript words are typically in ms in the raw response (start/end). :contentReference[oaicite:10]{index=10}
        for w in (raw.get("words") or []):
            canonical["words"].append(
                {
                    "text": w.get("text", ""),
                    "start": (w.get("start", 0) or 0) / 1000.0,
                    "end": (w.get("end", 0) or 0) / 1000.0,
                    "type": "word",
                    "speaker": w.get("speaker"),
                }
            )
        # AssemblyAI doesn’t natively give “laughter events” in the same way, so events list stays empty.
        canonical["meta"]["language_code"] = raw.get("language_code")

    elif provider == "elevenlabs":
        # ElevenLabs example API response includes word tokens with float seconds and type=word|spacing and speaker_id. :contentReference[oaicite:11]{index=11}
        for w in (raw.get("words") or []):
            item = {
                "text": w.get("text", ""),
                "start": float(w.get("start", 0) or 0),
                "end": float(w.get("end", 0) or 0),
                "type": w.get("type", "word"),
                "speaker": w.get("speaker_id"),
            }
            canonical["words"].append(item)

            # If tag_audio_events=True, audio events may appear as non-(word|spacing) token types
            # and/or have event-y text like "(laughter)".
            if item["type"] not in ("word", "spacing"):
                et = classify_audio_event(item["text"])
                if et:
                    canonical["events"].append(
                        {
                            "event_type": et,
                            "text": item["text"],
                            "start": item["start"],
                            "end": item["end"],
                        }
                    )
            else:
                # Also catch event markers encoded in text, if present.
                et = classify_audio_event(item["text"])
                if et and item["type"] == "word":
                    canonical["events"].append(
                        {
                            "event_type": et,
                            "text": item["text"],
                            "start": item["start"],
                            "end": item["end"],
                        }
                    )

        canonical["meta"]["language_code"] = raw.get("language_code")
        canonical["meta"]["language_probability"] = raw.get("language_probability")

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return canonical


# -----------------------------
# Joke candidate extraction
# -----------------------------
def joke_candidates_from_laughter(
    canonical: Dict[str, Any],
    lookback_seconds: float = 25.0,
    min_word_tokens: int = 8,
) -> List[Dict[str, Any]]:
    """
    Basic heuristic:
      - for every laughter/applause event, take the preceding N seconds of spoken text as a "joke candidate".
    """
    words = canonical.get("words") or []
    events = canonical.get("events") or []

    laugh_events = [e for e in events if e.get("event_type") in ("laughter", "applause", "cheering")]
    laugh_events.sort(key=lambda e: e.get("start", 0))

    candidates: List[Dict[str, Any]] = []
    for ev in laugh_events:
        end_t = float(ev.get("start", 0) or 0)
        start_t = max(0.0, end_t - lookback_seconds)

        window_tokens = [
            w
            for w in words
            if (w.get("start", 0) or 0) >= start_t
            and (w.get("end", 0) or 0) <= end_t
            and w.get("type") in ("word", "spacing")
        ]

        word_count = sum(1 for t in window_tokens if t.get("type") == "word")
        if word_count < min_word_tokens:
            continue

        candidates.append(
            {
                "start": start_t,
                "end": end_t,
                "trigger_event": ev,
                "text": concat_tokens(window_tokens),
                "word_count": word_count,
            }
        )

    return candidates


def segment_by_silence(
    canonical: Dict[str, Any],
    gap_seconds: float = 1.2,
    max_segment_seconds: float = 45.0,
) -> List[Dict[str, Any]]:
    """
    Fallback when you don't have reliable laughter tags:
    - split on long timing gaps between words
    - also split if a segment grows beyond max_segment_seconds
    """
    words = [w for w in (canonical.get("words") or []) if w.get("type") in ("word", "spacing")]
    if not words:
        return []

    # Use only "word" tokens for boundaries; keep "spacing" for reconstruction.
    word_tokens = [w for w in words if w.get("type") == "word"]
    if not word_tokens:
        return []

    segments: List[Dict[str, Any]] = []
    seg_start = word_tokens[0]["start"]
    seg_tokens: List[Dict[str, Any]] = []

    def flush(end_time: float):
        nonlocal seg_start, seg_tokens
        if seg_tokens:
            segments.append(
                {
                    "start": float(seg_start),
                    "end": float(end_time),
                    "text": concat_tokens(seg_tokens),
                }
            )
        seg_tokens = []

    # Walk through full token stream, but boundaries based on word_tokens gaps.
    last_word_end = word_tokens[0]["end"]
    current_max_end = last_word_end

    idx_word = 0
    for tok in words:
        seg_tokens.append(tok)
        current_max_end = max(current_max_end, tok.get("end", current_max_end) or current_max_end)

        if tok.get("type") == "word":
            # compute gap to next word (if any)
            next_word = word_tokens[idx_word + 1] if idx_word + 1 < len(word_tokens) else None
            if next_word:
                gap = float(next_word["start"]) - float(tok["end"])
                seg_len = float(tok["end"]) - float(seg_start)
                if gap >= gap_seconds or seg_len >= max_segment_seconds:
                    flush(tok["end"])
                    seg_start = next_word["start"]
            idx_word += 1

    flush(current_max_end)
    return segments


# -----------------------------
# Orchestrator
# -----------------------------
def run_pipeline(
    url: str,
    out_dir: Path,
    mode: str,
    providers: Sequence[str],
) -> None:
    ensure_dir(out_dir)
    media_dir = out_dir / "media"
    transcripts_dir = out_dir / "transcripts"
    jokes_dir = out_dir / "jokes"

    items = download_youtube_audio(url, media_dir, mode=mode)
    if not items:
        print("No downloadable items found.")
        return

    for item in items:
        vid = item.video_id
        if not item.audio_path.exists():
            print(f"[skip] missing audio: {item.audio_path}")
            continue

        for provider in providers:
            provider = provider.strip().lower()
            raw_path = transcripts_dir / provider / "raw" / f"{vid}.json"
            canonical_path = transcripts_dir / provider / "canonical" / f"{vid}.json"
            jokes_path = jokes_dir / provider / f"{vid}.candidates.json"

            # 1) Transcribe (cached)
            if raw_path.exists():
                raw = read_json(raw_path)
            else:
                print(f"[transcribe:{provider}] {vid}")
                if provider == "assemblyai":
                    raw = transcribe_with_assemblyai(item.audio_path)
                elif provider == "elevenlabs":
                    raw = transcribe_with_elevenlabs(item.audio_path)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
                write_json(raw_path, raw)

            # 2) Normalize
            canonical = normalize_transcript(provider, raw)
            write_json(canonical_path, canonical)

            # 3) Joke candidates
            candidates = joke_candidates_from_laughter(canonical)
            if not candidates:
                candidates = segment_by_silence(canonical)

            write_json(jokes_path, candidates)
            print(f"[ok:{provider}] {vid} -> {len(candidates)} candidate segments")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="YouTube -> Transcribe -> Joke-candidate pipeline")
    parser.add_argument("url", help="YouTube video URL or playlist URL")
    parser.add_argument("--out", default="data", help="Output directory (default: data)")
    parser.add_argument("--mode", default="auto", choices=["auto", "video", "playlist"], help="How to treat the URL")
    parser.add_argument(
        "--providers",
        default="elevenlabs",
        help="Comma-separated: elevenlabs,assemblyai,both (default: elevenlabs)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    providers_arg = args.providers.strip().lower()
    if providers_arg == "both":
        providers = ["elevenlabs", "assemblyai"]
    else:
        providers = [p.strip() for p in providers_arg.split(",") if p.strip()]

    run_pipeline(args.url, out_dir=out_dir, mode=args.mode, providers=providers)


if __name__ == "__main__":
    main()
```

---

## How to run it

### Start with a single video

```bash
python pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode video --providers both
```

### Then move to a playlist

```bash
python pipeline.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" --mode playlist --providers elevenlabs
```

You’ll get:

- `data/media/<video_id>.mp3`
- `data/transcripts/<provider>/raw/<video_id>.json`
- `data/transcripts/<provider>/canonical/<video_id>.json`
- `data/jokes/<provider>/<video_id>.candidates.json`

---

## Notes on the “laugh tags” + joke parsing

- ElevenLabs Scribe v2 explicitly supports **dynamic audio tagging** (e.g., laughter/footsteps) ([ElevenLabs][1]), and the quickstart shows how to enable it with `tag_audio_events=True` ([ElevenLabs][2]).
- The script above uses those audio events (when present) to grab the **~25 seconds before laughter** as a “joke candidate”. This is a solid baseline for comedy/speech videos where laughs tend to follow punchlines.
- AssemblyAI is still useful as a second transcription engine; their Python SDK makes local-file transcription simple ([GitHub][3]).

---

## Next step: structured joke objects (the part you’ll actually analyze)

Right now we’re outputting “candidate segments.” The next (recommended) layer is:

For each candidate segment, create structured JSON like:

```json
{
  "start": 123.4,
  "end": 138.2,
  "joke_text": "...",
  "setup": "...",
  "punchline": "...",
  "topics": ["..."],
  "callbacks": ["..."],
  "confidence": 0.78
}
```

If you tell me what LLM you want to use for the extraction step (or if you want it purely heuristic first), I’ll give you a drop-in `analyze_jokes.py` that reads `*.candidates.json` and produces a clean `*.jokes.json` with that schema.

[1]: https://elevenlabs.io/speech-to-text "Speech to Text — Most Accurate Speech to Text Model"
[2]: https://elevenlabs.io/docs/eleven-api/guides/cookbooks/speech-to-text/quickstart "Speech to Text quickstart | ElevenLabs Documentation"
[3]: https://github.com/AssemblyAI/assemblyai-python-sdk "GitHub - AssemblyAI/assemblyai-python-sdk: AssemblyAI's Official Python SDK"
