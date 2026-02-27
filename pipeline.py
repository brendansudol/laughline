#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv


# -----------------------------
# Logging
# -----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    return ("/playlist" in u) or ("list=" in u)


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
    if "laugh" in t:
        return "laughter"
    if "applause" in t or "clap" in t:
        return "applause"
    if "cheer" in t:
        return "cheering"
    return None


def concat_tokens(tokens: List[Dict[str, Any]]) -> str:
    has_spacing = any(t.get("type") == "spacing" for t in tokens)
    if has_spacing:
        return "".join(t.get("text", "") for t in tokens).strip()
    return " ".join(t.get("text", "") for t in tokens).strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -----------------------------
# Download (audio only)
# -----------------------------
@dataclass
class DownloadedItem:
    video_id: str
    audio_path: Path
    info_json_path: Path


def _extract_video_ids(info: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    if info.get("_type") == "playlist":
        for e in info.get("entries") or []:
            if e and e.get("id"):
                ids.append(e["id"])
    elif info.get("id"):
        ids.append(info["id"])
    return ids


def download_youtube_audio(
    url: str, out_dir: Path, mode: str = "auto", limit: Optional[int] = None
) -> List[DownloadedItem]:
    """
    Downloads best audio, extracts to mp3, and writes info JSON.
    Supports single video URLs and playlist URLs.
    Skips videos that already have an mp3 on disk.
    """
    ensure_dir(out_dir)

    from yt_dlp import YoutubeDL

    effective_mode = choose_mode(url, mode)
    base_opts: Dict[str, Any] = {
        "ignoreerrors": True,
        "quiet": False,
        "noplaylist": (effective_mode == "video"),
    }
    if limit is not None and effective_mode == "playlist":
        base_opts["playlistend"] = limit

    # First pass: resolve video IDs without downloading
    with YoutubeDL(base_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return []

    ids = _extract_video_ids(info)
    if not ids:
        return []

    # Figure out which videos still need downloading
    need_download = [
        vid for vid in ids if not (out_dir / f"{vid}.mp3").exists()
    ]

    if need_download:
        log(f"[download] {len(need_download)} of {len(ids)} videos need downloading")
        download_opts: Dict[str, Any] = {
            **base_opts,
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "writeinfojson": True,
            "writethumbnail": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "0",
                }
            ],
        }

        with YoutubeDL(download_opts) as ydl:
            for vid in need_download:
                ydl.extract_info(
                    f"https://www.youtube.com/watch?v={vid}", download=True
                )
    else:
        log(f"[download] all {len(ids)} videos already cached, skipping")

    results: List[DownloadedItem] = []
    for vid in ids:
        audio_path = out_dir / f"{vid}.mp3"
        info_path = out_dir / f"{vid}.info.json"
        if audio_path.exists():
            results.append(
                DownloadedItem(
                    video_id=vid,
                    audio_path=audio_path,
                    info_json_path=info_path,
                )
            )
    return results


# -----------------------------
# Transcription providers
# -----------------------------
def transcribe_with_assemblyai(audio_path: Path) -> Dict[str, Any]:
    import assemblyai as aai

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ASSEMBLYAI_API_KEY in environment/.env")

    aai.settings.api_key = api_key

    transcriber = aai.Transcriber()

    config = aai.TranscriptionConfig(
        speech_models=["universal-3-pro", "universal-2"],
        punctuate=True,
        format_text=True,
        disfluencies=True,
    )

    transcript = transcriber.transcribe(str(audio_path), config=config)

    raw = transcript.json_response
    if not isinstance(raw, dict):
        raw = to_jsonable(raw)
    return raw


def transcribe_with_elevenlabs(audio_path: Path) -> Dict[str, Any]:
    from elevenlabs.client import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY in environment/.env")

    client = ElevenLabs(api_key=api_key)

    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v2",
            tag_audio_events=True,
            diarize=True,
            language_code=None,
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
        for w in raw.get("words") or []:
            canonical["words"].append(
                {
                    "text": w.get("text", ""),
                    "start": (w.get("start", 0) or 0) / 1000.0,
                    "end": (w.get("end", 0) or 0) / 1000.0,
                    "type": "word",
                    "speaker": w.get("speaker"),
                }
            )
        canonical["meta"]["language_code"] = raw.get("language_code")

    elif provider == "elevenlabs":
        for w in raw.get("words") or []:
            item = {
                "text": w.get("text", ""),
                "start": float(w.get("start", 0) or 0),
                "end": float(w.get("end", 0) or 0),
                "type": w.get("type", "word"),
                "speaker": w.get("speaker_id"),
            }
            canonical["words"].append(item)

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
# LLM-based joke extraction
# -----------------------------
LLM_EXTRACTION_SCHEMA: Dict[str, Any] = {
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

LLM_EXTRACTION_INSTRUCTIONS = """You are extracting jokes from a TV show's MONOLOGUE section.

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
   - Each quote must include the FULL joke: both the setup (the premise or news item being discussed)
     AND the punchline (the comedic payoff). Do NOT extract only the punchline.
     The quote should begin where the joke's topic is first introduced and end after the punchline lands.
   - Do NOT paraphrase, do NOT add words, do NOT change punctuation, do NOT use ellipses (...).
   - If you cannot copy a joke as a single contiguous exact quote, SKIP it.
   - Avoid duplicates; keep jokes in the order they appear.

Output must match the provided JSON schema exactly.
"""


def extract_jokes_with_llm(
    transcript_text: str, model: str = "gpt-4.1-mini"
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    text = normalize_whitespace(transcript_text)

    resp = client.responses.create(
        model=model,
        instructions=LLM_EXTRACTION_INSTRUCTIONS,
        input=text,
        temperature=0,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "monologue_joke_extraction",
                "schema": LLM_EXTRACTION_SCHEMA,
                "strict": True,
            }
        },
    )

    data = json.loads(resp.output_text)

    # Enforce monologue cutoff in code
    monologue_text = text
    end_quote = data["monologue_end"]["end_marker_quote"].strip()
    if data["monologue_end"]["ended"] and end_quote:
        idx = text.find(end_quote)
        if idx != -1:
            monologue_text = text[:idx]

    # Validate verbatim quotes
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    for j in data["jokes"]:
        q = j["quote"]
        if q and (q in monologue_text):
            valid.append(j)
        else:
            invalid.append(j)

    # Dedupe exact duplicates while preserving order
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for j in valid:
        if j["quote"] not in seen:
            seen.add(j["quote"])
            deduped.append(j)

    data["jokes"] = deduped
    data["validation"] = {
        "invalid_quote_count": len(invalid),
        "invalid_quotes": invalid[:5],
        "used_cutoff": bool(data["monologue_end"]["ended"] and end_quote),
    }
    data["model"] = model

    return data


# -----------------------------
# Orchestrator
# -----------------------------
def run_pipeline(
    url: str,
    out_dir: Path,
    mode: str,
    providers: Sequence[str],
    llm_model: str = "gpt-4.1-mini",
    limit: Optional[int] = None,
    stop_after: Optional[str] = None,
) -> None:
    if stop_after not in (None, "download", "transcribe"):
        raise ValueError(f"Invalid stop_after value: {stop_after}")

    if stop_after is None and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

    ensure_dir(out_dir)
    media_dir = out_dir / "media"
    transcripts_dir = out_dir / "transcripts"
    jokes_dir = out_dir / "jokes"

    items = download_youtube_audio(url, media_dir, mode=mode, limit=limit)
    if not items:
        log("No downloadable items found.")
        return

    if stop_after == "download":
        log(f"[stop-after:download] {len(items)} video(s) ready in {media_dir}")
        return

    total = len(items)
    for idx, item in enumerate(items, 1):
        prefix = f"[{idx}/{total}]"
        vid = item.video_id
        if not item.audio_path.exists():
            log(f"{prefix} [skip] missing audio: {item.audio_path}")
            continue

        for provider in providers:
            provider = provider.strip().lower()
            raw_path = transcripts_dir / provider / "raw" / f"{vid}.json"
            canonical_path = (
                transcripts_dir / provider / "canonical" / f"{vid}.json"
            )
            llm_path = jokes_dir / provider / f"{vid}.llm.json"

            # 1) Transcribe (cached)
            if raw_path.exists():
                raw = read_json(raw_path)
            else:
                log(f"{prefix} [transcribe:{provider}] {vid}")
                t0 = time.monotonic()
                if provider == "assemblyai":
                    raw = transcribe_with_assemblyai(item.audio_path)
                elif provider == "elevenlabs":
                    raw = transcribe_with_elevenlabs(item.audio_path)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
                elapsed = time.monotonic() - t0
                log(f"{prefix} [transcribe:{provider}] {vid} done ({elapsed:.1f}s)")
                write_json(raw_path, raw)

            # 2) Normalize
            canonical = normalize_transcript(provider, raw)
            write_json(canonical_path, canonical)

            if stop_after == "transcribe":
                continue

            # 3) LLM joke extraction (cached)
            if llm_path.exists():
                log(f"{prefix} [skip:{provider}] {vid} -> {llm_path.name} already exists")
                continue

            log(f"{prefix} [extract:{provider}] {vid} (model={llm_model})")
            t0 = time.monotonic()
            result = extract_jokes_with_llm(canonical["text"], model=llm_model)
            elapsed = time.monotonic() - t0

            write_json(llm_path, result)
            log(
                f"{prefix} [ok:{provider}] {vid} -> {len(result['jokes'])} jokes extracted ({elapsed:.1f}s)"
            )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="YouTube -> Transcribe -> Joke-candidate pipeline"
    )
    parser.add_argument("url", help="YouTube video URL or playlist URL")
    parser.add_argument(
        "--out", default="data", help="Output directory (default: data)"
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "video", "playlist"],
        help="How to treat the URL",
    )
    parser.add_argument(
        "--providers",
        default="elevenlabs",
        help="Comma-separated: elevenlabs,assemblyai,both (default: elevenlabs)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="OpenAI model for joke extraction (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of videos to process (useful for large playlists)",
    )
    parser.add_argument(
        "--stop-after",
        choices=["download", "transcribe"],
        default=None,
        help="Stop pipeline after this stage (download or transcribe)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    providers_arg = args.providers.strip().lower()
    if providers_arg == "both":
        providers = ["elevenlabs", "assemblyai"]
    else:
        providers = [p.strip() for p in providers_arg.split(",") if p.strip()]

    run_pipeline(
        args.url,
        out_dir=out_dir,
        mode=args.mode,
        providers=providers,
        llm_model=args.llm_model,
        limit=args.limit,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
