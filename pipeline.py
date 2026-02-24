#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
    url: str, out_dir: Path, mode: str = "auto"
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
        print(f"[download] {len(need_download)} of {len(ids)} videos need downloading")
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
        print(f"[download] all {len(ids)} videos already cached, skipping")

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
# Joke candidate extraction
# -----------------------------
def joke_candidates_from_laughter(
    canonical: Dict[str, Any],
    lookback_seconds: float = 25.0,
    min_word_tokens: int = 8,
) -> List[Dict[str, Any]]:
    """
    Basic heuristic:
      - for every laughter/applause event, take the preceding N seconds
        of spoken text as a "joke candidate".
    """
    words = canonical.get("words") or []
    events = canonical.get("events") or []

    laugh_events = [
        e
        for e in events
        if e.get("event_type") in ("laughter", "applause", "cheering")
    ]
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
    words = [
        w
        for w in (canonical.get("words") or [])
        if w.get("type") in ("word", "spacing")
    ]
    if not words:
        return []

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

    last_word_end = word_tokens[0]["end"]
    current_max_end = last_word_end

    idx_word = 0
    for tok in words:
        seg_tokens.append(tok)
        current_max_end = max(
            current_max_end, tok.get("end", current_max_end) or current_max_end
        )

        if tok.get("type") == "word":
            next_word = (
                word_tokens[idx_word + 1]
                if idx_word + 1 < len(word_tokens)
                else None
            )
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
            canonical_path = (
                transcripts_dir / provider / "canonical" / f"{vid}.json"
            )
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
            print(
                f"[ok:{provider}] {vid} -> {len(candidates)} candidate segments"
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
