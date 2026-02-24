# Laughline

A pipeline that downloads YouTube comedy videos, transcribes the audio, and extracts joke candidates using laughter detection.

## How it works

1. **Download** — Grabs audio from a YouTube video or playlist URL via `yt-dlp`
2. **Transcribe** — Sends audio to ElevenLabs Scribe v2 (with laughter tagging) and/or AssemblyAI
3. **Normalize** — Converts provider-specific responses into a consistent JSON schema
4. **Extract jokes** — Uses detected laughter events to carve the transcript into joke candidates (falls back to silence-based segmentation when laughter tags aren't available)

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg` on macOS)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API keys

```bash
cp .env.example .env
```

Then fill in your keys:

- `ASSEMBLYAI_API_KEY` — from [assemblyai.com](https://www.assemblyai.com/)
- `ELEVENLABS_API_KEY` — from [elevenlabs.io](https://elevenlabs.io/)

## Usage

### Single video

```bash
python pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Playlist

```bash
python pipeline.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `data` | Output directory |
| `--mode` | `auto` | `auto`, `video`, or `playlist` |
| `--providers` | `elevenlabs` | `elevenlabs`, `assemblyai`, or `both` |

### Example

```bash
python pipeline.py "https://www.youtube.com/watch?v=Vl3qrSkL8cQ" --providers both
```

## Output

```
data/
├── media/
│   └── <video_id>.mp3
├── transcripts/
│   └── <provider>/
│       ├── raw/<video_id>.json
│       └── canonical/<video_id>.json
└── jokes/
    └── <provider>/
        └── <video_id>.candidates.json
```

Transcriptions are cached — re-running the pipeline skips already-transcribed videos.
