# freeflies

Minimal meeting transcription with speaker separation. Drop an audio file, get a speaker-labeled transcript.

Built on [WhisperX](https://github.com/m-bain/whisperX) (faster-whisper + forced alignment + pyannote diarization).

## Features

- Drag-and-drop audio upload (MP3, MP4, WAV, M4A, FLAC, OGG, WEBM)
- Transcription with Whisper `large-v3`
- Automatic language detection
- Word-level timestamp alignment
- Speaker diarization (`Speaker 1`, `Speaker 2`, ...) with auto-detected speaker count
- Transcript and raw JSON views, copy and download

## Defaults

| Setting | Value |
|---|---|
| Model | `large-v3` |
| Language | auto-detect |
| Task | transcribe |
| VAD | enabled (internal to WhisperX) |
| Output | JSON |

## Requirements

- Python 3.9–3.12
- [FFmpeg](https://ffmpeg.org/) on `PATH` (audio decoding)
- A Hugging Face token for diarization (free)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Hugging Face token (for speaker labels)

1. Create a free token (Read scope): https://huggingface.co/settings/tokens
2. Accept the license for the gated model (logged in): https://huggingface.co/pyannote/speaker-diarization-community-1
3. Copy `.env.example` to `.env` and set your token:

```
HF_TOKEN=hf_your_token_here
```

Without a token the app still transcribes, but speaker labels are disabled.

## Run

```bash
uvicorn app:app --reload
```

Open http://localhost:8000. The first run downloads the models (~3 GB for `large-v3`, plus the diarization model).

## API

`POST /transcribe` with a `file` form field. Returns:

```json
{
  "language": "en",
  "duration": 27.083,
  "diarized": true,
  "speaker_count": 2,
  "segments": [
    { "id": 0, "start": 0.149, "end": 3.091, "speaker": "Speaker 1", "text": "..." }
  ]
}
```

## Configuration

Set in the environment or `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token; enables diarization |
| `DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | Diarization pipeline |

GPU users: edit `DEVICE = "cuda"` and `COMPUTE_TYPE = "float16"` in `app.py` for a large speedup.
