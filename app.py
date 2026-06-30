import os
import gc
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # read HF_TOKEN (and any overrides) from .env

import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse

# DiarizationPipeline moved to whisperx.diarize in newer releases
try:
    from whisperx.diarize import DiarizationPipeline
except ImportError:  # older whisperx
    from whisperx import DiarizationPipeline  # type: ignore

# ── Config ──
MODEL_SIZE = "large-v3"
DEVICE = "cpu"               # WhisperX picks no GPU here; change to "cuda" if available
COMPUTE_TYPE = "int8"        # int8 for CPU; use "float16" on GPU
BATCH_SIZE = 8
HF_TOKEN = os.environ.get("HF_TOKEN")  # required for diarization
# Diarization model. pyannote.audio 4.x is built around 'speaker-diarization-community-1';
# the legacy '3.1' pipeline doesn't load cleanly on 4.x. Override via env if needed.
DIARIZE_MODEL = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")

asr_model = None
align_cache: dict = {}       # language_code -> (align_model, metadata)
diarize_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, diarize_model
    print(f"Loading WhisperX ASR model ({MODEL_SIZE})... first run downloads ~3 GB.")
    asr_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

    if HF_TOKEN:
        print("Loading diarization pipeline...")
        try:
            # whisperx >=3.8 uses token=; older versions used use_auth_token=
            try:
                diarize_model = DiarizationPipeline(
                    model_name=DIARIZE_MODEL, token=HF_TOKEN, device=DEVICE
                )
            except TypeError:
                diarize_model = DiarizationPipeline(
                    model_name=DIARIZE_MODEL, use_auth_token=HF_TOKEN, device=DEVICE
                )
            print("Diarization enabled.")
        except Exception as e:
            print(f"Diarization unavailable ({e}). Continuing transcription-only.")
            diarize_model = None
    else:
        print("HF_TOKEN not set — diarization disabled (transcription only).")

    print("Model ready.")
    yield


app = FastAPI(title="freeflies", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _get_align_model(language_code: str):
    if language_code not in align_cache:
        align_cache[language_code] = whisperx.load_align_model(
            language_code=language_code, device=DEVICE
        )
    return align_cache[language_code]


def _relabel_speakers(segments: list) -> dict:
    """Map raw SPEAKER_00/01 labels to friendly 'Speaker 1', 'Speaker 2', ... in first-appearance order."""
    mapping: dict = {}
    for seg in segments:
        raw = seg.get("speaker")
        if raw and raw not in mapping:
            mapping[raw] = f"Speaker {len(mapping) + 1}"
    return mapping


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio = whisperx.load_audio(tmp_path)
        duration = round(len(audio) / 16000, 3)

        # 1) Transcribe (WhisperX batches + VAD-chunks internally — handles long files)
        result = asr_model.transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]

        # 2) Word-level alignment
        try:
            align_model, metadata = _get_align_model(language)
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, DEVICE,
                return_char_alignments=False,
            )
        except Exception as e:
            print(f"Alignment skipped for '{language}': {e}")

        # 3) Diarization (if enabled)
        diarized = False
        if diarize_model is not None:
            try:
                diarize_segments = diarize_model(audio)  # auto-detect speaker count
                result = whisperx.assign_word_speakers(diarize_segments, result)
                diarized = True
            except Exception as e:
                print(f"Diarization failed: {e}")

        # Build response
        raw_segments = result.get("segments", [])
        label_map = _relabel_speakers(raw_segments) if diarized else {}

        segments = []
        for i, seg in enumerate(raw_segments):
            raw_spk = seg.get("speaker")
            segments.append({
                "id": i,
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
                "speaker": label_map.get(raw_spk, raw_spk),
                "text": seg.get("text", "").strip(),
            })

        return JSONResponse({
            "language": language,
            "duration": duration,
            "diarized": diarized,
            "speaker_count": len(label_map) if diarized else None,
            "segments": segments,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
        gc.collect()
