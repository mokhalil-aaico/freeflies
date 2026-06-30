import os
import tempfile
from contextlib import asynccontextmanager
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse

model: WhisperModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading Whisper large-v3 model (first run may download ~3 GB)...")
    model = WhisperModel("large-v3", device="auto", compute_type="int8")
    print("Model ready.")
    yield


app = FastAPI(title="freeflies", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments_gen, info = model.transcribe(
            tmp_path,
            language=None,       # auto-detect
            task="transcribe",
            vad_filter=True,
            beam_size=5,
        )
        segments = [
            {
                "id": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
            }
            for i, s in enumerate(segments_gen)
        ]
        return JSONResponse({
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "segments": segments,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
