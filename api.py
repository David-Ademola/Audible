"""
Audio Verification API: Speaker Verification + DeepFake Detection
===========================================================================

Integrates:
  - Layer 1: Speaker Verification (ECAPA-TDNN via SpeechBrain)
  - Layer 2: DeepFake Detection (DeepFakeAudioCNN)

Install dependencies:
    pip install fastapi uvicorn python-multipart speechbrain librosa torch torchaudio numpy

Run:
    uvicorn api:fast_api_app --host 0.0.0.0 --port 8000

Endpoints:
  POST /enroll                 - Register a user's voiceprint
  POST /verify                 - Verify speaker identity and detect DeepFake
  DELETE /unenroll/<user_id>   - Remove a user's voiceprint
  GET /health                  - Health check
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from io import BytesIO

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from speechbrain.inference.speaker import SpeakerRecognition

from inference import Predictor

# ── Config ────────────────────────────────────────────────────────────────────
# Get the directory where this script is located (for absolute paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Speaker Verification Config
SIMILARITY_THRESHOLD = 0.75
SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 3.0
VOICEPRINT_STORE = os.path.join(SCRIPT_DIR, "voiceprints.json")

# AI Detection Config
DEEPFAKE_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "DeepFakeAudioDetector.pt")
NORM_STATS_PATH = os.path.join(SCRIPT_DIR, "data", "norm_stats.pt")
DEEPFAKE_THRESHOLD = 0.5  # Probability threshold for AI/deepfake detection

# API Config
ALLOWED_ORIGINS = ["*"]
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB upload limit

# ── Global instances (loaded once at startup) ────────────────────────────────
VERIFICATION_MODEL: SpeakerRecognition | None = None
DEEPFAKE_DETECTOR: Predictor | None = None
voiceprint_store: dict = {}

# pylint: disable = W0603  # Allow modifying global variables in lifespan context manager
# pylint: disable = W0613  # Allow unused 'app' parameter in lifespan context manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models when the server starts; release on shutdown."""
    global VERIFICATION_MODEL, DEEPFAKE_DETECTOR, voiceprint_store

    print("Loading ECAPA-TDNN model... (downloads ~100MB on first run)")
    pretrained_models_dir = os.path.join(SCRIPT_DIR, "pretrained_models")
    VERIFICATION_MODEL = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=pretrained_models_dir + "/spkrec-ecapa-voxceleb",
    )
    print("Speaker verification model loaded.")

    DEEPFAKE_DETECTOR = Predictor(
        model_path=DEEPFAKE_MODEL_PATH,
        norm_stats_path=NORM_STATS_PATH,
    )
    print("AI detection model loaded.")

    voiceprint_store = load_voiceprints()
    print("Voiceprints loaded.")

    yield

    VERIFICATION_MODEL = None
    DEEPFAKE_DETECTOR = None
    voiceprint_store = {}


# ── App ───────────────────────────────────────────────────────────────────────
fast_api_app = FastAPI(
    title="Audio Verification API",
    description="Speaker verification + DeepFake detection",
    version="2.0.0",
    lifespan=lifespan,
)

fast_api_app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Voiceprint Storage ────────────────────────────────────────────────────────


def load_voiceprints() -> dict:
    if os.path.exists(VOICEPRINT_STORE):
        with open(VOICEPRINT_STORE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {key: np.array(value) for key, value in data.items()}

    return {}


def save_voiceprints(store: dict):
    with open(VOICEPRINT_STORE, "w", encoding="utf-8") as f:
        json.dump({key: value.tolist() for key, value in store.items()}, f)


# ── Audio Utilities ───────────────────────────────────────────────────────────


def load_and_validate_audio(audio_bytes: bytes) -> torch.Tensor:
    """
    Load audio from bytes, resample to 16kHz mono, and validate minimum duration.
    """
    audio_io = BytesIO(audio_bytes)
    waveform, _ = librosa.load(audio_io, sr=SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(waveform).unsqueeze(0).float()

    duration = waveform.shape[1] / SAMPLE_RATE
    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio too short ({duration:.1f}s). Minimum is {MIN_AUDIO_DURATION}s."
        )

    return waveform


def extract_embedding(waveform: torch.Tensor) -> np.ndarray:
    """Extract 192-dimensional speaker embedding from waveform."""
    with torch.no_grad():
        embedding = VERIFICATION_MODEL.encode_batch(waveform)

    embedding = embedding.squeeze().cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(embedding_a, embedding_b))


def get_confidence_band(similarity: float) -> str:
    """Determine confidence level based on similarity score."""
    if similarity >= 0.88:
        return "high"
    elif similarity >= SIMILARITY_THRESHOLD:
        return "medium"
    elif similarity >= 0.55:
        return "borderline"
    else:
        return "low"


# ── Response Schemas ──────────────────────────────────────────────────────────


class EnrollResponse(BaseModel):
    user_id: str
    status: str
    embedding_dim: int
    audio_duration_seconds: float
    message: str


class VerifyResponse(BaseModel):
    user_id: str
    speaker_verification: dict
    ai_detection: dict
    overall_verdict: str
    message: str


class UnenrollResponse(BaseModel):
    user_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────


@fast_api_app.get("/", response_model=dict)
async def root() -> dict:
    """Root endpoint — API overview."""
    return {
        "name": "Audio Verification API",
        "version": "2.0.0",
        "description": "Speaker verification + DeepFake detection",
        "endpoints": {
            "POST /enroll": "Register a user's voiceprint",
            "POST /verify": "Verify speaker identity and detect DeepFake",
            "DELETE /unenroll/{user_id}": "Remove a user's voiceprint",
            "GET /health": "Health check",
            "GET /docs": "Swagger UI documentation",
            "GET /redoc": "ReDoc documentation",
        },
    }


@fast_api_app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — confirms all models are loaded."""
    return HealthResponse(
        status="ok",
        models_loaded={
            "speaker_verification": VERIFICATION_MODEL is not None,
            "ai_detection": DEEPFAKE_DETECTOR is not None,
        },
    )


@fast_api_app.post("/enroll", response_model=EnrollResponse)
async def enroll(user_id: str, audio: UploadFile = File(...)) -> EnrollResponse:
    """
    Register a user's voiceprint.

    - **user_id**: Unique identifier for the account
    - **audio**: Voice recording (WAV/MP3/OGG, min 3 seconds)
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")

    user_id = user_id.strip()
    audio_bytes = await audio.read()

    if len(audio_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_BYTES // (1024*1024)} MB.",
        )

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        waveform = load_and_validate_audio(audio_bytes)
        embedding = extract_embedding(waveform)

        voiceprint_store[user_id] = embedding
        save_voiceprints(voiceprint_store)

        duration = waveform.shape[1] / SAMPLE_RATE

        return EnrollResponse(
            user_id=user_id,
            status="enrolled",
            embedding_dim=len(embedding),
            audio_duration_seconds=round(duration, 2),
            message="Voiceprint stored. User can now verify.",
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Enrollment failed: {str(e)}"
        ) from e


@fast_api_app.post("/verify", response_model=VerifyResponse)
async def verify(user_id: str, audio: UploadFile = File(...)) -> VerifyResponse:
    """
    Verify speaker identity AND detect DeepFake in a single request.

    - **user_id**: The account to verify against
    - **audio**: Audio file to verify

    Returns:
        - Speaker verification result (similarity score)
        - DeepFake detection result (probability score)
        - Overall verdict (PASS only if both checks pass)
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")

    user_id = user_id.strip()

    if user_id not in voiceprint_store:
        raise HTTPException(
            status_code=404,
            detail=f"No voiceprint found for user '{user_id}'. Enroll first.",
        )

    audio_bytes = await audio.read()

    if len(audio_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_BYTES // (1024*1024)} MB.",
        )

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # ── Layer 1: Speaker Verification ─────────────────────────────────
        waveform = load_and_validate_audio(audio_bytes)
        new_embedding = extract_embedding(waveform)
        stored_embedding = voiceprint_store[user_id]

        similarity = cosine_similarity(new_embedding, stored_embedding)
        speaker_passed = similarity >= SIMILARITY_THRESHOLD
        confidence = get_confidence_band(similarity)

        speaker_result = {
            "verdict": "PASS" if speaker_passed else "FAIL",
            "similarity_score": round(similarity, 4),
            "threshold": SIMILARITY_THRESHOLD,
            "confidence": confidence,
        }

        # ── Layer 2: DeepFake Detection ────────────────────────────────
        temp_path = f"/tmp/verify_{uuid.uuid4().hex}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        try:
            ai_probability = DEEPFAKE_DETECTOR.predict(temp_path)
            is_real = ai_probability >= DEEPFAKE_THRESHOLD
            ai_result = {
                "verdict": "Real" if is_real else "FAKE",
                "ai_probability": round(ai_probability, 4),
                "threshold": DEEPFAKE_THRESHOLD,
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # ── Overall Verdict ───────────────────────────────────────────────
        overall_verdict = "PASS" if (speaker_passed and not is_real) else "FAIL"

        return VerifyResponse(
            user_id=user_id,
            speaker_verification=speaker_result,
            ai_detection=ai_result,
            overall_verdict=overall_verdict,
            message=(
                "Verification complete. Speaker confirmed and audio is authentic."
                if overall_verdict == "PASS"
                else "Verification failed. Either speaker mismatch or audio is AI-generated."
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Verification failed: {str(e)}"
        ) from e


@fast_api_app.delete("/unenroll/{user_id}", response_model=UnenrollResponse)
async def unenroll(user_id: str) -> UnenrollResponse:
    """Remove a user's voiceprint."""
    if user_id not in voiceprint_store:
        raise HTTPException(
            status_code=404, detail=f"No voiceprint found for '{user_id}'"
        )

    del voiceprint_store[user_id]
    save_voiceprints(voiceprint_store)

    return UnenrollResponse(user_id=user_id, status="removed")
