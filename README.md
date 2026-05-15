# Audio Verification API — Speaker Verification + DeepFake Detection

A combined speaker verification and deepfake-audio detection service. The API performs two sequential checks:

- Layer 1: speaker verification (ECAPA-TDNN via SpeechBrain)
- Layer 2: deepfake / AI-generated audio detection (DeepFakeAudioCNN)

This repository contains the model, preprocessing, and a FastAPI server for inference and voiceprint management.

---

## 📋 Features

- **Speaker verification**: register and verify user voiceprints (ECAPA-TDNN)
- **Deepfake detection**: per-clip authenticity scoring using a ResNet-based CNN
- **Combined verdict**: `PASS` only if speaker matches and audio is authentic
- **REST API** using FastAPI with CORS, upload limits, and graceful error handling

---

## 🏗️ Architecture

### Model: `DeepFakeAudioCNN`

```
Audio File
    ↓
Resample to 16 kHz + normalize
    ↓
Extract Mel-Spectrogram (80 bins)
    ↓
Compute Δ (delta) and Δ-Δ (delta-delta)
    ↓
Stack into 3-channel feature tensor + z-score normalize
    ↓
ResNet18 backbone → Attentive Statistics Pooling
    ↓
Binary classifier (real vs. fake)
    ↓
Sigmoid → Probability [0, 1]
```

**Key components:**

- **ResNet18** — Pre-trained image classifier adapted for spectral features
- **Attentive Statistics Pooling** — Learns weighted mean and std over time; detects subtle artifacts
- **Attention Masking** — Ignores zero-padded frames from variable-length audio

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA 12.8 (optional, for GPU acceleration)
- ~2 GB free disk space (for models + data)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/David-Ademola/Audible.git
   cd Audible
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Option 1: Command-Line Inference

```python
from inference import Predictor

# Initialize
predictor = Predictor(
    model_path="models/DeepFakeAudioDetector.pt",
    norm_stats_path="data/norm_stats.pt"
)

# Predict
probability = predictor.predict("path/to/audio.wav")
print(f"Probability (authentic): {probability:.4f}")
```

**Output:** Float between 0 and 1

- `0.0` → very confident the audio is **fake**
- `1.0` → very confident the audio is **real**

### FastAPI Server

Start the API server (serves enrollment, verification, and health endpoints):

```bash
uvicorn api:fast_api_app --host 0.0.0.0 --port 8000
```

**Endpoints (summary):**

- `POST /enroll` — Register a user's voiceprint (fields: `user_id`, `audio`)
- `POST /verify` — Verify speaker identity and detect deepfake (fields: `user_id`, `audio`)
- `DELETE /unenroll/{user_id}` — Remove a stored voiceprint
- `GET /health` — Server and model health

**Examples:**

Enroll a user:

```bash
curl -X POST -F "user_id=alice" -F "audio=@alice_enroll.wav" http://localhost:8000/enroll
```

Verify a user and detect deepfake:

```bash
curl -X POST -F "user_id=alice" -F "audio=@alice_test.wav" http://localhost:8000/verify
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## 📁 Project Structure

```
Audible/
├── api.py                          # FastAPI inference server
├── inference.py                    # Predictor class & preprocessing
├── main.ipynb                      # Main training/evaluation notebook
├── download_data.ipynb             # Data acquisition script
├── create_csv.py                   # Dataset split utilities
├── order.py                        # Audio ordering/sorting utilities
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── train.csv                   # Training set metadata
│   ├── val.csv                     # Validation set metadata
│   ├── test.csv                    # Test set metadata
│   ├── norm_stats.pt               # Normalization statistics (mean/std)
│   ├── fake/                       # Fake audio samples
│   └── real/                       # Real audio samples
│
├── models/
│   ├── DeepFakeAudioDetector.pt    # Full precision model
│   └── DeepFakeAudioDetector_half.pt # Half-precision model (faster)
│
└── src/
    ├── model.py                    # CNN architecture & pooling layer
    └── utils.py                    # Helper functions
```

---

## 📊 Audio Processing Pipeline

### 1. **Waveform Loading**

- Loads audio via `librosa` (any torchaudio-supported format)
- Resamples to **16 kHz** (standard for speech/voice)
- Normalizes amplitude to [-1, 1]

### 2. **Padding / Cropping**

- Clips are standardized to **10 seconds** (160,000 samples)
- Shorter clips: zero-padded on the right
- Longer clips: center-cropped
- *Attention mask* tracks valid (non-padded) frames

### 3. **Feature Extraction**

- **Mel-Spectrogram**: 80 mel-frequency bins, 10 ms hop length
- **Frequency range**: 20 Hz – 8 kHz (speech range)
- **Delta features**: First-order time derivatives (Δ)
- **Delta-delta features**: Second-order derivatives (Δ-Δ)
- **Output shape**: (3, 80, 1001) for a 10-second clip

### 4. **Normalization**

- Per-channel z-score normalization using training-set statistics
- Applied independently: mean, Δ mean, Δ-Δ mean
- Improves model generalization across different domains

### 5. **Model Inference**

- ResNet18 processes the 3-channel spectrogram tensor
- Attention pooling weights frames based on diagnostic value
- Binary classifier outputs logit → sigmoid → probability

---

## 🎯 Model Configuration

Key preprocessing hyperparameters (in `inference.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SAMPLE_RATE` | 16,000 Hz | Standard for voice/speech |
| `CLIP_DURATION` | 10 seconds | Length of audio clips |
| `N_FFT` | 1024 | FFT window size for spectrogram |
| `HOP_LENGTH` | 160 | 10 ms hops at 16 kHz |
| `N_MELS` | 80 | Mel-frequency bins |
| `F_MIN` | 20 Hz | Minimum frequency |
| `F_MAX` | 8,000 Hz | Maximum frequency |

---

## 🔧 Configuration

### API Server Config (in `api.py`)

```python
MODEL_PATH = "models/DeepFakeAudioDetector.pt"        # Path to model weights
NORM_STATS_PATH = "data/norm_stats.pt"                # Normalization statistics
ALLOWED_ORIGINS = ["*"]                               # CORS origins (restrict in production)
MAX_FILE_BYTES = 50 * 1024 * 1024                     # Max upload: 50 MB
```

---

## 💾 Model Checkpoints

Two model versions are provided:

| Model | Precision | Use Case |
|-------|-----------|----------|
| `DeepFakeAudioDetector.pt` | Full (FP32) | High accuracy, slower inference |
| `DeepFakeAudioDetector_half.pt` | Half (FP16) | Faster inference, GPU-optimized |

To use the half-precision model, update `api.py`:

```python
MODEL_PATH = "models/DeepFakeAudioDetector_half.pt"
```

---

## 🧪 Testing

Run inference on test samples:

```python
from inference import Predictor

predictor = Predictor(
    model_path="models/DeepFakeAudioDetector.pt",
    norm_stats_path="data/norm_stats.pt"
)

# Test on real audio
real_prob = predictor.predict("data/real/sample_001.wav")
print(f"Real audio → Probability (authentic): {real_prob}")

# Test on fake audio
fake_prob = predictor.predict("data/fake/sample_001.wav")
print(f"Fake audio → Probability (authentic): {1 - fake_prob}")
```

**Expected results:**

- Real audio: probability close to **1**
- Fake audio: probability close to **0**

---

## 🚦 Performance & Deployment

### GPU Acceleration

- Models automatically use GPU if CUDA is available
- Check device in `Predictor` logs: "device: cuda" or "device: cpu"

### Inference Speed

- Typical latency: **100–500 ms** per 10-second clip (GPU: ~100 ms, CPU: ~500 ms)
- Throughput: ~10 clips/sec on GPU, ~2 clips/sec on CPU

### Deployment Tips

1. Use **half-precision model** for lower latency
2. Pre-warm the model with a dummy batch before serving
3. Set reasonable **upload size limits** (default: 50 MB)
4. Monitor GPU memory; consider batch inference for high throughput
5. In production, restrict **CORS origins** to your frontend domain

---

## 📚 Notebooks

- **`main.ipynb`** — Model training, evaluation, and performance metrics
- **`download_data.ipynb`** — Dataset acquisition and preprocessing

Run notebooks with Jupyter:

```bash
jupyter notebook main.ipynb
```

---

## 📦 Dependencies

See `requirements.txt`:

- **PyTorch** 2.11.0 (with CUDA 12.8 wheel)
- **librosa** — Audio loading and resampling
- **torchaudio** — Spectrogram & delta extraction
- **FastAPI** + **uvicorn** — API server
- **pandas**, **scikit-learn**, **matplotlib** — Data & visualization utilities

---

## 🐛 Troubleshooting

### `FileNotFoundError: Model file not found`

- Ensure `MODEL_PATH` and `NORM_STATS_PATH` are correct
- Check files exist: `ls models/` and `ls data/`

### Out of Memory (GPU)

- Use half-precision model: `DeepFakeAudioDetector_half.pt`
- Reduce batch size if doing batch inference
- Or switch to CPU: pass `device="cpu"` to `Predictor`

### Audio not loading

- Verify format is supported by `torchaudio` (WAV, MP3, FLAC, etc.)
- Try resampling with `ffmpeg`: `ffmpeg -i input.mp3 -ar 16000 output.wav`

### CORS errors in frontend

- Update `ALLOWED_ORIGINS` in `api.py` to include your frontend domain:

  ```python
  ALLOWED_ORIGINS = ["https://myapp.com", "http://localhost:3000"]
  ```

---

## 📝 License

[MIT License](LICENSE)

---

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

---

## 📧 Contact

For questions or issues, reach out to me on [Gmail](mailto:dakinwande350@gmail.com)

---

**Last Updated:** May 2026
