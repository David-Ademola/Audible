import librosa
import torch
from torch.nn.functional import pad
from torchaudio.functional import compute_deltas
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from src.model import DeepFakeAudioCNN

# ─────────────────────────────────────────────────────────────────────────────
# Constants / default hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000  # All audio resampled to this
# Seconds. Shorter clips are zero-padded; longer are centre-cropped.
CLIP_DURATION = 10.0
N_FFT = 1024  # Fast Fourier Transform window size
HOP_LENGTH = 160  # 10 ms at 16 kHz
N_MELS = 80
F_MIN = 20.0
F_MAX = 8_000.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 160,000


class Predictor:
    """
    Loads the trained model and normalisation stats once, then runs
    inference on individual audio clips.

    Args:
        model_path: Path to the saved model state dict (.pt file).
        norm_stats_path: Path to the saved normalisation stats (.pt file).
        device: "cuda" or "cpu"
    """

    def __init__(
        self,
        model_path: str,
        norm_stats_path: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = DeepFakeAudioCNN(num_classes=1, dropout=0.3)

        try:
            print(f"Loading model from {model_path}")
            state = torch.load(model_path, map_location=self.device)

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            self.model.load_state_dict(state)
        except FileNotFoundError:
            print(f"Model file {model_path} not found.")

        self.model.to(self.device)
        self.model.eval()

        # ── Normalisation stats ────────────────────────────────────────────────
        stats = torch.load(norm_stats_path, map_location="cpu")
        self._norm_mean = stats["mean"].view(3, 1, 1)  # (3, 1, 1)
        self._norm_std = stats["std"].view(3, 1, 1)

        # ── Feature extractor ─────────────────────────────────────────────────
        self._mel_transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        self._amplitude_to_db = AmplitudeToDB(stype="power", top_db=80)

        print(f"Predictor ready — device: {self.device}")

    # ── Preprocessing ───────────

    def _load_waveform(self, path: str) -> tuple[torch.Tensor, int]:
        """Load, resample, mono-convert, normalise amplitude, pad/crop."""
        waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        # librosa returns numpy array (T,), convert to torch tensor (1, T)
        waveform = torch.from_numpy(waveform).unsqueeze(0).float()

        # Normalise amplitude to [-1, 1] (some .wav files are in int16 range)
        peak = waveform.abs().max()
        if peak > 1.0:
            waveform = waveform / peak

        # Pad (right) or centre-crop to clip_samples
        length = waveform.shape[-1]
        valid_length = min(length, CLIP_SAMPLES)

        if length < CLIP_SAMPLES:
            padding = CLIP_SAMPLES - length
            waveform = pad(waveform, (0, padding))
        elif length > CLIP_SAMPLES:
            start = (length - CLIP_SAMPLES) // 2
            waveform = waveform[:, start : start + CLIP_SAMPLES]

        return waveform, valid_length

    def _extract_features(
        self, waveform: torch.Tensor, valid_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mel, Δ and Δ-Δ extraction, normalisation, and mask generation."""
        mel = self._mel_transform(waveform)  # (1, N_MELS, T)
        mel = self._amplitude_to_db(mel)
        mel = mel.squeeze(0)  # (N_MELS, T)

        delta = compute_deltas(mel, win_length=9)
        delta2 = compute_deltas(delta, win_length=9)

        features = torch.stack([mel, delta, delta2], dim=0)  # (3, N_MELS, T)

        # Per-channel z-score — uses the training-set statistics
        features = (features - self._norm_mean) / (self._norm_std + 1e-8)

        # Attention mask — True for real frames, False for zero-padded frames
        num_frames = features.shape[-1]
        valid_frames = min(int(valid_length // HOP_LENGTH) + 1, num_frames)
        mask = torch.zeros(num_frames, dtype=torch.bool)
        mask[:valid_frames] = True

        return features, mask  # (3, N_MELS, T), (T,)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, audio_path: str) -> float:
        """
        Run inference on an audio file on disk.

        Args:
            audio_path: Path to a .wav (or any torchaudio-supported) audio file.

        Returns:
            float of probability score.
        """
        waveform, valid_length = self._load_waveform(audio_path)
        features, mask = self._extract_features(waveform, valid_length)

        # Add batch dimension
        features = features.unsqueeze(0).to(self.device)  # (1, 3, N_MELS, T)
        mask = mask.unsqueeze(0).to(self.device)  # (1, T)

        logit = self.model(features, mask).squeeze()  # scalar
        prob = torch.sigmoid(logit).item()

        return round(prob, 4)  # Round to 4 decimal places for cleaner output
