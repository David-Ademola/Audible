import os
import warnings

import librosa
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import pad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import compute_deltas
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm

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

# pylint: disable=W0718


class MelDeltaExtractor:
    """
    Stateless feature extractor that converts a raw waveform
    to a 3-channel (mel, Δ, ΔΔ) log-mel tensor.

    Args:
        sample_rate:  Expected sample rate of input waveforms.
        n_fft:        FFT window length.
        hop_length:   STFT hop size in samples.
        n_mels:       Number of mel filterbanks.
        f_min/f_max:  Mel filter frequency bounds.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
    ):
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        # Convert from power spectrogram scale decibel scale
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=80)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) mono waveform tensor at the configured sample rate.
        Returns:
            (3, n_mels, time_frames) float32 tensor: [mel, delta, delta-delta].
        """
        mel = self.mel_transform(waveform)  # (1, n_mels, T)
        mel = self.amplitude_to_db(mel)  # log scale -> dB
        mel = mel.squeeze(0)  # (n_mels, T)

        # Compute deltas (first and second order differences along time axis)
        delta = compute_deltas(mel, win_length=9)  # (n_mels, T)
        delta2 = compute_deltas(delta, win_length=9)  # (n_mels, T)

        # Stack to a 3-channel "image" tensor for CNN input
        return torch.stack([mel, delta, delta2], dim=0)  # (3, n_mels, T)


class NormalisationStats:
    """
    Holds per-channel (mel, delta, delta-delta) mean and std computed over
    the training set. Used to normalise features to zero mean, unit variance.

    Per-channel normalisation is preferred over global normalisation because
    the three channels live on very different scales (mel ≈ dB values,
    deltas ≈ much smaller numbers), and per-channel z-scoring keeps them
    all in the same range for the CNN.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # mean, std: (3,) — one value per channel
        self.mean = mean.view(3, 1, 1)  # (3, 1, 1) for broadcast over (n_mels, T)
        self.std = std.view(3, 1, 1)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """x: (3, n_mels, T) → normalised (3, n_mels, T)."""
        return (x - self.mean) / (self.std + 1e-8)

    def save(self, path: str):
        torch.save({"mean": self.mean.squeeze(), "std": self.std.squeeze()}, path)

    @classmethod
    def load(cls, path: str) -> "NormalisationStats":
        d = torch.load(path, map_location="cpu")
        return cls(d["mean"], d["std"])

    @classmethod
    def compute_from_dataset(
        cls,
        csv_path: str,
        extractor: MelDeltaExtractor,
        clip_samples: int,
        num_workers: int = 0,
    ) -> "NormalisationStats":
        """
        Compute mean and std over the training set in a single pass.

        Args:
            csv_path: Path to the training CSV.
            extractor: MelDeltaExtractor
            clip_samples: Number of waveform samples per clip (sample_rate * duration).
            num_workers: Passed to a temporary DataLoader.

        Returns:
            NormalisationStats ready to use.
        """
        print("Computing normalisation statistics over training set...")

        # Temporary dataset that returns raw features, no normalisation
        temp_dataset = DeepFakeAudioDataset(
            csv_path=csv_path,
            extractor=extractor,
            clip_samples=clip_samples,
            norm_stats=None,  # no normalisation yet
            augment=False,
        )

        loader = DataLoader(
            temp_dataset, batch_size=64, num_workers=num_workers, shuffle=False
        )

        # Online mean / variance in a single pass (Welford-style via accumulation)
        channel_sum, channel_sum_sq = torch.zeros(3), torch.zeros(3)
        n_frames = 0

        for features, _, _ in tqdm(loader, desc="Stats"):
            # features: (B, 3, n_mels, T)
            _, num_channels, _, _ = features.shape
            flat = features.permute(1, 0, 2, 3).reshape(
                num_channels, -1
            )  # (3, B * num_mels * T)
            channel_sum += flat.sum(dim=1)
            channel_sum_sq += (flat**2).sum(dim=1)
            n_frames += flat.shape[1]

        mean = channel_sum / n_frames
        std = ((channel_sum_sq / n_frames) - mean**2).sqrt()
        print(f"  mean: {mean.tolist()}")
        print(f"  std:  {std.tolist()}")
        return cls(mean, std)


class WaveformAugmenter:
    """
    Waveform-level augmentations applied on raw audio before feature extraction.

    Augmentations included:
      - Additive Gaussian noise (signal-to-noise ratio (SNR) 15-40 dB)
      - Random gain scaling (±6 dB)
      - Random polarity inversion (sign flip)
    """

    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_min_snr_db: float = 15.0,
        noise_max_snr_db: float = 40.0,
        gain_prob: float = 0.5,
        gain_min_db: float = -6.0,
        gain_max_db: float = 6.0,
        polarity_prob: float = 0.5,
    ):
        self.noise_prob = noise_prob
        self.noise_min_snr = noise_min_snr_db
        self.noise_max_snr = noise_max_snr_db
        self.gain_prob = gain_prob
        self.gain_min_db = gain_min_db
        self.gain_max_db = gain_max_db
        self.polarity_prob = polarity_prob

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (1, T) → (1, T) augmented."""
        # Additive Gaussian noise at a random SNR
        if torch.rand(1).item() < self.noise_prob:
            snr_db = (
                torch.empty(1).uniform_(self.noise_min_snr, self.noise_max_snr).item()
            )
            signal_power = waveform.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(waveform) * noise_power.sqrt()
            waveform = waveform + noise

        # Random gain
        if torch.rand(1).item() < self.gain_prob:
            gain_db = torch.empty(1).uniform_(self.gain_min_db, self.gain_max_db).item()
            gain = 10 ** (gain_db / 20)
            waveform = waveform * gain

        # Polarity inversion (phase flip)
        if torch.rand(1).item() < self.polarity_prob:
            waveform = -waveform

        # Clip to prevent overflow after gain
        return waveform.clamp_(-1.0, 1.0)


class SpecAugment:
    """
    Applies time and frequency masking to the (3, n_mels, T) feature tensor.
    Applied uniformly across all 3 channels so the masking is consistent.

    Reference: Park et al. (2019) "SpecAugment: A Simple Data Augmentation
    Method for Automatic Speech Recognition."
    """

    def __init__(
        self,
        freq_mask_param: int = 15,  # max frequency bins to mask
        time_mask_param: int = 30,  # max time frames to mask
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (3, n_mels, T) → (3, n_mels, T)."""
        _, n_mels, duration = x.shape

        # Frequency masking (mask horizontal bands)
        for _ in range(self.n_freq_masks):
            f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            f0 = int(torch.randint(0, max(1, n_mels - f), (1,)).item())
            x[:, f0 : f0 + f, :] = 0.0

        # Time masking (mask vertical stripes)
        for _ in range(self.n_time_masks):
            t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            t0 = int(torch.randint(0, max(1, duration - t), (1,)).item())
            x[:, :, t0 : t0 + t] = 0.0

        return x


class DeepFakeAudioDataset(Dataset):
    """
    PyTorch Dataset for binary deepfake audio classification.

    Reads a CSV file with columns:
        file_path — str, path to .wav file
        label— int, 1 = real, 0 = fake

    Returns:
        features: (3, n_mels, time_frames) float32 tensor
        label:    scalar int64 tensor

    Args:
        csv_path:      Path to the CSV file.
        extractor:     MelDeltaExtractor instance.
        clip_samples:  Fixed length in samples (sample_rate * clip_duration).
        norm_stats:    NormalisationStats for z-scoring. Pass None to skip.
        augment:       If True, apply WaveformAugmenter + SpecAugment.
        sample_rate:   Target sample rate for resampling.
        spec_augment:  SpecAugment config. Only used when augment=True.
        wave_augment:  WaveformAugmenter config. Only used when augment=True.
    """

    def __init__(
        self,
        csv_path: str,
        extractor: MelDeltaExtractor,
        clip_samples: int,
        norm_stats: NormalisationStats | None = None,
        augment: bool = False,
        sample_rate: int = SAMPLE_RATE,
        spec_augment: SpecAugment | None = None,
        wave_augment: WaveformAugmenter | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.extractor = extractor
        self.clip_samples = clip_samples
        self.norm_stats = norm_stats
        self.augment = augment
        self.sample_rate = sample_rate

        # Validate required columns
        required = {"file_path", "label"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        self.df = self.df.reset_index(drop=True)

        # Augmenters (only used when self.augment=True)
        self.wave_aug = wave_augment or WaveformAugmenter()
        self.spec_aug = spec_augment or SpecAugment()

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_preprocess_waveform(self, path: str) -> tuple[torch.Tensor, int]:
        """Load a .wav file, resample if necessary, convert to mono, and pad/crop."""
        try:
            waveform, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            # librosa returns numpy array (T,), convert to torch tensor (1, T)
            waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}. Returning silence.")
            return torch.zeros(1, self.clip_samples), 0

        # Normalise amplitude to [-1, 1] (some .wav files are in int16 range)
        peak = waveform.abs().max()
        if peak > 1.0:
            waveform = waveform / peak

        # Pad (right) or centre-crop to clip_samples
        length = waveform.shape[-1]
        valid_length = min(length, self.clip_samples)

        if length < self.clip_samples:
            padding = self.clip_samples - length
            waveform = pad(waveform, (0, padding))
        elif length > self.clip_samples:
            start = (length - self.clip_samples) // 2
            waveform = waveform[:, start : start + self.clip_samples]

        return waveform, valid_length

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        path = str(row["file_path"])
        label = int(row["label"])

        waveform, valid_length = self._load_and_preprocess_waveform(path)

        # Waveform augmentation (before feature extraction)
        if self.augment:
            waveform = self.wave_aug(waveform)

        # Extract 3-channel mel feature
        features = self.extractor(waveform)  # (3, n_mels, T)

        # Per-channel z-score normalisation
        if self.norm_stats is not None:
            features = self.norm_stats.normalise(features)

        # SpecAugment (after normalisation, training only)
        if self.augment:
            features = self.spec_aug(features)

        # Generate the mask based on the hop length
        # T_frames = samples // hop_length + 1
        num_frames = features.shape[-1]
        valid_frames = int(valid_length // self.extractor.mel_transform.hop_length) + 1
        valid_frames = min(valid_frames, num_frames)

        mask = torch.zeros(num_frames, dtype=torch.bool)
        mask[:valid_frames] = True

        return features, torch.tensor(label, dtype=torch.long), mask


def train(
    model: DeepFakeAudioCNN,
    optimizer: Optimizer,
    criterion: BCEWithLogitsLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    epochs: int = 50,
    patience: int = 5,
    save_path: str = "best_model.pt",
    lr_decay: float = 0.1,
    lr_patience: int = 3,
    min_lr: float = 1e-6,
) -> dict[str, list[float]]:
    """
    Train the given model using the specified optimizer and loss function.

    Args:
        model (Module): The neural network model to be trained.
        optimizer (Optimizer): The optimizer for updating model parameters.
        criterion (BCEWithLogitsLoss): The loss function to compute the training loss.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        save_path (str): Path to save the best model checkpoint.
        lr_decay (float): Factor to decay learning rate when plateauing.
        lr_patience (int): Number of epochs with no improvement before reducing LR.
        min_lr (float): Minimum learning rate.

    Returns
    -------
    dict[str, list]
        A dictionary containing:
        - ``train_losses`` : list of per-epoch training losses
        - ``val_losses``   : list of per-epoch validation losses
        - ``precision``    : list of validation precision values
        - ``recall``       : list of validation recall values
        - ``f1_scores``    : list of validation F1-scores
    """
    model.to(device)
    criterion.to(device)

    train_losses, val_losses = [], []
    precision_list, recall_list, f1_list = [], [], []

    best_performance = float("inf")
    early_stop_counter = 0

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_decay, patience=lr_patience, min_lr=min_lr
    )

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", save_path)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # --- TRAINING LOOP ---
        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"
        ) as pbar:
            for features, labels, masks in train_loader:
                features, labels, masks = (
                    features.to(device),
                    labels.to(device),
                    masks.to(device),
                )

                optimizer.zero_grad()
                outputs = model(features, masks).squeeze(1)  # (B,)

                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * features.size(0)

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        train_loss /= len(train_loader.dataset)  # type: ignore
        train_losses.append(train_loss)

        # --- VALIDATION LOOP ---
        val_loss, precision, recall, f1 = validate(model, criterion, val_loader, device)

        val_losses.append(val_loss)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}"
        )

        # Check for improvement
        if val_loss < best_performance:
            best_performance = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> New best model saved to {save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

            # Early stopping check
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "precision": precision_list,
        "recall": recall_list,
        "f1_scores": f1_list,
    }


@torch.no_grad()
def validate(
    model: DeepFakeAudioCNN,
    criterion: BCEWithLogitsLoss,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, float, float]:
    """
    Validate the given model using the specified loss function.

    Args:
        model (Module): The neural network model to be validated.
        criterion (BCEWithLogitsLoss): The loss function to compute the validation loss.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the validation on (CPU or GPU).

    Returns
    -------
    tuple
        A tuple containing:
        - ``val_loss`` : float, average validation loss
        - ``precision`` : float, precision score
        - ``recall`` : float, recall score
        - ``f1_score`` : float, F1-score
    """
    model.eval()
    model.to(device)

    val_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels, masks in val_loader:
        features, labels, masks = (
            features.to(device),
            labels.to(device),
            masks.to(device),
        )

        outputs = model(features, masks).squeeze(1)  # (B,)
        loss = criterion(outputs, labels.float())

        val_loss += loss.item() * features.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)  # type: ignore

    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return val_loss, precision, recall, f1  # type: ignore


@torch.no_grad()
def test(
    model: DeepFakeAudioCNN,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, float, float, float]:
    """
    Test the trained model on the test dataset.

    Args:
        model (Module): The trained neural network model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the testing on (CPU or GPU).

    Returns
    -------
    tuple
        A tuple containing:
        - ``precision`` : float, precision score
        - ``recall`` : float, recall score
        - ``f1_score`` : float, F1-score
        - ``all_labels`` : list, true labels
        - ``all_preds`` : list, predicted labels
    """
    model.eval().half()  # Use half precision for faster inference if supported
    model.to(device)

    all_preds = []
    all_labels = []

    for features, labels, masks in test_loader:
        features, labels, masks = (
            features.to(device).half(),
            labels.to(device).half(),
            masks.to(device).half(),
        )

        outputs = model(features, masks).squeeze(1)  # (B,)

        preds = (torch.sigmoid(outputs) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return precision, recall, f1, all_labels, all_preds  # type: ignore
