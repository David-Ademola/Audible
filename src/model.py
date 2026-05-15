import torch
from torch.nn import GELU, Dropout, LayerNorm, Linear, Module, Sequential, Tanh
from torch.nn.functional import interpolate, softmax
from torchvision.models.resnet import ResNet18_Weights, resnet18


class AttentiveStatisticsPooling(Module):
    """
    Attentive Statistics Pooling (ASP).

    Computes a weighted mean and weighted standard deviation over the time axis,
    where weights are predicted by a small attention network. Concatenating mean
    and std doubles the representation and gives the classifier information about
    both the centre and spread of features — important for detecting subtle
    synthesis artifacts that may appear only briefly.

    Output dim = 2 * input_dim.

    Reference: Okabe et al. (2018), popularised in anti-spoofing by ECAPA-TDNN.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attention = Sequential(
            Linear(input_dim, bottleneck_dim),
            Tanh(),
            Linear(bottleneck_dim, 1),  # scalar score per frame
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — frame-level features
            mask: (B, T) — boolean, True for valid frames
        Returns:
            (B, 2 * D)
        """
        # Attention weights
        scores = self.attention(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)

        mean = (weights * x).sum(dim=1)  # (B, D)

        # Numerically stable weighted variance
        var = (weights * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = (var + 1e-8).sqrt()  # (B, D)

        return torch.cat([mean, std], dim=-1)  # (B, 2D)


class ClassifierHead(Module):
    """
    Two-layer MLP classifier: Linear → GELU → Dropout → Linear → logits.
    Applied after pooling. Input dim is typically 2 * D due to ASP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 1,  # Binary classification
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = Sequential(
            LayerNorm(input_dim),
            Linear(input_dim, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepFakeAudioCNN(Module):
    """
    Deepfake audio detection model based on convolutional neural networks.
    Built around a ResNet18 backbone pretrained on ImageNet, adapted for mel spectrogram inputs.

    Args:
        num_classes: 1 for binary classification.
        dropout: Applied in classifier head.
    """

    def __init__(self, num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        # ── ResNet18 backbone ─────────────────────────────────────────────────
        # pylint: disable = E1121
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the adaptive average pooling and final FC
        self.feature_extractor = Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,  # output: (B, 512, H', T'')
        )

        resnet_dim = 512
        self.pooling = AttentiveStatisticsPooling(resnet_dim, bottleneck_dim=128)
        self.classifier = ClassifierHead(
            input_dim=2 * resnet_dim,
            hidden_dim=512,
            num_classes=num_classes,
            dropout=dropout,
        )

    def freeze_backbone(self):
        self.feature_extractor.requires_grad_(False)

    def unfreeze_backbone(self):
        self.feature_extractor.requires_grad_(True)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, 3, N_MELS, T) float32 — Channels are [mel, Δ, Δ-Δ].
            attention_mask: (B, T) — boolean, True for valid frames.
        Returns:
            Tensor with logits (B, 1) for binary classification.
        """
        x = self.feature_extractor(features)  # (B, 512, H', T'')

        # Collapse the frequency (mel) axis, keep time for attentive pooling
        x = x.mean(dim=2)  # (B, 512, T'')
        x = x.permute(0, 2, 1)  # (B, T'', 512)

        # Downsample the mask to match T''
        mask = attention_mask.unsqueeze(1).float()
        # Match the output size of the CNN exactly
        final_mask = (
            interpolate(mask, size=x.shape[1], mode="nearest").squeeze(1).bool()
        )

        embeddings = self.pooling(x, mask=final_mask)  # (B, 1024)
        logits = self.classifier(embeddings)  # (B, 1)

        return logits


if __name__ == "__main__":
    # Sanity check: can we do a forward pass?
    BATCH_SIZE = 1
    N_MELS = 80  # Number of mel bins after feature extraction
    TIME_FRAMES = 1001  # Number of time frames after feature extraction

    model = DeepFakeAudioCNN(num_classes=1, dropout=0.3).eval()
    dummy_input = torch.randn(BATCH_SIZE, 3, N_MELS, TIME_FRAMES)  # (B, C, F, T)
    dummy_mask = torch.ones(BATCH_SIZE, TIME_FRAMES, dtype=torch.bool)  # (B, T)

    with torch.no_grad():
        output = model(dummy_input, dummy_mask)

    print(f"Output shape: {output.shape}")  # Expected: (B, 1)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    input("Model sanity check passed. Press Enter to exit.")
