"""Style encoder — maps images to 512-d style embeddings via VGG19 features."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# VGG19 layers used for feature extraction (conv1_1 through conv5_1)
VGG_FEATURE_LAYERS = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv5_1": 28,
}

# ImageNet normalisation expected by VGG
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD = torch.tensor([0.229, 0.224, 0.225])


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG19 feature extractor for perceptual / style losses.

    Returns intermediate feature maps at the standard style-transfer taps
    (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1).  These are used both
    by the StyleEncoder projection head and by the loss functions.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg.features.children())

        # Split VGG into sub-networks at each tap point
        layer_indices = sorted(VGG_FEATURE_LAYERS.values())
        self.slices = nn.ModuleList()
        prev = 0
        for idx in layer_indices:
            self.slices.append(nn.Sequential(*features[prev : idx + 1]))
            prev = idx + 1

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", VGG_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std", VGG_STD.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: (B, 3, H, W) images in [0, 1]

        Returns:
            List of 5 feature maps at increasing depth.
        """
        x = (x - self.mean) / self.std
        feats = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


class StyleEncoder(nn.Module):
    """Encode an image into a 512-d style embedding.

    Architecture:
        VGG19 multi-scale features → channel-wise global average pooling
        → concatenate → MLP projection → 512-d style vector

    The VGG backbone is frozen; only the projection MLP trains.
    """

    def __init__(self, style_dim: int = 512):
        super().__init__()
        self.style_dim = style_dim
        self.vgg = VGGFeatureExtractor()

        # VGG19 feature channels at each tap: 64, 128, 256, 512, 512
        vgg_channels = [64, 128, 256, 512, 512]
        total_channels = sum(vgg_channels)  # 1472

        self.projection = nn.Sequential(
            nn.Linear(total_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to style embeddings.

        Args:
            x: (B, 3, H, W) images in [0, 1]

        Returns:
            (B, style_dim) style embeddings
        """
        feats = self.vgg(x)
        # Global average pool each feature map → (B, C_i)
        pooled = [f.mean(dim=(2, 3)) for f in feats]
        concatenated = torch.cat(pooled, dim=1)  # (B, 1472)
        return self.projection(concatenated)

    @torch.no_grad()
    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without gradients (for building the style bank)."""
        self.eval()
        return self.forward(x)
