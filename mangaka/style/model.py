"""Style transfer generator — encoder-decoder with AdaIN style injection.

Architecture:
    1. Content encoder extracts spatial features from the infilled page.
    2. A learned style query token cross-attends over the style bank to
       produce a single aggregated style vector.
    3. The decoder reconstructs the image, with AdaIN at every layer
       modulating features using the style vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive Instance Normalisation.

    Given content features and a style vector, computes per-channel
    affine parameters (gamma, beta) from the style vector and applies
    them to instance-normalised content features.
    """

    def __init__(self, num_features: int, style_dim: int = 512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) content features
            style: (B, style_dim) style vector

        Returns:
            (B, C, H, W) style-modulated features
        """
        h = self.norm(x)
        params = self.fc(style)  # (B, C*2)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * h + beta


class StyleAggregator(nn.Module):
    """Aggregate a style bank into a single style vector via cross-attention.

    A single learned query token attends over all bank embeddings (or a
    top-k retrieved subset) to produce one style vector per sample.
    """

    def __init__(self, style_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.style_dim = style_dim
        self.query = nn.Parameter(torch.randn(1, 1, style_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=style_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, bank: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bank: (B, N, style_dim) style embeddings from the bank

        Returns:
            (B, style_dim) aggregated style vector
        """
        B = bank.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        out, _ = self.attn(q, bank, bank)  # (B, 1, D)
        out = self.norm(out)
        return out.squeeze(1)  # (B, D)


class ContentEncoder(nn.Module):
    """Convolutional encoder that extracts multi-scale content features.

    Downsamples the input image through 4 strided conv blocks,
    producing features at 1/2, 1/4, 1/8, 1/16 resolution.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        bc = base_channels

        self.conv1 = self._block(in_channels, bc)          # → bc @ H/2
        self.conv2 = self._block(bc, bc * 2)                # → 2bc @ H/4
        self.conv3 = self._block(bc * 2, bc * 4)            # → 4bc @ H/8
        self.conv4 = self._block(bc * 4, bc * 8)            # → 8bc @ H/16

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            [f1, f2, f3, f4] feature maps at decreasing resolution
        """
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        return [f1, f2, f3, f4]


class StyleDecoder(nn.Module):
    """Decoder with AdaIN at every layer and skip connections from encoder.

    Mirrors the encoder structure, upsampling from 1/16 back to full
    resolution with AdaIN style injection at each scale.
    """

    def __init__(self, base_channels: int = 64, style_dim: int = 512, out_channels: int = 3):
        super().__init__()
        bc = base_channels

        # Upsample blocks: 8bc→4bc, 4bc→2bc, 2bc→bc, bc→out
        # Each receives skip-connected encoder features (doubled channels)
        self.up4 = self._up_block(bc * 8 * 2, bc * 4)  # cat(f4, d) → 4bc
        self.up3 = self._up_block(bc * 4 * 2, bc * 2)  # cat(f3, up4) → 2bc
        self.up2 = self._up_block(bc * 2 * 2, bc)       # cat(f2, up3) → bc
        self.up1 = self._up_block(bc * 2, bc)            # cat(f1, up2) → bc

        self.adain4 = AdaIN(bc * 4, style_dim)
        self.adain3 = AdaIN(bc * 2, style_dim)
        self.adain2 = AdaIN(bc, style_dim)
        self.adain1 = AdaIN(bc, style_dim)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(bc, bc, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc, out_channels, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self,
        encoder_feats: list[torch.Tensor],
        bottleneck: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_feats: [f1, f2, f3, f4] from ContentEncoder
            bottleneck: (B, 8*bc, H/16, W/16) bottleneck features
            style: (B, style_dim) aggregated style vector

        Returns:
            (B, 3, H, W) output image in [-1, 1]
        """
        f1, f2, f3, f4 = encoder_feats

        d = self.up4(torch.cat([f4, bottleneck], dim=1))
        d = self.adain4(d, style)

        d = self.up3(torch.cat([f3, d], dim=1))
        d = self.adain3(d, style)

        d = self.up2(torch.cat([f2, d], dim=1))
        d = self.adain2(d, style)

        d = self.up1(torch.cat([f1, d], dim=1))
        d = self.adain1(d, style)

        return self.to_rgb(d)


class StyleTransferGenerator(nn.Module):
    """Full style transfer generator.

    Forward pass:
        1. Encode content image → multi-scale features
        2. Aggregate style bank → single style vector (via cross-attention)
        3. Decode with AdaIN injection → styled output image

    The style bank is provided as a (B, N, style_dim) tensor of
    pre-computed embeddings (from StyleEncoder + StyleBank).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        style_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.encoder = ContentEncoder(in_channels, base_channels)
        self.aggregator = StyleAggregator(style_dim, num_heads)
        self.decoder = StyleDecoder(base_channels, style_dim)

        # Bottleneck: extra processing at the lowest resolution
        bc = base_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bc * 8, bc * 8, 3, padding=1),
            nn.InstanceNorm2d(bc * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 8, bc * 8, 3, padding=1),
            nn.InstanceNorm2d(bc * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self,
        content: torch.Tensor,
        style_bank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            content: (B, 3, H, W) content image (infilled page) in [-1, 1]
            style_bank: (B, N, style_dim) style embeddings from the bank

        Returns:
            (B, 3, H, W) styled output in [-1, 1]
        """
        # Normalise to [0, 1] for encoder then content features stay as-is
        feats = self.encoder(content)
        bn = self.bottleneck(feats[-1])
        style_vec = self.aggregator(style_bank)
        return self.decoder(feats, bn, style_vec)
