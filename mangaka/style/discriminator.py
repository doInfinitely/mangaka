"""PatchGAN discriminator for style transfer adversarial training."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator — classifies overlapping NxN patches as real/fake.

    Standard architecture from pix2pix: 4 strided conv layers producing a
    spatial map of real/fake predictions rather than a single scalar.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_channels
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(prev_ch, ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Second-to-last: stride 1
        prev_ch = ch
        ch = min(ch * 2, 512)
        layers += [
            nn.Conv2d(prev_ch, ch, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final: 1-channel prediction map
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image in [-1, 1]

        Returns:
            (B, 1, H', W') patch-level real/fake predictions
        """
        return self.model(x)
