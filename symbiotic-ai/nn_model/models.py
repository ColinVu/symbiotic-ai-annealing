"""PyTorch models for per-frame state classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameMLP(nn.Module):
    """Per-frame linear classifier (interpretable feature weights)."""

    def __init__(self, input_dim: int, num_classes: int = 4, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) or (B, D)
        if x.dim() == 3:
            b, t, d = x.shape
            logits = self.net(x.reshape(b * t, d))
            return logits.reshape(b, t, -1)
        return self.net(x)


class TemporalConvNet(nn.Module):
    """Lightweight 1D TCN over feature sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        channels: int = 32,
        kernel_size: int = 5,
        num_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(
                nn.Conv1d(in_ch, channels, kernel_size, padding=padding, dilation=dilation)
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(channels))
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> logits (B, T, C)
        x = x.transpose(1, 2)  # (B, D, T)
        h = self.conv(x)
        logits = self.head(h)
        return logits.transpose(1, 2)


def build_model(
    model_type: str,
    input_dim: int,
    num_classes: int = 4,
    hidden: int = 0,
    tcn_channels: int = 32,
) -> nn.Module:
    if model_type == "mlp":
        return FrameMLP(input_dim, num_classes, hidden=hidden)
    if model_type == "tcn":
        return TemporalConvNet(input_dim, num_classes, channels=tcn_channels)
    raise ValueError(f"Unknown model_type: {model_type!r}")
