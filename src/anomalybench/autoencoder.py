"""A small fully-connected autoencoder scored by reconstruction error (PyTorch, CPU)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .detectors import Detector


class MLPAutoencoder(nn.Module):
    """``d -> hidden -> bottleneck -> hidden -> d`` with ReLU activations."""

    def __init__(self, d: int, hidden: int, bottleneck: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class Autoencoder(Detector):
    """Reconstruction-error detector trained on the full, contaminated data (no labels).

    The bottleneck is ``min(8, d // 2)`` (at least 1), always narrower than the input, so the
    network cannot learn the identity map. Training: Adam (lr 1e-3), MSE, batch 256, 10 epochs
    but at least ``min_steps`` optimiser steps so tiny datasets are not under-trained.
    The score is the per-row mean squared reconstruction error.
    """

    name = "Autoencoder"
    stochastic = True

    def __init__(
        self,
        hidden: int = 64,
        epochs: int = 10,
        min_steps: int = 300,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        super().__init__(seed)
        self.hidden = hidden
        self.epochs = epochs
        self.min_steps = min_steps
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X: np.ndarray) -> Autoencoder:
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        data = torch.as_tensor(np.asarray(X, dtype=np.float32))
        n, d = data.shape
        self.model_ = MLPAutoencoder(d, self.hidden, max(1, min(8, d // 2)))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        batches_per_epoch = math.ceil(n / self.batch_size)
        epochs = max(self.epochs, math.ceil(self.min_steps / batches_per_epoch))
        self.loss_history_: list[float] = []
        self.model_.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch = data[perm[start : start + self.batch_size]]
                loss = nn.functional.mse_loss(self.model_(batch), batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * batch.shape[0]
            self.loss_history_.append(total / n)
        self.model_.eval()
        return self

    @torch.no_grad()
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        data = torch.as_tensor(np.asarray(X, dtype=np.float32))
        out = np.empty(data.shape[0], dtype=np.float64)
        for start in range(0, data.shape[0], 8192):
            batch = data[start : start + 8192]
            out[start : start + 8192] = ((self.model_(batch) - batch) ** 2).mean(dim=1).numpy()
        return out
