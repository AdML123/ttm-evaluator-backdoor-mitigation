"""Frozen-encoder evaluator backbone contracts and score-level fusion."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from .heads import MLPHead


def _expand_text(audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    if text.shape[-1] != 512:
        raise ValueError(f"expected 512-dimensional CLAP text features, got {text.shape[-1]}")
    if audio.ndim == text.ndim + 1:
        text = text.unsqueeze(-2)
    if text.shape[:-1] != audio.shape[:-1]:
        try:
            text = text.expand(*audio.shape[:-1], text.shape[-1])
        except RuntimeError as exc:
            raise ValueError("audio and text batch shapes are incompatible") from exc
    return text


class CLAPBaseline(nn.Module):
    """CLAP audio MI head and CLAP audio+text TA head.

    The supplied MusicEval paper specifies independent three-layer heads but
    does not publish the TA input wiring.  The protocol-defined variant used
    here makes that wiring explicit: MI consumes audio, while TA consumes the
    concatenation of CLAP audio and prompt embeddings.
    """

    name = "clap_baseline"

    def __init__(self, *, hidden_dim: int = 256, second_hidden_dim: int = 128) -> None:
        super().__init__()
        self.mi_head = MLPHead(512, hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
        self.ta_head = MLPHead(1024, hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)

    def forward(self, clap_audio: torch.Tensor, clap_text: torch.Tensor) -> dict[str, torch.Tensor]:
        audio = torch.as_tensor(clap_audio, dtype=torch.float32)
        text = _expand_text(audio, torch.as_tensor(clap_text, dtype=torch.float32))
        return {
            "mi": self.mi_head(audio),
            "ta": self.ta_head(torch.cat([audio, text], dim=-1)),
        }


class MERTAudio(nn.Module):
    """MERT audio MI head and MERT-audio plus CLAP-text TA head."""

    name = "mert_audio"

    def __init__(self, *, hidden_dim: int = 256, second_hidden_dim: int = 128) -> None:
        super().__init__()
        self.mi_head = MLPHead(768, hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
        self.ta_head = MLPHead(1280, hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)

    def forward(self, mert_audio: torch.Tensor, clap_text: torch.Tensor) -> dict[str, torch.Tensor]:
        audio = torch.as_tensor(mert_audio, dtype=torch.float32)
        text = _expand_text(audio, torch.as_tensor(clap_text, dtype=torch.float32))
        return {
            "mi": self.mi_head(audio),
            "ta": self.ta_head(torch.cat([audio, text], dim=-1)),
        }


class CLAPMERT(nn.Module):
    """Independent CLAP/MERT heads followed by score-level MI/TA fusion."""

    name = "clap_mert"

    def __init__(self, *, hidden_dim: int = 256, second_hidden_dim: int = 128) -> None:
        super().__init__()
        self.clap = CLAPBaseline(hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
        self.mert = MERTAudio(hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)

    def forward(
        self,
        clap_audio: torch.Tensor,
        mert_audio: torch.Tensor,
        clap_text: torch.Tensor,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
            raise ValueError("fusion weights must lie in [0, 1]")
        clap_output = self.clap(clap_audio, clap_text)
        mert_output = self.mert(mert_audio, clap_text)
        return {
            "mi": alpha * clap_output["mi"] + (1.0 - alpha) * mert_output["mi"],
            "ta": beta * clap_output["ta"] + (1.0 - beta) * mert_output["ta"],
            "clap_mi": clap_output["mi"],
            "mert_mi": mert_output["mi"],
            "clap_ta": clap_output["ta"],
            "mert_ta": mert_output["ta"],
        }


def select_fusion_weight(
    first: torch.Tensor | np.ndarray,
    second: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    grid: Iterable[float],
) -> float:
    """Choose the first minimum-MSE weight, making ties deterministic."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or a.shape != y.shape or a.size == 0:
        raise ValueError("fusion arrays must have the same non-empty shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(y).all():
        raise ValueError("fusion arrays must be finite")
    candidates = [float(value) for value in grid]
    if not candidates or any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("fusion grid must contain values in [0, 1]")
    best_weight = candidates[0]
    best_error = float("inf")
    for weight in candidates:
        prediction = weight * a + (1.0 - weight) * b
        error = float(np.mean((prediction - y) ** 2))
        if error < best_error - 1e-12:
            best_error = error
            best_weight = weight
    return best_weight


__all__ = ["CLAPBaseline", "CLAPMERT", "MERTAudio", "select_fusion_weight"]
