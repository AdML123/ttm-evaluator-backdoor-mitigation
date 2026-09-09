"""Deterministic CPU training utilities for prediction heads."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_head(
    model: nn.Module,
    features: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    seed: int = 20260907,
) -> dict[str, list[float]]:
    """Fit a head with L1 loss on CPU and return per-epoch losses."""

    if epochs <= 0 or learning_rate <= 0 or batch_size <= 0:
        raise ValueError("epochs, learning_rate, and batch_size must be positive")
    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    y = torch.as_tensor(targets, dtype=torch.float32, device="cpu").reshape(-1)
    if x.ndim < 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("features and targets must have matching non-empty rows")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    model.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).reshape(-1)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            rows = batch_x.shape[0]
            total += float(loss.detach()) * rows
            count += rows
        history.append(total / count)
    return {"loss": history}


def predict_head(model: nn.Module, features: torch.Tensor | np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        values = model(torch.as_tensor(features, dtype=torch.float32, device="cpu"))
    return values.detach().cpu().numpy().astype(np.float32, copy=False)


def split_target_hash(rows: Iterable[tuple[str, float, float] | Mapping[str, Any]]) -> str:
    """Hash clip IDs and target values independently of input row order."""

    canonical: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            clip_id = str(row["clip_id"])
            mi = float(row["mi"])
            ta = float(row["ta"])
        else:
            clip_id, mi, ta = row
            clip_id = str(clip_id)
            mi = float(mi)
            ta = float(ta)
        canonical.append({"clip_id": clip_id, "mi": mi, "ta": ta})
    encoded = json.dumps(sorted(canonical, key=lambda item: item["clip_id"]), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_model_bundle(model: nn.Module, path: str | Path, metadata: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "metadata": dict(metadata)},
        destination,
    )
    return destination


def load_model_bundle(model: nn.Module, path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return dict(payload.get("metadata", {}))
