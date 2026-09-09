"""Run the CPU-only MusicEval prediction and temporal pooling analyses.

The encoder stage is deliberately separate from this script.  This runner
loads manifest-checked CLAP/MERT embeddings and the frozen prediction-head
bundles produced by :mod:`scripts.train_heads`, then writes auditable JSON
artifacts for the baseline, calibration, variance diagnostic, temperature
selection, and test-set aggregation analyses.  No encoder is instantiated and
no CUDA operation is requested here.

``--smoke`` uses a small, deterministic subset of each available split and a
short bootstrap/training schedule.  It is intended for CI and wiring checks;
it must not be reported as the study result.  ``--all`` is accepted explicitly
for the full suite (and is the default when no phase flag is supplied).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

# Scripts are runnable both from the repository root and as an absolute path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregation.pooling import (
    apply_pool,
    attention_pool,
    mean_pool,
    min_pool,
    soft_min,
    source_auto_pool,
    source_power_pool,
)
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, read_manifest
from src.metrics.bootstrap import bootstrap_clip_metric
from src.metrics.evaluate import (
    compute_regression_metrics,
    make_variance_subgroups,
    segment_dispersion,
)
from src.models.backbones import CLAPBaseline, CLAPMERT, MERTAudio, select_fusion_weight
from src.models.training import load_model_bundle, set_global_seed


BACKBONES = ("clap_baseline", "clap_mert", "mert_audio")
DIMENSIONS = ("MI", "TA")
SPLITS = ("train", "dev", "test")
DEFAULT_FUSION_GRID = tuple(round(index / 10.0, 1) for index in range(11))
DEFAULT_PUBLISHED_REFERENCES: dict[str, dict[str, dict[str, Any]]] = {
    # These values are retained as a comparison ledger only.  The local
    # protocol uses official splits, frozen encoders, and a different head
    # wiring from the cited implementations, so they are never treated as a
    # direct reproduction target.
    "clap_mert": {
        "MI": {
            "mse": 0.027,
            "source": "protocol note in experiment specification (FUSEMOS comparison)",
        },
        "TA": {
            "spearman_rho": 0.940,
            "source": "protocol note in experiment specification (FUSEMOS comparison)",
        },
    }
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="run every analysis phase")
    parser.add_argument("--smoke", action="store_true", help="run a small deterministic wiring check")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--clap-cache", type=Path, default=None)
    parser.add_argument("--mert-cache", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="maximum rows per split")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--pooling-epochs", type=int, default=None)
    return parser.parse_args()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    # The checked-in config lives in ``configs/``.  For a temporary test config
    # use its parent as the project root instead of assuming that layout.
    return resolved.parent.parent if resolved.parent.name.lower() == "configs" else resolved.parent


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert NumPy/PyTorch values to strict JSON values.

    JSON has no NaN representation.  Degenerate correlations are therefore
    emitted as ``null`` rather than relying on Python's non-standard ``NaN``
    token; this keeps result files consumable by strict parsers.
    """

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.generic,)):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        # External data/checkpoint paths are represented by a portable marker
        # and basename.  Hashes in the surrounding metadata preserve identity
        # without exposing machine-specific directories.
        return f"external/{path.name}"


def _git_revision(root: Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def _base_metadata(
    *,
    root: Path,
    config_path: Path,
    mode: str,
    seed: int,
    manifest_path: Path,
    clap_cache: Path,
    mert_cache: Path,
    model_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_mode": mode,
        "seed": int(seed),
        "device": "cpu",
        "cuda_used": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_revision": _git_revision(root),
        "config_sha256": _sha256(config_path),
        "inputs": {
            "manifest": _relative(manifest_path, root),
            "clap_cache": _relative(clap_cache, root),
            "mert_cache": _relative(mert_cache, root),
            "model_dir": _relative(model_dir, root),
        },
        "variants": {
            "ta_wiring": "concatenate audio and CLAP text embeddings",
            "clap_long_audio": "deterministic center crop declared by encoder adapter",
            "prediction_range_rule": "clip_epsilon for source/generalized power weights only",
            "source_pool_formulas": "source_auto_pool and source_power_pool weighted arithmetic",
            "variance_group_rule": "development median; test high group is strictly greater",
            "bootstrap_unit": "complete clip (all segment scores retained)",
        },
    }


def _feature(entry: Mapping[str, Any], *, label: str) -> np.ndarray:
    if not entry_is_valid(entry):
        raise RuntimeError(f"invalid {label} cache entry: {entry.get('clip_id')}")
    try:
        values = np.load(str(entry["path"]), allow_pickle=False)
        output = np.asarray(values, dtype=np.float32).copy()
        del values
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"unable to read {label} feature: {entry.get('path')}") from exc
    if output.size == 0 or not np.isfinite(output).all():
        raise RuntimeError(f"non-finite or empty {label} feature: {entry.get('clip_id')}")
    return output


def _entry(
    entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    key: tuple[str, str, str],
    *,
    label: str,
) -> Mapping[str, Any]:
    value = entries.get(key)
    if value is None:
        raise KeyError(f"missing {label} cache entry for {key[2]}")
    return value


def _complete_rows(
    rows: Sequence[Mapping[str, Any]],
    clap_entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    mert_entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    complete: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in rows:
        row = dict(source)
        clip_id = str(row["clip_id"])
        prompt_id = str(row["prompt_id"])
        required = (
            (clap_entries, ("clap", "audio_full", clip_id), "CLAP full audio"),
            (clap_entries, ("clap", "audio_seg", clip_id), "CLAP segment audio"),
            (clap_entries, ("clap", "text", prompt_id), "CLAP text"),
            (mert_entries, ("mert", "audio_full", clip_id), "MERT full audio"),
            (mert_entries, ("mert", "audio_seg", clip_id), "MERT segment audio"),
        )
        absent = [label for mapping, key, label in required if key not in mapping]
        if absent:
            missing.append(f"{clip_id}: {', '.join(absent)}")
            continue
        # Validate once up front.  This gives a useful missing/corrupt count
        # before model inference starts and prevents silent partial analyses.
        invalid = [
            label
            for mapping, key, label in required
            if not entry_is_valid(mapping[key])
        ]
        if invalid:
            missing.append(f"{clip_id}: invalid {', '.join(invalid)}")
            continue
        complete.append(row)
    return complete, missing


def _select_rows(
    all_rows: Sequence[Mapping[str, Any]],
    *,
    clap_entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    mert_entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    smoke: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {"requested": {}, "available": {}, "missing_examples": {}}
    cap = limit if limit is not None else (8 if smoke else None)
    for split in SPLITS:
        candidates = [dict(row) for row in all_rows if str(row.get("split")) == split]
        complete, missing = _complete_rows(candidates, clap_entries, mert_entries)
        complete.sort(key=lambda row: str(row["clip_id"]))
        if cap is not None:
            complete = complete[: max(0, int(cap))]
        report["requested"][split] = len(candidates)
        report["available"][split] = len(complete)
        report["missing_examples"][split] = missing[:5]
        if not smoke and missing:
            raise RuntimeError(
                f"{split} split has {len(missing)} missing/invalid cache rows; "
                f"first example: {missing[0]}"
            )
        selected.extend(complete)
    if not selected:
        raise RuntimeError("no rows have a complete verified CLAP/MERT/text cache")
    return selected, report


def _load_models(
    model_dir: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    hidden = int(config.get("training", {}).get("head_hidden_dim", 256))
    second_hidden = int(config.get("training", {}).get("head_second_hidden_dim", 128))
    constructors = {
        "clap_baseline": lambda: CLAPBaseline(hidden_dim=hidden, second_hidden_dim=second_hidden),
        "mert_audio": lambda: MERTAudio(hidden_dim=hidden, second_hidden_dim=second_hidden),
        "clap_mert": lambda: CLAPMERT(hidden_dim=hidden, second_hidden_dim=second_hidden),
    }
    models: dict[str, torch.nn.Module] = {}
    metadata: dict[str, dict[str, Any]] = {}
    filenames = {
        "clap_baseline": "clap_baseline.pt",
        "mert_audio": "mert_audio.pt",
        "clap_mert": "clap_mert.pt",
    }
    for name in BACKBONES:
        path = model_dir / filenames[name]
        if not path.is_file():
            raise FileNotFoundError(
                f"missing trained model bundle {path}; run scripts/train_heads.py first"
            )
        model = constructors[name]()
        bundle_metadata = load_model_bundle(model, path)
        model.to("cpu")
        model.eval()
        models[name] = model
        # Keep result files portable: an absolute local model path would leak
        # the author's machine path into a candidate public derived artifact.
        display_path = (
            _relative(path, project_root)
            if project_root is not None
            else path.name
        )
        metadata[name] = {
            "path": display_path,
            "sha256": _sha256(path),
            **bundle_metadata,
        }
    return models, metadata


def _forward_simple(
    model: torch.nn.Module,
    *,
    audio_full: np.ndarray,
    audio_segments: np.ndarray,
    text: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    full = torch.from_numpy(np.asarray(audio_full, dtype=np.float32).reshape(1, -1))
    segments = torch.from_numpy(np.asarray(audio_segments, dtype=np.float32))
    if segments.ndim == 1:
        segments = segments.reshape(1, -1)
    prompt = torch.from_numpy(np.asarray(text, dtype=np.float32).reshape(-1))
    with torch.inference_mode():
        if kind == "clap_baseline":
            full_out = model(full, prompt)
            segment_out = model(segments, prompt)
        elif kind == "mert_audio":
            full_out = model(full, prompt)
            segment_out = model(segments, prompt)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unsupported simple model kind: {kind}")
    return {
        "full": {"MI": float(full_out["mi"].reshape(-1)[0]), "TA": float(full_out["ta"].reshape(-1)[0])},
        "segments": {
            "MI": segment_out["mi"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
            "TA": segment_out["ta"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
        },
    }


def _forward_fused_branches(
    model: CLAPMERT,
    *,
    clap_full: np.ndarray,
    clap_segments: np.ndarray,
    mert_full: np.ndarray,
    mert_segments: np.ndarray,
    text: np.ndarray,
) -> dict[str, Any]:
    ca_full = torch.from_numpy(np.asarray(clap_full, dtype=np.float32).reshape(1, -1))
    ca_segments = torch.from_numpy(np.asarray(clap_segments, dtype=np.float32))
    ma_full = torch.from_numpy(np.asarray(mert_full, dtype=np.float32).reshape(1, -1))
    ma_segments = torch.from_numpy(np.asarray(mert_segments, dtype=np.float32))
    if ca_segments.ndim == 1:
        ca_segments = ca_segments.reshape(1, -1)
    if ma_segments.ndim == 1:
        ma_segments = ma_segments.reshape(1, -1)
    prompt = torch.from_numpy(np.asarray(text, dtype=np.float32).reshape(-1))
    with torch.inference_mode():
        full_out = model(ca_full, ma_full, prompt, alpha=0.5, beta=0.5)
        segment_out = model(ca_segments, ma_segments, prompt, alpha=0.5, beta=0.5)
    return {
        "full": {
            "MI": float(full_out["mi"].reshape(-1)[0]),
            "TA": float(full_out["ta"].reshape(-1)[0]),
        },
        "segments": {
            "MI": segment_out["mi"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
            "TA": segment_out["ta"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
        },
        "branches": {
            "full": {
                "clap": {
                    "MI": float(full_out["clap_mi"].reshape(-1)[0]),
                    "TA": float(full_out["clap_ta"].reshape(-1)[0]),
                },
                "mert": {
                    "MI": float(full_out["mert_mi"].reshape(-1)[0]),
                    "TA": float(full_out["mert_ta"].reshape(-1)[0]),
                },
            },
            "segments": {
                "clap": {
                    "MI": segment_out["clap_mi"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
                    "TA": segment_out["clap_ta"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
                },
                "mert": {
                    "MI": segment_out["mert_mi"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
                    "TA": segment_out["mert_ta"].detach().cpu().numpy().reshape(-1).astype(np.float64).tolist(),
                },
            },
        },
    }


def _predict_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    clap_entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    mert_entries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    models: Mapping[str, torch.nn.Module],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        clip_id = str(row["clip_id"])
        prompt_id = str(row["prompt_id"])
        clap_full = _feature(_entry(clap_entries, ("clap", "audio_full", clip_id), label="CLAP full"), label="CLAP full")
        clap_segments = _feature(_entry(clap_entries, ("clap", "audio_seg", clip_id), label="CLAP segments"), label="CLAP segments")
        clap_text = _feature(_entry(clap_entries, ("clap", "text", prompt_id), label="CLAP text"), label="CLAP text")
        mert_full = _feature(_entry(mert_entries, ("mert", "audio_full", clip_id), label="MERT full"), label="MERT full")
        mert_segments = _feature(_entry(mert_entries, ("mert", "audio_seg", clip_id), label="MERT segments"), label="MERT segments")
        clap_result = _forward_simple(
            models["clap_baseline"],
            audio_full=clap_full,
            audio_segments=clap_segments,
            text=clap_text,
            kind="clap_baseline",
        )
        mert_result = _forward_simple(
            models["mert_audio"],
            audio_full=mert_full,
            audio_segments=mert_segments,
            text=clap_text,
            kind="mert_audio",
        )
        fused_result = _forward_fused_branches(
            models["clap_mert"],
            clap_full=clap_full,
            clap_segments=clap_segments,
            mert_full=mert_full,
            mert_segments=mert_segments,
            text=clap_text,
        )
        output.append(
            {
                "clip_id": clip_id,
                "split": str(row["split"]),
                "prompt_id": prompt_id,
                "mi": float(row["mi"]),
                "ta": float(row["ta"]),
                "predictions": {
                    "clap_baseline": clap_result,
                    "mert_audio": mert_result,
                    "clap_mert": fused_result,
                },
            }
        )
    return output


def _apply_fusion_weights(rows: Sequence[dict[str, Any]], weights: Mapping[str, Mapping[str, float]]) -> None:
    for row in rows:
        fused = row["predictions"]["clap_mert"]
        branches = fused["branches"]
        for dimension, alpha_key in (("MI", "alpha"), ("TA", "beta")):
            weight = float(weights.get(alpha_key, 0.5))
            clap_full = float(branches["full"]["clap"][dimension])
            mert_full = float(branches["full"]["mert"][dimension])
            clap_segments = np.asarray(branches["segments"]["clap"][dimension], dtype=np.float64)
            mert_segments = np.asarray(branches["segments"]["mert"][dimension], dtype=np.float64)
            fused["full"][dimension] = weight * clap_full + (1.0 - weight) * mert_full
            fused["segments"][dimension] = (
                weight * clap_segments + (1.0 - weight) * mert_segments
            ).tolist()


def _select_fusion_weights(rows: Sequence[Mapping[str, Any]], grid: Sequence[float]) -> dict[str, Any]:
    dev = [row for row in rows if str(row["split"]) == "dev"]
    result: dict[str, Any] = {
        "method": "minimum development full-clip MSE",
        "grid": [float(value) for value in grid],
        "split": "dev",
        "weights": {"alpha": 0.5, "beta": 0.5},
        "search": {},
    }
    if not dev:
        result["status"] = "default_no_development_rows"
        return result
    for dimension, key in (("MI", "alpha"), ("TA", "beta")):
        first = np.asarray(
            [row["predictions"]["clap_mert"]["branches"]["full"]["clap"][dimension] for row in dev],
            dtype=np.float64,
        )
        second = np.asarray(
            [row["predictions"]["clap_mert"]["branches"]["full"]["mert"][dimension] for row in dev],
            dtype=np.float64,
        )
        target = np.asarray([row["mi" if dimension == "MI" else "ta"] for row in dev], dtype=np.float64)
        candidates: list[dict[str, float]] = []
        for weight in grid:
            prediction = float(weight) * first + (1.0 - float(weight)) * second
            candidates.append({"weight": float(weight), "mse": float(np.mean((prediction - target) ** 2))})
        chosen = select_fusion_weight(first, second, target, grid)
        result["weights"][key] = float(chosen)
        result["search"][dimension] = candidates
    result["status"] = "selected_on_complete_available_dev_rows"
    result["n_dev"] = len(dev)
    return result


def _fit_calibration(
    full: np.ndarray,
    segment_mean: np.ndarray,
    *,
    threshold: float,
    slope_lower: float,
    slope_upper: float,
) -> dict[str, Any]:
    full = np.asarray(full, dtype=np.float64).reshape(-1)
    segment_mean = np.asarray(segment_mean, dtype=np.float64).reshape(-1)
    if full.size == 0 or full.size != segment_mean.size:
        raise ValueError("calibration vectors must have equal non-empty length")
    if not np.isfinite(full).all() or not np.isfinite(segment_mean).all():
        raise ValueError("calibration vectors must be finite")
    difference = segment_mean - full
    mean_difference = float(np.mean(difference))
    variance = float(np.sum((segment_mean - np.mean(segment_mean)) ** 2))
    ols_slope = float(
        np.sum((segment_mean - np.mean(segment_mean)) * (full - np.mean(full))) / variance
    ) if variance > 1e-14 else 1.0
    ols_intercept = float(np.mean(full) - ols_slope * np.mean(segment_mean))
    reasons: list[str] = []
    if abs(mean_difference) > threshold:
        reasons.append("absolute_mean_difference")
    if ols_slope < slope_lower or ols_slope > slope_upper:
        reasons.append("slope_outside_range")
    triggered = bool(reasons)
    apply_slope = ols_slope if triggered else 1.0
    apply_intercept = ols_intercept if triggered else 0.0
    metrics = compute_regression_metrics(full, segment_mean)
    return {
        "mean_difference": mean_difference,
        "ols_slope": float(ols_slope),
        "ols_intercept": float(ols_intercept),
        "pearson_r": metrics["pearson_r"],
        "bland_altman_bias": metrics["bland_altman_bias"],
        "bland_altman_lower": metrics["bland_altman_lower"],
        "bland_altman_upper": metrics["bland_altman_upper"],
        "triggered": triggered,
        "trigger_reasons": reasons,
        "apply_slope": float(apply_slope),
        "apply_intercept": float(apply_intercept),
        "rule": {
            "mean_difference_threshold": float(threshold),
            "slope_lower": float(slope_lower),
            "slope_upper": float(slope_upper),
            "fit_target": "full_clip_prediction",
        },
    }


def _apply_calibration(rows: Sequence[dict[str, Any]], calibration: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    for row in rows:
        for backbone in BACKBONES:
            source = row["predictions"][backbone]["segments"]
            destination: dict[str, list[float]] = {}
            for dimension in DIMENSIONS:
                item = calibration[backbone][dimension]
                values = np.asarray(source[dimension], dtype=np.float64)
                transformed = float(item["apply_slope"]) * values + float(item["apply_intercept"])
                destination[dimension] = transformed.tolist()
            row["predictions"][backbone]["calibrated_segments"] = destination


def _pool(values: Sequence[float], method: str, parameter: float | None, config: Mapping[str, Any]) -> float:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("segment scores must be finite and non-empty")
    pooling_config = config.get("pooling", {})
    kwargs = {
        "positive_rule": str(pooling_config.get("positive_score_rule", "clip_epsilon")),
        "epsilon": float(pooling_config.get("positive_score_epsilon", 1e-6)),
    }
    if method == "mean":
        result = mean_pool(vector)
    elif method == "min":
        result = min_pool(vector)
    elif method == "soft_min":
        result = soft_min(vector, tau=1.0 if parameter is None else parameter)
    elif method == "attention":
        result = attention_pool(vector, temperature=1.0 if parameter is None else parameter)
    elif method == "auto_pool":
        result = source_auto_pool(vector, alpha=1.0 if parameter is None else parameter)
    elif method == "power_pool":
        result = source_power_pool(vector, n=1.0 if parameter is None else parameter, **kwargs)
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    scalar = float(np.asarray(result).reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f"non-finite result from {method}")
    return scalar


def _ragged_matrix(rows: Sequence[Mapping[str, Any]], backbone: str, dimension: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    values = [np.asarray(row["predictions"][backbone]["calibrated_segments"][dimension], dtype=np.float64) for row in rows]
    if not values:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=bool), []
    lengths = [int(value.size) for value in values]
    maximum = max(lengths)
    matrix = np.zeros((len(values), maximum), dtype=np.float64)
    mask = np.zeros((len(values), maximum), dtype=bool)
    for index, value in enumerate(values):
        matrix[index, : value.size] = value
        mask[index, : value.size] = True
    return matrix, mask, lengths


def _fit_pool_parameter(
    rows: Sequence[Mapping[str, Any]],
    *,
    backbone: str,
    dimension: str,
    method: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    """Fit one scalar score-pooling parameter on training clips.

    The protocol's attention baseline is represented as a learned score
    temperature.  This is intentionally labeled as a score-temperature
    variant because the public source does not specify a unique attention
    architecture for the evaluator head.
    """

    if not rows:
        return {
            "status": "not_fitted_no_training_rows",
            "parameter": 1.0,
            "method_variant": "score_temperature" if method == "attention" else "source_formula",
            "trace": [],
        }
    matrix, mask, _ = _ragged_matrix(rows, backbone, dimension)
    targets = np.asarray([row["mi" if dimension == "MI" else "ta"] for row in rows], dtype=np.float32)
    if epochs <= 0:
        return {"status": "not_fitted_zero_epochs", "parameter": 1.0, "trace": []}
    set_global_seed(seed)
    score = torch.as_tensor(matrix, dtype=torch.float32)
    valid = torch.as_tensor(mask, dtype=torch.bool)
    target = torch.as_tensor(targets, dtype=torch.float32)
    raw = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
    trace: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        if method == "auto_pool":
            alpha = torch.clamp(raw, -20.0, 20.0)
            logits = alpha * score
            logits = logits.masked_fill(~valid, -torch.inf)
            maximum = torch.amax(logits, dim=-1, keepdim=True)
            weights = torch.exp(logits - maximum).masked_fill(~valid, 0.0)
            prediction = torch.sum(weights * score, dim=-1) / torch.sum(weights, dim=-1)
            parameter = alpha
        elif method == "power_pool":
            exponent = torch.nn.functional.softplus(raw)
            positive = torch.clamp(score, min=float(epsilon))
            log_weights = exponent * torch.log(positive)
            log_weights = log_weights.masked_fill(~valid, -torch.inf)
            maximum = torch.amax(log_weights, dim=-1, keepdim=True)
            weights = torch.exp(log_weights - maximum).masked_fill(~valid, 0.0)
            prediction = torch.sum(weights * positive, dim=-1) / torch.sum(weights, dim=-1)
            parameter = exponent
        elif method == "attention":
            temperature = torch.nn.functional.softplus(raw)
            logits = temperature * score
            logits = logits.masked_fill(~valid, -torch.inf)
            maximum = torch.amax(logits, dim=-1, keepdim=True)
            weights = torch.exp(logits - maximum).masked_fill(~valid, 0.0)
            prediction = torch.sum(weights * score, dim=-1) / torch.sum(weights, dim=-1)
            parameter = temperature
        else:
            raise ValueError(f"unsupported learned pooling method: {method}")
        loss = torch.mean(torch.abs(prediction - target))
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach().cpu()))
    return {
        "status": "fitted_training_split",
        "parameter": float(parameter.detach().cpu()),
        "trace": trace,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "n_train": len(rows),
        "method_variant": "score_temperature" if method == "attention" else "source_formula",
        "target_dimension": dimension,
    }


def _metrics_with_bootstrap(
    clip_ids: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    segment_scores: Sequence[Sequence[float]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    metrics = compute_regression_metrics(targets, predictions)
    bootstrap: dict[str, Any] = {}
    for metric in ("mse", "spearman"):
        result = bootstrap_clip_metric(
            clip_ids,
            targets,
            predictions,
            metric=metric,
            n_resamples=int(n_resamples),
            seed=int(seed),
            segment_scores=segment_scores,
        )
        bootstrap[metric] = result.as_dict(include_samples=False)
    return {"metrics": metrics, "bootstrap": bootstrap}


def _subgroup_metrics(
    clip_ids: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if selected.size != len(clip_ids):
        raise ValueError("subgroup mask is not clip aligned")
    if not np.any(selected):
        return {"n": 0, "metrics": None}
    return {
        "n": int(np.sum(selected)),
        "clip_ids": [str(clip_ids[index]) for index in np.flatnonzero(selected)],
        "metrics": compute_regression_metrics(targets[selected], predictions[selected]),
    }


def _split_rows(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["split"]) == split]


def _write_phase_aliases(
    *,
    root: Path,
    smoke: bool,
    phase_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write convenient stable paths in addition to the run-specific files."""

    if smoke:
        return
    aliases = {
        "p0_baseline": root / "results" / "p0" / "baseline.json",
        "calibration": root / "results" / "p0" / "calibration.json",
        "diagnostic": root / "results" / "p0" / "diagnostic.json",
        "temperature_selection": root / "results" / "tables" / "temperature_selection.json",
        "main_aggregation": root / "results" / "tables" / "main_aggregation.json",
    }
    for name, path in aliases.items():
        payload = phase_payloads.get(name)
        if payload is not None:
            _write_json(path, payload)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic, strict-JSON line records for audit consumers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=True,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _portable_command(root: Path) -> list[str]:
    """Return an audit-friendly command without absolute machine paths."""

    result: list[str] = []
    for token in sys.argv:
        candidate = Path(token).expanduser() if ("\\" in token or "/" in token) else None
        if candidate is not None and candidate.is_absolute():
            result.append(_relative(candidate, root))
        else:
            result.append(token)
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    root = _project_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project = config.get("project", {})
    outputs = config.get("outputs", {})
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else _resolve_path(outputs.get("manifest_file", "cache/manifest.jsonl"), root=root)
    )
    clap_cache = (
        args.clap_cache.expanduser().resolve()
        if args.clap_cache is not None
        else _resolve_path(Path(project.get("cache_root", "cache")) / "clap" / "features.jsonl", root=root)
    )
    mert_cache = (
        args.mert_cache.expanduser().resolve()
        if args.mert_cache is not None
        else _resolve_path(Path(project.get("cache_root", "cache")) / "mert" / "features.jsonl", root=root)
    )
    model_dir = (
        args.model_dir.expanduser().resolve()
        if args.model_dir is not None
        else _resolve_path(outputs.get("model_dir", "results/models"), root=root)
    )
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_path(outputs.get("run_dir", "results/runs"), root=root)
    )
    smoke = bool(args.smoke)
    mode = "smoke" if smoke else "full"
    seed = int(args.seed if args.seed is not None else project.get("seed", 20260907))
    set_global_seed(seed)
    # Explicitly keep this phase on CPU even when CUDA is available for the
    # preceding extraction phase.
    torch.set_grad_enabled(True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not clap_cache.is_file() or not mert_cache.is_file():
        raise FileNotFoundError("both CLAP and MERT feature manifests are required")
    all_rows = read_manifest(manifest_path)
    clap_entries = load_existing_entries(clap_cache)
    mert_entries = load_existing_entries(mert_cache)
    rows, availability = _select_rows(
        all_rows,
        clap_entries=clap_entries,
        mert_entries=mert_entries,
        smoke=smoke,
        limit=args.limit,
    )
    models, model_metadata = _load_models(model_dir, config, project_root=root)
    predictions = _predict_rows(rows, clap_entries=clap_entries, mert_entries=mert_entries, models=models)

    temperature_config = config.get("temperature", {})
    fusion_grid = tuple(float(value) for value in config.get("fusion", {}).get("grid", DEFAULT_FUSION_GRID))
    fusion_selection = _select_fusion_weights(predictions, fusion_grid)
    _apply_fusion_weights(predictions, fusion_selection["weights"])

    base = _base_metadata(
        root=root,
        config_path=config_path,
        mode=mode,
        seed=seed,
        manifest_path=manifest_path,
        clap_cache=clap_cache,
        mert_cache=mert_cache,
        model_dir=model_dir,
    )
    base["availability"] = availability
    base["n_rows"] = len(predictions)
    base["model_bundles"] = model_metadata
    base["fusion_selection"] = fusion_selection

    phase_payloads: dict[str, dict[str, Any]] = {}
    p0_rows = _split_rows(predictions, "test")
    if not p0_rows:
        # A smoke run may only have dev rows; retaining an explicit empty test
        # section makes the limitation visible instead of silently relabeling.
        p0_rows = []
    p0: dict[str, Any] = {
        "metadata": {**base, "phase": "P0 baseline full-clip reproduction"},
        "split": "test",
        "published_comparison": {
            "status": "protocol_variant_not_direct_reproduction",
            "targets": DEFAULT_PUBLISHED_REFERENCES,
            "note": "Official splits, frozen encoders, and the declared TA concatenation path differ from cited implementations.",
        },
        "backbones": {},
    }
    for backbone in BACKBONES:
        result: dict[str, Any] = {"n": len(p0_rows), "dimensions": {}}
        for dimension in DIMENSIONS:
            target = np.asarray([row["mi" if dimension == "MI" else "ta"] for row in p0_rows], dtype=np.float64)
            prediction = np.asarray([row["predictions"][backbone]["full"][dimension] for row in p0_rows], dtype=np.float64)
            result["dimensions"][dimension] = (
                compute_regression_metrics(target, prediction) if len(target) else None
            )
        p0["backbones"][backbone] = result
    phase_payloads["p0_baseline"] = p0

    calibration_cfg = config.get("calibration", {})
    threshold = float(calibration_cfg.get("mean_difference_threshold", 0.1))
    slope_lower = float(calibration_cfg.get("slope_lower", 0.9))
    slope_upper = float(calibration_cfg.get("slope_upper", 1.1))
    dev_rows = _split_rows(predictions, "dev")
    calibration: dict[str, Any] = {
        "metadata": {**base, "phase": "Table I segment-to-full calibration"},
        "split": "dev",
        "n_dev": len(dev_rows),
        "rows": [],
    }
    calibration_map: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in BACKBONES}
    for backbone in BACKBONES:
        for dimension in DIMENSIONS:
            full = np.asarray([row["predictions"][backbone]["full"][dimension] for row in dev_rows], dtype=np.float64)
            segment_mean = np.asarray(
                [np.mean(row["predictions"][backbone]["segments"][dimension]) for row in dev_rows],
                dtype=np.float64,
            )
            if len(full):
                item = _fit_calibration(
                    full,
                    segment_mean,
                    threshold=threshold,
                    slope_lower=slope_lower,
                    slope_upper=slope_upper,
                )
            else:
                item = {
                    "mean_difference": None,
                    "ols_slope": None,
                    "ols_intercept": None,
                    "pearson_r": None,
                    "bland_altman_bias": None,
                    "bland_altman_lower": None,
                    "bland_altman_upper": None,
                    "triggered": False,
                    "trigger_reasons": ["no_development_rows"],
                    "apply_slope": 1.0,
                    "apply_intercept": 0.0,
                    "rule": {
                        "mean_difference_threshold": threshold,
                        "slope_lower": slope_lower,
                        "slope_upper": slope_upper,
                        "fit_target": "full_clip_prediction",
                    },
                }
            calibration_map[backbone][dimension] = item
            calibration["rows"].append({"backbone": backbone, "dimension": dimension, **item})
    _apply_calibration(predictions, calibration_map)
    phase_payloads["calibration"] = calibration

    # Variance groups are fixed once from raw CLAP-Baseline MI segment scores
    # on development and then applied unchanged to every test row.
    dev_variance = [row["predictions"]["clap_baseline"]["segments"]["MI"] for row in dev_rows]
    test_rows = _split_rows(predictions, "test")
    test_variance = [row["predictions"]["clap_baseline"]["segments"]["MI"] for row in test_rows]
    if dev_variance:
        dev_groups = make_variance_subgroups(dev_variance, threshold=None, ddof=0, high_inclusive=False)
        variance_threshold = float(dev_groups["threshold"])
    else:
        variance_threshold = 0.0
        dev_groups = {"dispersion": np.asarray([], dtype=np.float64), "high_mask": np.asarray([], dtype=bool), "low_mask": np.asarray([], dtype=bool), "n_high": 0, "n_low": 0, "threshold": variance_threshold}
    if test_variance:
        test_groups = make_variance_subgroups(test_variance, threshold=variance_threshold, ddof=0, high_inclusive=False)
    else:
        test_groups = {"dispersion": np.asarray([], dtype=np.float64), "high_mask": np.asarray([], dtype=bool), "low_mask": np.asarray([], dtype=bool), "n_high": 0, "n_low": 0, "threshold": variance_threshold}
    for index, row in enumerate(dev_rows):
        row["variance_sd"] = float(dev_groups["dispersion"][index])
        row["variance_group"] = "high" if bool(dev_groups["high_mask"][index]) else "low"
    for index, row in enumerate(test_rows):
        row["variance_sd"] = float(test_groups["dispersion"][index])
        row["variance_group"] = "high" if bool(test_groups["high_mask"][index]) else "low"
    # Write group fields back into the master list (the split helper returns
    # shallow copies for convenient filtering).
    by_id = {row["clip_id"]: row for row in predictions}
    for row in dev_rows + test_rows:
        by_id[row["clip_id"]]["variance_sd"] = row["variance_sd"]
        by_id[row["clip_id"]]["variance_group"] = row["variance_group"]

    # Preserve every selected clip's full/segment predictions (including the
    # frozen calibration map) in one machine-readable artifact.  Downstream
    # tables are derived from this file rather than recomputing hidden values.
    all_per_clip_path = output_root / mode / "all_per_clip_predictions.jsonl"
    _write_jsonl(all_per_clip_path, predictions)

    diagnostic: dict[str, Any] = {
        "metadata": {**base, "phase": "Table II variance diagnostic"},
        "variance_source": "clap_baseline.raw_segment_MI",
        "ddof": 0,
        "development_threshold": variance_threshold,
        "development": {"n": len(dev_rows), "n_high": int(dev_groups["n_high"]), "n_low": int(dev_groups["n_low"])},
        "test": {"n": len(test_rows), "n_high": int(test_groups["n_high"]), "n_low": int(test_groups["n_low"])},
        "groups": {},
    }
    for group_name, mask in (("high", test_groups["high_mask"]), ("low", test_groups["low_mask"])):
        group_payload: dict[str, Any] = {"n": int(np.sum(mask)), "backbones": {}}
        for backbone in ("clap_baseline",):
            dimensions: dict[str, Any] = {}
            for dimension in DIMENSIONS:
                target = np.asarray([row["mi" if dimension == "MI" else "ta"] for row in test_rows], dtype=np.float64)
                full = np.asarray([row["predictions"][backbone]["full"][dimension] for row in test_rows], dtype=np.float64)
                segment = np.asarray([np.mean(row["predictions"][backbone]["calibrated_segments"][dimension]) for row in test_rows], dtype=np.float64)
                dimensions[dimension] = {
                    "full_mean": compute_regression_metrics(target[mask], full[mask]) if np.any(mask) else None,
                    "segment_mean": compute_regression_metrics(target[mask], segment[mask]) if np.any(mask) else None,
                }
            group_payload["backbones"][backbone] = dimensions
        diagnostic["groups"][group_name] = group_payload
    high_full_mi = diagnostic["groups"]["high"]["backbones"]["clap_baseline"]["MI"]["full_mean"]
    high_segment_mi = diagnostic["groups"]["high"]["backbones"]["clap_baseline"]["MI"]["segment_mean"]
    if high_full_mi is None or high_segment_mi is None:
        diagnostic["causal_interpretation"] = "insufficient_test_rows"
    elif high_full_mi["signed_error"] > 0.1 and abs(high_segment_mi["signed_error"]) <= 0.1:
        diagnostic["causal_interpretation"] = "context_length_effect_consistent"
    elif high_full_mi["signed_error"] > 0.1 and high_segment_mi["signed_error"] > 0.1:
        diagnostic["causal_interpretation"] = "mean_pooling_effect_consistent"
    elif abs(high_full_mi["signed_error"]) <= 0.1 and abs(high_segment_mi["signed_error"]) <= 0.1:
        diagnostic["causal_interpretation"] = "plan_b_trigger_weak_signal"
    else:
        diagnostic["causal_interpretation"] = "inconclusive"
    phase_payloads["diagnostic"] = diagnostic

    # Development-only temperature search, after the calibration map and
    # fusion weights are frozen.
    temperature_grid = tuple(float(value) for value in temperature_config.get("grid", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]))
    temperature_selection: dict[str, Any] = {
        "metadata": {**base, "phase": "development temperature selection"},
        "split": "dev",
        "grid": list(temperature_grid),
        "selection_metric": str(temperature_config.get("selection_metric", "mse")),
        "backbones": {},
    }
    temperatures: dict[str, dict[str, float]] = {name: {} for name in BACKBONES}
    for backbone in BACKBONES:
        temperature_selection["backbones"][backbone] = {}
        for dimension in DIMENSIONS:
            target = np.asarray([row["mi" if dimension == "MI" else "ta"] for row in dev_rows], dtype=np.float64)
            segments = [row["predictions"][backbone]["calibrated_segments"][dimension] for row in dev_rows]
            search: list[dict[str, Any]] = []
            for tau in temperature_grid:
                pooled = np.asarray([_pool(values, "soft_min", tau, config) for values in segments], dtype=np.float64)
                search.append({"tau": float(tau), "mse": float(np.mean((pooled - target) ** 2)) if len(target) else None, "n": len(target)})
            finite_search = [item for item in search if item["mse"] is not None]
            if finite_search:
                chosen = min(finite_search, key=lambda item: (float(item["mse"]), temperature_grid.index(float(item["tau"]))))
                selected_tau = float(chosen["tau"])
                status = "selected_dev_mse"
            else:
                selected_tau = float(temperature_grid[0])
                status = "default_no_development_rows"
            temperatures[backbone][dimension] = selected_tau
            temperature_selection["backbones"][backbone][dimension] = {
                "selected_tau": selected_tau,
                "status": status,
                "search": search,
            }
    phase_payloads["temperature_selection"] = temperature_selection

    # Learn the three scalar pooling baselines on the training rows.  Their
    # parameters are fitted once and then reused unchanged on test.
    train_rows = _split_rows(predictions, "train")
    pooling_config = config.get("pooling", {})
    training_config = config.get("training", {})
    pooling_epochs = int(args.pooling_epochs if args.pooling_epochs is not None else training_config.get("pooling_epochs", 100))
    if smoke:
        pooling_epochs = min(pooling_epochs, 10)
    pooling_lr = float(training_config.get("pooling_learning_rate", 0.01))
    epsilon = float(pooling_config.get("positive_score_epsilon", 1e-6))
    learned: dict[str, dict[str, dict[str, Any]]] = {"clap_baseline": {}}
    for dimension in DIMENSIONS:
        learned["clap_baseline"][dimension] = {}
        for method, offset in (("attention", 101), ("auto_pool", 202), ("power_pool", 303)):
            learned["clap_baseline"][dimension][method] = _fit_pool_parameter(
                train_rows,
                backbone="clap_baseline",
                dimension=dimension,
                method=method,
                epochs=pooling_epochs,
                learning_rate=pooling_lr,
                seed=seed + offset + (0 if dimension == "MI" else 1),
                epsilon=epsilon,
            )

    methods_by_backbone = {
        "clap_baseline": ("mean", "soft_min", "min", "attention", "auto_pool", "power_pool"),
        "clap_mert": ("mean", "soft_min"),
        "mert_audio": ("mean", "soft_min"),
    }
    n_resamples = int(args.bootstrap_resamples if args.bootstrap_resamples is not None else config.get("statistics", {}).get("bootstrap_resamples", 2000))
    if smoke:
        n_resamples = min(n_resamples, 64)
    main_results: dict[str, Any] = {
        "metadata": {**base, "phase": "Table III-IV test aggregation"},
        "split": "test",
        "n_test": len(test_rows),
        "fusion_weights": fusion_selection["weights"],
        "calibration": calibration_map,
        "temperatures": temperatures,
        "pooling_parameters": learned,
        "bootstrap": {"n_resamples": n_resamples, "seed": seed, "unit": "clip"},
        "rows": [],
        "per_clip_file": None,
        "all_per_clip_file": _relative(all_per_clip_path, root),
        "all_per_clip_sha256": _sha256(all_per_clip_path),
    }
    clip_ids = [str(row["clip_id"]) for row in test_rows]
    per_clip: list[dict[str, Any]] = []
    for row in test_rows:
        per_clip.append(
            {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "prompt_id": row["prompt_id"],
                "mi": row["mi"],
                "ta": row["ta"],
                "variance_sd": row.get("variance_sd"),
                "variance_group": row.get("variance_group"),
                "backbones": {},
            }
        )
    for backbone, methods in methods_by_backbone.items():
        for method in methods:
            dimensions: dict[str, Any] = {}
            for dimension in DIMENSIONS:
                parameter: float | None = None
                if method == "soft_min":
                    parameter = temperatures[backbone][dimension]
                elif method in ("attention", "auto_pool", "power_pool"):
                    parameter = float(learned["clap_baseline"][dimension][method]["parameter"])
                predictions_vector = np.asarray(
                    [
                        _pool(
                            row["predictions"][backbone]["calibrated_segments"][dimension],
                            method,
                            parameter,
                            config,
                        )
                        for row in test_rows
                    ],
                    dtype=np.float64,
                )
                targets_vector = np.asarray([row["mi" if dimension == "MI" else "ta"] for row in test_rows], dtype=np.float64)
                segment_vectors = [row["predictions"][backbone]["calibrated_segments"][dimension] for row in test_rows]
                if len(targets_vector):
                    evaluated = _metrics_with_bootstrap(
                        clip_ids,
                        targets_vector,
                        predictions_vector,
                        segment_scores=segment_vectors,
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                    high_mask = np.asarray([row.get("variance_group") == "high" for row in test_rows], dtype=bool)
                    low_mask = ~high_mask
                    evaluated["subgroups"] = {
                        "high": _subgroup_metrics(clip_ids, targets_vector, predictions_vector, high_mask),
                        "low": _subgroup_metrics(clip_ids, targets_vector, predictions_vector, low_mask),
                    }
                else:
                    evaluated = {"metrics": None, "bootstrap": {}, "subgroups": {"high": {"n": 0, "metrics": None}, "low": {"n": 0, "metrics": None}}}
                dimensions[dimension] = {"parameter": parameter, **evaluated}
                for clip_index, clip in enumerate(per_clip):
                    clip.setdefault("backbones", {}).setdefault(backbone, {}).setdefault("aggregations", {}).setdefault(method, {})[dimension] = float(predictions_vector[clip_index])
            main_results["rows"].append({"backbone": backbone, "aggregation": method, "dimensions": dimensions})
    per_clip_path = output_root / mode / "per_clip_predictions.jsonl"
    _write_jsonl(per_clip_path, per_clip)
    main_results["per_clip_file"] = _relative(per_clip_path, root)
    main_results["per_clip_sha256"] = _sha256(per_clip_path)
    phase_payloads["main_aggregation"] = main_results

    run_dir = output_root / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    phase_files = {
        "p0_baseline": run_dir / "p0_baseline.json",
        "calibration": run_dir / "calibration.json",
        "diagnostic": run_dir / "diagnostic.json",
        "temperature_selection": run_dir / "temperature_selection.json",
        "main_aggregation": run_dir / "main_aggregation.json",
    }
    for name, path in phase_files.items():
        _write_json(path, phase_payloads[name])
    _write_phase_aliases(root=root, smoke=smoke, phase_payloads=phase_payloads)
    run_manifest = {
        **base,
        "phase": "run_manifest",
        "command": _portable_command(root),
        "phase_files": {name: _relative(path, root) for name, path in phase_files.items()},
        "per_clip_file": _relative(per_clip_path, root),
        "all_per_clip_file": _relative(all_per_clip_path, root),
        "rows_by_split": {split: len(_split_rows(predictions, split)) for split in SPLITS},
        "temperature_grid": list(temperature_grid),
        "bootstrap_resamples": n_resamples,
        "status": "complete",
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    return {
        "run_manifest": _relative(run_dir / "run_manifest.json", root),
        "phase_files": {name: _relative(path, root) for name, path in phase_files.items()},
        "rows_by_split": run_manifest["rows_by_split"],
        "mode": mode,
    }


def main() -> int:
    args = _args()
    # ``--all`` is intentionally explicit in the documented command, but the
    # runner has only one coherent suite, so an omitted flag remains useful.
    summary = _run(args)
    print(json.dumps(_jsonable(summary), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_fit_calibration",
    "_fit_pool_parameter",
    "_pool",
    "_select_fusion_weights",
    "_run",
    "main",
]
