"""Contracts for the deterministic CPU experiment runner helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from src.features.cache import atomic_save_npy, cache_entry, write_manifest_entry
from src.models.backbones import CLAPBaseline, CLAPMERT, MERTAudio
from src.models.training import save_model_bundle
from scripts.run_experiments import (
    _fit_calibration,
    _fit_pool_parameter,
    _pool,
    _run,
    _select_fusion_weights,
)


def _row(clip_id: str, split: str, mi: float, ta: float, clap: list[float], mert: list[float]) -> dict:
    return {
        "clip_id": clip_id,
        "split": split,
        "mi": mi,
        "ta": ta,
        "predictions": {
            "clap_mert": {
                "branches": {
                    "full": {
                        "clap": {"MI": clap[0], "TA": clap[1]},
                        "mert": {"MI": mert[0], "TA": mert[1]},
                    }
                }
            }
        },
    }


def test_calibration_trigger_and_identity_policy_are_deterministic():
    # Segment means are exactly one point above full predictions, so the
    # predeclared 0.1 trigger must fit an affine correction.
    triggered = _fit_calibration(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        threshold=0.1,
        slope_lower=0.9,
        slope_upper=1.1,
    )
    assert triggered["triggered"]
    assert "absolute_mean_difference" in triggered["trigger_reasons"]
    np.testing.assert_allclose(
        triggered["apply_slope"] * np.array([2.0, 3.0, 4.0]) + triggered["apply_intercept"],
        np.array([1.0, 2.0, 3.0]),
        atol=1e-7,
    )

    identity = _fit_calibration(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.01, 2.01, 3.01]),
        threshold=0.1,
        slope_lower=0.9,
        slope_upper=1.1,
    )
    assert not identity["triggered"]
    assert identity["apply_slope"] == pytest.approx(1.0)
    assert identity["apply_intercept"] == pytest.approx(0.0)


def test_fusion_weight_search_uses_first_minimum_on_ties():
    rows = [
        _row("a", "dev", 1.0, 1.0, [1.0, 1.0], [1.0, 1.0]),
        _row("b", "dev", 2.0, 2.0, [2.0, 2.0], [2.0, 2.0]),
    ]
    selected = _select_fusion_weights(rows, [0.0, 0.5, 1.0])
    assert selected["weights"] == {"alpha": 0.0, "beta": 0.0}
    assert selected["status"] == "selected_on_complete_available_dev_rows"


def test_pool_helper_applies_source_formula_and_power_sign_policy():
    scores = np.array([1.0, 2.0, 3.0])
    expected = np.sum(scores**2) / np.sum(scores)
    assert _pool(scores, "power_pool", 1.0, {"pooling": {}}) == pytest.approx(expected)
    with pytest.raises(ValueError, match="strictly positive"):
        _pool(
            np.array([0.0, 1.0]),
            "power_pool",
            1.0,
            {"pooling": {"positive_score_rule": "raise"}},
        )


def test_learned_pool_parameter_fit_is_reproducible():
    rows = []
    for index in range(5):
        rows.append(
            {
                "predictions": {
                    "clap_baseline": {
                        "calibrated_segments": {"MI": [1.0 + index, 2.0 + index]}
                    }
                },
                "mi": 1.5 + index,
            }
        )
    first = _fit_pool_parameter(
        rows,
        backbone="clap_baseline",
        dimension="MI",
        method="attention",
        epochs=5,
        learning_rate=0.01,
        seed=9,
        epsilon=1e-6,
    )
    second = _fit_pool_parameter(
        rows,
        backbone="clap_baseline",
        dimension="MI",
        method="attention",
        epochs=5,
        learning_rate=0.01,
        seed=9,
        epsilon=1e-6,
    )
    assert first["parameter"] == pytest.approx(second["parameter"])
    np.testing.assert_allclose(first["trace"], second["trace"])


def test_smoke_runner_writes_all_phase_files_from_verified_tiny_cache(tmp_path):
    """Exercise the complete CPU wiring without downloading either encoder."""

    config_path = tmp_path / "configs" / "project.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "project": {"seed": 17, "cache_root": "cache", "results_root": "results"},
        "outputs": {
            "manifest_file": "cache/manifest.jsonl",
            "model_dir": "results/models",
            "run_dir": "results/runs",
        },
        "training": {"head_hidden_dim": 4, "head_second_hidden_dim": 3, "pooling_epochs": 2, "pooling_learning_rate": 0.01},
        "temperature": {"grid": [0.1, 1.0], "selection_metric": "mse"},
        "calibration": {"mean_difference_threshold": 0.1, "slope_lower": 0.9, "slope_upper": 1.1},
        "pooling": {"positive_score_rule": "clip_epsilon", "positive_score_epsilon": 1e-6},
        "statistics": {"bootstrap_resamples": 4},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cache_root = tmp_path / "cache"
    manifest_path = cache_root / "manifest.jsonl"
    clap_manifest = cache_root / "clap" / "features.jsonl"
    mert_manifest = cache_root / "mert" / "features.jsonl"
    rows = []
    # Two rows per split provide non-degenerate smoke metrics and a train set
    # for scalar pooling parameter fitting.
    for split_index, split in enumerate(("train", "dev", "test")):
        for item in range(2):
            clip_id = f"clip-{split}-{item}.wav"
            rows.append({
                "clip_id": clip_id,
                "prompt_id": f"P{item + 1:03d}",
                "split": split,
                "mi": float(1.0 + split_index + item),
                "ta": float(1.5 + split_index + item),
            })
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")

    def add_feature(cache_manifest, encoder, kind, clip_id, array, split):
        destination = cache_root / encoder / kind / f"{clip_id.replace('.wav', '')}.npy"
        atomic_save_npy(np.asarray(array, dtype=np.float32), destination)
        entry = cache_entry(
            destination,
            split=split,
            clip_id=clip_id,
            encoder=encoder,
            checkpoint="unit",
            preprocessing_version="unit",
            segment_plan_hash="unit",
            kind=kind,
        )
        write_manifest_entry(entry, cache_manifest)

    for row in rows:
        split = row["split"]
        clip_id = row["clip_id"]
        prompt = row["prompt_id"]
        add_feature(clap_manifest, "clap", "audio_full", clip_id, np.ones(512), split)
        add_feature(clap_manifest, "clap", "audio_seg", clip_id, np.ones((2, 512)), split)
        add_feature(clap_manifest, "clap", "text", prompt, np.ones(512), "all")
        add_feature(mert_manifest, "mert", "audio_full", clip_id, np.ones(768), split)
        add_feature(mert_manifest, "mert", "audio_seg", clip_id, np.ones((2, 768)), split)

    model_dir = tmp_path / "results" / "models"
    save_model_bundle(CLAPBaseline(hidden_dim=4, second_hidden_dim=3), model_dir / "clap_baseline.pt", {})
    save_model_bundle(MERTAudio(hidden_dim=4, second_hidden_dim=3), model_dir / "mert_audio.pt", {})
    save_model_bundle(CLAPMERT(hidden_dim=4, second_hidden_dim=3), model_dir / "clap_mert.pt", {})

    args = type("Args", (), {
        "config": config_path,
        "all": True,
        "smoke": True,
        "manifest": None,
        "clap_cache": None,
        "mert_cache": None,
        "model_dir": None,
        "output_dir": None,
        "limit": 1,
        "seed": 17,
        "bootstrap_resamples": 4,
        "pooling_epochs": 1,
    })()
    summary = _run(args)
    assert summary["mode"] == "smoke"
    for relative in summary["phase_files"].values():
        assert (tmp_path / relative).is_file()
