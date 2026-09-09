"""Focused schema and portability tests for result-table rendering."""

from __future__ import annotations

import csv
import json
import re

import pytest

from scripts.render_results import ResultSchemaError, render_result_tables


TABLE_ROWS = (
    ("clap_baseline", "mean"),
    ("clap_baseline", "soft_min"),
    ("clap_baseline", "min"),
    ("clap_baseline", "attention"),
    ("clap_baseline", "auto_pool"),
    ("clap_baseline", "power_pool"),
    ("clap_mert", "mean"),
    ("clap_mert", "soft_min"),
    ("mert_audio", "mean"),
    ("mert_audio", "soft_min"),
)


def _metrics(value: float) -> dict:
    return {
        "mse": value,
        "mae": value + 0.01,
        "signed_error": value - 0.02,
        "spearman_rho": 0.8 - value / 10.0,
    }


def _bootstrap(metric: str, estimate: float) -> dict:
    return {
        "metric": metric,
        "estimate": estimate,
        "lower": estimate - 0.01,
        "upper": estimate + 0.01,
        "confidence_level": 0.95,
        "n_resamples": 2000,
        "valid_resamples": 2000,
    }


def _payloads() -> tuple[dict, dict, dict]:
    calibration_rows = []
    for backbone in ("clap_baseline", "clap_mert", "mert_audio"):
        for dimension in ("MI", "TA"):
            calibration_rows.append(
                {
                    "backbone": backbone,
                    "dimension": dimension,
                    "mean_difference": 0.01,
                    "pearson_r": 0.93,
                    "bland_altman_lower": -0.12,
                    "bland_altman_upper": 0.14,
                    "triggered": False,
                }
            )
    calibration = {"split": "dev", "n_dev": 412, "rows": calibration_rows}

    groups = {}
    for group, n, offset in (("low", 210, 0.10), ("high", 203, 0.20)):
        groups[group] = {
            "n": n,
            "backbones": {
                "clap_baseline": {
                    "MI": {
                        "full_mean": _metrics(offset),
                        "segment_mean": _metrics(offset + 0.01),
                    },
                    "TA": {
                        "full_mean": _metrics(offset + 0.02),
                        "segment_mean": _metrics(offset + 0.03),
                    },
                }
            },
        }
    diagnostic = {
        "ddof": 0,
        "development_threshold": 0.25,
        "test": {"n": 413, "n_high": 203, "n_low": 210},
        "groups": groups,
    }

    rows = []
    for index, (backbone, aggregation) in enumerate(TABLE_ROWS, start=1):
        dimensions = {}
        for dim_index, dimension in enumerate(("MI", "TA")):
            value = index / 100.0 + dim_index / 1000.0
            dimensions[dimension] = {
                "metrics": _metrics(value),
                "bootstrap": {
                    "mse": _bootstrap("mse", value),
                    "spearman": _bootstrap("spearman_rho", 0.8 - value / 10.0),
                },
                "subgroups": {
                    "high": {"n": 203, "metrics": _metrics(value + 0.02)},
                    "low": {"n": 210, "metrics": _metrics(value - 0.002)},
                },
            }
        rows.append(
            {
                "backbone": backbone,
                "aggregation": aggregation,
                "dimensions": dimensions,
            }
        )
    main = {
        "split": "test",
        "n_test": 413,
        "temperatures": {
            "clap_baseline": {"MI": 1.0, "TA": 2.0},
            "clap_mert": {"MI": 0.5, "TA": 1.0},
            "mert_audio": {"MI": 2.0, "TA": 5.0},
        },
        "rows": rows,
    }
    return calibration, diagnostic, main


def _write_inputs(tmp_path, payloads):
    paths = []
    for name, payload in zip(("calibration", "diagnostic", "main_aggregation"), payloads):
        path = tmp_path / "input" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    return paths


def test_renderer_emits_exact_table_shapes_and_portable_fragments(tmp_path):
    calibration, diagnostic, main = _write_inputs(tmp_path, _payloads())
    output = tmp_path / "results" / "tables"

    manifest = render_result_tables(
        calibration_path=calibration,
        diagnostic_path=diagnostic,
        main_aggregation_path=main,
        output_dir=output,
        project_root=tmp_path,
    )

    assert manifest["row_counts"] == {
        "table_i": 5,
        "table_ii": 2,
        "table_ii_causal_control": 4,
        "table_iii": 10,
        "table_iii_bootstrap": 40,
        "table_iv": 8,
        "selected_temperatures": 6,
    }
    expected = {
        "table_i_calibration.csv",
        "table_i_calibration.tex",
        "table_ii_diagnostic.csv",
        "table_ii_diagnostic.tex",
        "table_ii_causal_control.csv",
        "table_ii_causal_control.tex",
        "table_iii_full_test.csv",
        "table_iii_full_test.tex",
        "table_iii_bootstrap.csv",
        "table_iv_variance_groups.csv",
        "table_iv_variance_groups.tex",
        "selected_temperatures.csv",
        "selected_temperatures.tex",
        "render_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    with (output / "table_iii_full_test.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert rows[0]["backbone"] == "CLAP-Baseline"
    assert rows[-1]["aggregation"] == "Soft-min"
    assert len((output / "table_iv_variance_groups.csv").read_text(encoding="utf-8").splitlines()) == 9

    combined = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    assert "TBD" not in combined
    assert not re.search(r"[A-Za-z]:[\\/]", combined)
    assert "\\begin{tabular}" in (output / "table_i_calibration.tex").read_text(encoding="utf-8")


def test_renderer_fails_before_writing_when_required_metric_is_missing(tmp_path):
    payloads = list(_payloads())
    del payloads[2]["rows"][0]["dimensions"]["MI"]["metrics"]["mse"]
    calibration, diagnostic, main = _write_inputs(tmp_path, payloads)
    output = tmp_path / "tables"

    with pytest.raises(ResultSchemaError, match=r"mse"):
        render_result_tables(
            calibration_path=calibration,
            diagnostic_path=diagnostic,
            main_aggregation_path=main,
            output_dir=output,
            project_root=tmp_path,
        )
    assert not output.exists()


def test_renderer_rejects_zero_sized_subgroups_instead_of_filling_tbd(tmp_path):
    payloads = list(_payloads())
    payloads[1]["groups"]["high"]["n"] = 0
    calibration, diagnostic, main = _write_inputs(tmp_path, payloads)

    with pytest.raises(ResultSchemaError, match=r"positive integer"):
        render_result_tables(
            calibration_path=calibration,
            diagnostic_path=diagnostic,
            main_aggregation_path=main,
            output_dir=tmp_path / "tables",
            project_root=tmp_path,
        )


def test_renderer_rejects_placeholder_strings_anywhere_in_source(tmp_path):
    payloads = list(_payloads())
    payloads[0]["metadata"] = {"note": "TBD after rerun"}
    calibration, diagnostic, main = _write_inputs(tmp_path, payloads)

    with pytest.raises(ResultSchemaError, match=r"placeholder"):
        render_result_tables(
            calibration_path=calibration,
            diagnostic_path=diagnostic,
            main_aggregation_path=main,
            output_dir=tmp_path / "tables",
            project_root=tmp_path,
        )
