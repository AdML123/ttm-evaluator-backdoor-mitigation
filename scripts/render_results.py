"""Render experiment JSON artifacts into CSV and LaTeX table fragments.

This script is intentionally a pure presentation step: it never recomputes a
metric and never fills a missing value.  It validates the Table I--IV source
schema written by ``scripts/run_experiments.py`` and fails if a required split,
row, subgroup, metric, or confidence interval is absent or non-finite.

The generated files contain only canonical method labels and measured numeric
values.  Input paths are represented by repository-relative names in the
render manifest, preventing machine-specific paths from entering shareable
derived outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


BACKBONE_LABELS = {
    "clap_baseline": "CLAP-Baseline",
    "clap_mert": "CLAP+MERT",
    "mert_audio": "MERT-audio",
}

AGGREGATION_LABELS = {
    "mean": "Mean",
    "soft_min": "Soft-min",
    "min": "Min",
    "attention": "Attention",
    "auto_pool": "Auto-pool",
    "power_pool": "Power-pool",
}

TABLE_I_ROWS = (
    ("clap_baseline", "MI"),
    ("clap_baseline", "TA"),
    ("clap_mert", "MI"),
    ("clap_mert", "TA"),
    ("mert_audio", "MI"),
)

TABLE_III_ROWS = (
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

TABLE_IV_ROWS = (
    ("clap_baseline", "mean"),
    ("clap_baseline", "soft_min"),
    ("clap_baseline", "min"),
    ("clap_baseline", "auto_pool"),
    ("clap_mert", "mean"),
    ("clap_mert", "soft_min"),
    ("mert_audio", "mean"),
    ("mert_audio", "soft_min"),
)

_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])[A-Z]:[\\/]")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?:^|[\s='\"])/(?:home|Users|mnt|tmp|var)/")


class ResultSchemaError(ValueError):
    """Raised when a result artifact cannot support the planned table."""


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="directory containing calibration.json, diagnostic.json, and main_aggregation.json",
    )
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--diagnostic", type=Path, default=None)
    parser.add_argument("--main-aggregation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    return resolved.parent.parent if resolved.parent.name.lower() == "configs" else resolved.parent


def _resolve(path: str | Path, *, root: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _repository_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return f"external/{path.name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} result not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultSchemaError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ResultSchemaError(f"{label} must contain a JSON object")
    _reject_placeholders(payload, context=label)
    return payload


def _reject_placeholders(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, context=f"{context}[{index}]")
    elif isinstance(value, str) and re.search(r"\b(?:TBD|TODO|PLACEHOLDER)\b", value, re.IGNORECASE):
        raise ResultSchemaError(f"unresolved placeholder at {context}")


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultSchemaError(f"{context} must be an object")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ResultSchemaError(f"{context} must be an array")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise ResultSchemaError(f"missing {context}.{key}")
    return mapping[key]


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ResultSchemaError(f"{context} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"{context} must be a finite number") from exc
    if not math.isfinite(number):
        raise ResultSchemaError(f"{context} must be a finite number")
    return number


def _positive_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ResultSchemaError(f"{context} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"{context} must be a positive integer") from exc
    if number <= 0 or number != value:
        raise ResultSchemaError(f"{context} must be a positive integer")
    return number


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ResultSchemaError(f"{context} must be boolean")
    return value


def _unique_index(
    rows: Sequence[Any],
    *,
    keys: tuple[str, str],
    context: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, context=f"{context}[{index}]")
        identity = (
            str(_required(row, keys[0], context=f"{context}[{index}]")),
            str(_required(row, keys[1], context=f"{context}[{index}]")),
        )
        if identity in output:
            raise ResultSchemaError(f"duplicate {context} row: {identity[0]}/{identity[1]}")
        output[identity] = row
    return output


def _metric(metrics: Any, name: str, *, context: str) -> float:
    values = _mapping(metrics, context=context)
    return _finite(_required(values, name, context=context), context=f"{context}.{name}")


def _validate_bootstrap(dimension: Mapping[str, Any], *, context: str) -> None:
    bootstrap = _mapping(_required(dimension, "bootstrap", context=context), context=f"{context}.bootstrap")
    for key in ("mse", "spearman"):
        result = _mapping(_required(bootstrap, key, context=f"{context}.bootstrap"), context=f"{context}.bootstrap.{key}")
        lower = _finite(_required(result, "lower", context=f"{context}.bootstrap.{key}"), context=f"{context}.bootstrap.{key}.lower")
        upper = _finite(_required(result, "upper", context=f"{context}.bootstrap.{key}"), context=f"{context}.bootstrap.{key}.upper")
        estimate = _finite(_required(result, "estimate", context=f"{context}.bootstrap.{key}"), context=f"{context}.bootstrap.{key}.estimate")
        if lower > upper:
            raise ResultSchemaError(f"{context}.bootstrap.{key} has reversed interval")
        # Percentile intervals need not contain the point estimate for every
        # statistic, so only finiteness/order are hard schema requirements.
        _ = estimate
        _positive_integer(_required(result, "n_resamples", context=f"{context}.bootstrap.{key}"), context=f"{context}.bootstrap.{key}.n_resamples")
        _positive_integer(_required(result, "valid_resamples", context=f"{context}.bootstrap.{key}"), context=f"{context}.bootstrap.{key}.valid_resamples")


def table_i_rows(calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(_required(calibration, "split", context="calibration")) != "dev":
        raise ResultSchemaError("Table I calibration split must be dev")
    _positive_integer(_required(calibration, "n_dev", context="calibration"), context="calibration.n_dev")
    rows = _sequence(_required(calibration, "rows", context="calibration"), context="calibration.rows")
    index = _unique_index(rows, keys=("backbone", "dimension"), context="calibration.rows")
    output: list[dict[str, Any]] = []
    for backbone, dimension in TABLE_I_ROWS:
        source = index.get((backbone, dimension))
        if source is None:
            raise ResultSchemaError(f"missing Table I row: {backbone}/{dimension}")
        output.append(
            {
                "backbone": BACKBONE_LABELS[backbone],
                "dimension": dimension,
                "mean_difference": _finite(_required(source, "mean_difference", context=f"calibration.{backbone}.{dimension}"), context=f"calibration.{backbone}.{dimension}.mean_difference"),
                "pearson_r": _finite(_required(source, "pearson_r", context=f"calibration.{backbone}.{dimension}"), context=f"calibration.{backbone}.{dimension}.pearson_r"),
                "bland_altman_lower": _finite(_required(source, "bland_altman_lower", context=f"calibration.{backbone}.{dimension}"), context=f"calibration.{backbone}.{dimension}.bland_altman_lower"),
                "bland_altman_upper": _finite(_required(source, "bland_altman_upper", context=f"calibration.{backbone}.{dimension}"), context=f"calibration.{backbone}.{dimension}.bland_altman_upper"),
                "calibration_triggered": _boolean(_required(source, "triggered", context=f"calibration.{backbone}.{dimension}"), context=f"calibration.{backbone}.{dimension}.triggered"),
            }
        )
    return output


def table_ii_rows(diagnostic: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = _finite(_required(diagnostic, "development_threshold", context="diagnostic"), context="diagnostic.development_threshold")
    if int(_required(diagnostic, "ddof", context="diagnostic")) != 0:
        raise ResultSchemaError("Table II requires population segment standard deviation (ddof=0)")
    test = _mapping(_required(diagnostic, "test", context="diagnostic"), context="diagnostic.test")
    _positive_integer(_required(test, "n", context="diagnostic.test"), context="diagnostic.test.n")
    groups = _mapping(_required(diagnostic, "groups", context="diagnostic"), context="diagnostic.groups")
    table: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    for group_name in ("low", "high"):
        group = _mapping(_required(groups, group_name, context="diagnostic.groups"), context=f"diagnostic.groups.{group_name}")
        n = _positive_integer(_required(group, "n", context=f"diagnostic.groups.{group_name}"), context=f"diagnostic.groups.{group_name}.n")
        backbones = _mapping(_required(group, "backbones", context=f"diagnostic.groups.{group_name}"), context=f"diagnostic.groups.{group_name}.backbones")
        clap = _mapping(_required(backbones, "clap_baseline", context=f"diagnostic.groups.{group_name}.backbones"), context=f"diagnostic.groups.{group_name}.backbones.clap_baseline")
        row: dict[str, Any] = {
            "group": "Low variance" if group_name == "low" else "High variance",
            "sigma_rule": f"sigma <= {threshold:.10g}" if group_name == "low" else f"sigma > {threshold:.10g}",
            "mi_mae": None,
            "mi_signed_error": None,
            "ta_mae": None,
            "ta_signed_error": None,
            "n": n,
        }
        for dimension, prefix in (("MI", "mi"), ("TA", "ta")):
            dimension_data = _mapping(_required(clap, dimension, context=f"diagnostic.groups.{group_name}.clap_baseline"), context=f"diagnostic.groups.{group_name}.clap_baseline.{dimension}")
            full = _mapping(_required(dimension_data, "full_mean", context=f"diagnostic.groups.{group_name}.{dimension}"), context=f"diagnostic.groups.{group_name}.{dimension}.full_mean")
            segment = _mapping(_required(dimension_data, "segment_mean", context=f"diagnostic.groups.{group_name}.{dimension}"), context=f"diagnostic.groups.{group_name}.{dimension}.segment_mean")
            row[f"{prefix}_mae"] = _metric(full, "mae", context=f"diagnostic.groups.{group_name}.{dimension}.full_mean")
            row[f"{prefix}_signed_error"] = _metric(full, "signed_error", context=f"diagnostic.groups.{group_name}.{dimension}.full_mean")
            causal.append(
                {
                    "group": row["group"],
                    "dimension": dimension,
                    "full_clip_mae": row[f"{prefix}_mae"],
                    "full_clip_signed_error": row[f"{prefix}_signed_error"],
                    "segment_mean_mae": _metric(segment, "mae", context=f"diagnostic.groups.{group_name}.{dimension}.segment_mean"),
                    "segment_mean_signed_error": _metric(segment, "signed_error", context=f"diagnostic.groups.{group_name}.{dimension}.segment_mean"),
                    "n": n,
                }
            )
        table.append(row)
    if sum(int(row["n"]) for row in table) != int(test["n"]):
        raise ResultSchemaError(
            "Table II subgroup counts do not sum to diagnostic.test.n"
        )
    return table, causal


def _main_index(main: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    if str(_required(main, "split", context="main_aggregation")) != "test":
        raise ResultSchemaError("Tables III-IV require the test split")
    _positive_integer(_required(main, "n_test", context="main_aggregation"), context="main_aggregation.n_test")
    rows = _sequence(_required(main, "rows", context="main_aggregation"), context="main_aggregation.rows")
    return _unique_index(rows, keys=("backbone", "aggregation"), context="main_aggregation.rows")


def table_iii_rows(main: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = _main_index(main)
    output: list[dict[str, Any]] = []
    uncertainty: list[dict[str, Any]] = []
    for backbone, aggregation in TABLE_III_ROWS:
        source = index.get((backbone, aggregation))
        if source is None:
            raise ResultSchemaError(f"missing Table III row: {backbone}/{aggregation}")
        dimensions = _mapping(_required(source, "dimensions", context=f"main.{backbone}.{aggregation}"), context=f"main.{backbone}.{aggregation}.dimensions")
        row: dict[str, Any] = {
            "backbone": BACKBONE_LABELS[backbone],
            "aggregation": AGGREGATION_LABELS[aggregation],
        }
        for dimension, prefix in (("MI", "mi"), ("TA", "ta")):
            value = _mapping(_required(dimensions, dimension, context=f"main.{backbone}.{aggregation}.dimensions"), context=f"main.{backbone}.{aggregation}.{dimension}")
            metrics = _mapping(_required(value, "metrics", context=f"main.{backbone}.{aggregation}.{dimension}"), context=f"main.{backbone}.{aggregation}.{dimension}.metrics")
            row[f"{prefix}_mse"] = _metric(metrics, "mse", context=f"main.{backbone}.{aggregation}.{dimension}.metrics")
            row[f"{prefix}_srcc"] = _metric(metrics, "spearman_rho", context=f"main.{backbone}.{aggregation}.{dimension}.metrics")
            _validate_bootstrap(value, context=f"main.{backbone}.{aggregation}.{dimension}")
            bootstrap = _mapping(value["bootstrap"], context=f"main.{backbone}.{aggregation}.{dimension}.bootstrap")
            for metric_key, metric_label in (("mse", "MSE"), ("spearman", "SRCC")):
                interval = _mapping(bootstrap[metric_key], context=f"main.{backbone}.{aggregation}.{dimension}.bootstrap.{metric_key}")
                uncertainty.append(
                    {
                        "backbone": BACKBONE_LABELS[backbone],
                        "aggregation": AGGREGATION_LABELS[aggregation],
                        "dimension": dimension,
                        "metric": metric_label,
                        "estimate": _finite(interval["estimate"], context="bootstrap.estimate"),
                        "ci_lower": _finite(interval["lower"], context="bootstrap.lower"),
                        "ci_upper": _finite(interval["upper"], context="bootstrap.upper"),
                        "confidence_level": _finite(_required(interval, "confidence_level", context="bootstrap"), context="bootstrap.confidence_level"),
                        "valid_resamples": _positive_integer(interval["valid_resamples"], context="bootstrap.valid_resamples"),
                    }
                )
        output.append(row)
    return output, uncertainty


def table_iv_rows(main: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = _main_index(main)
    output: list[dict[str, Any]] = []
    for backbone, aggregation in TABLE_IV_ROWS:
        source = index.get((backbone, aggregation))
        if source is None:
            raise ResultSchemaError(f"missing Table IV row: {backbone}/{aggregation}")
        dimensions = _mapping(_required(source, "dimensions", context=f"main.{backbone}.{aggregation}"), context=f"main.{backbone}.{aggregation}.dimensions")
        mi = _mapping(_required(dimensions, "MI", context=f"main.{backbone}.{aggregation}.dimensions"), context=f"main.{backbone}.{aggregation}.MI")
        subgroups = _mapping(_required(mi, "subgroups", context=f"main.{backbone}.{aggregation}.MI"), context=f"main.{backbone}.{aggregation}.MI.subgroups")
        values: dict[str, float] = {}
        counts: dict[str, int] = {}
        for group in ("high", "low"):
            subgroup = _mapping(_required(subgroups, group, context=f"main.{backbone}.{aggregation}.MI.subgroups"), context=f"main.{backbone}.{aggregation}.MI.subgroups.{group}")
            counts[group] = _positive_integer(_required(subgroup, "n", context=f"main.{backbone}.{aggregation}.MI.subgroups.{group}"), context=f"main.{backbone}.{aggregation}.MI.subgroups.{group}.n")
            values[group] = _metric(_required(subgroup, "metrics", context=f"main.{backbone}.{aggregation}.MI.subgroups.{group}"), "mse", context=f"main.{backbone}.{aggregation}.MI.subgroups.{group}.metrics")
        output.append(
            {
                "backbone": BACKBONE_LABELS[backbone],
                "aggregation": AGGREGATION_LABELS[aggregation],
                "high_var_mse": values["high"],
                "low_var_mse": values["low"],
                "high_n": counts["high"],
                "low_n": counts["low"],
            }
        )
    return output


def selected_temperature_rows(main: Mapping[str, Any]) -> list[dict[str, Any]]:
    temperatures = _mapping(_required(main, "temperatures", context="main_aggregation"), context="main_aggregation.temperatures")
    output: list[dict[str, Any]] = []
    for backbone in BACKBONE_LABELS:
        values = _mapping(_required(temperatures, backbone, context="main_aggregation.temperatures"), context=f"main_aggregation.temperatures.{backbone}")
        for dimension in ("MI", "TA"):
            output.append(
                {
                    "backbone": BACKBONE_LABELS[backbone],
                    "dimension": dimension,
                    "selected_tau": _finite(_required(values, dimension, context=f"main_aggregation.temperatures.{backbone}"), context=f"main_aggregation.temperatures.{backbone}.{dimension}"),
                }
            )
    return output


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        raise ResultSchemaError("cannot render an empty CSV")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row[field]) for field in fields})
    return handle.getvalue()


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _number(value: float, digits: int = 3) -> str:
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def _tabular(columns: str, header: str, rows: Iterable[str]) -> str:
    body = "\n".join(f"{row} \\\\" for row in rows)
    return (
        "% Generated by scripts/render_results.py; do not edit manually.\n"
        f"\\begin{{tabular}}{{{columns}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _table_i_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    return _tabular(
        "llrrr",
        "Backbone & Dim. & Mean diff. & Pearson $r$ & 95\\% limits",
        (
            f"{row['backbone']} & {row['dimension']} & {_number(row['mean_difference'])} & {_number(row['pearson_r'])} & [{_number(row['bland_altman_lower'])}, {_number(row['bland_altman_upper'])}]"
            for row in rows
        ),
    )


def _table_ii_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    def sigma(row: Mapping[str, Any]) -> str:
        value = str(row["sigma_rule"])
        threshold = value.split()[-1]
        return f"$\\sigma_s \\le {threshold}$" if "<=" in value else f"$\\sigma_s > {threshold}$"

    return _tabular(
        "llrrrrr",
        "Group & $\\sigma_s$ range & MI MAE & MI error & TA MAE & TA error & $n$",
        (
            f"{row['group']} & {sigma(row)} & {_number(row['mi_mae'])} & {_number(row['mi_signed_error'])} & {_number(row['ta_mae'])} & {_number(row['ta_signed_error'])} & {row['n']}"
            for row in rows
        ),
    )


def _table_ii_causal_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    return _tabular(
        "llrrrr",
        "Group & Dim. & Full MAE & Full error & Segment-mean MAE & Segment-mean error",
        (
            f"{row['group']} & {row['dimension']} & {_number(row['full_clip_mae'])} & {_number(row['full_clip_signed_error'])} & {_number(row['segment_mean_mae'])} & {_number(row['segment_mean_signed_error'])}"
            for row in rows
        ),
    )


def _table_iii_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    return _tabular(
        "llrrrr",
        "Backbone & Aggregation & MI MSE & MI SRCC & TA MSE & TA SRCC",
        (
            f"{row['backbone']} & {row['aggregation']} & {_number(row['mi_mse'])} & {_number(row['mi_srcc'])} & {_number(row['ta_mse'])} & {_number(row['ta_srcc'])}"
            for row in rows
        ),
    )


def _table_iv_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    return _tabular(
        "llrr",
        "Backbone & Aggregation & High-var MSE & Low-var MSE",
        (
            f"{row['backbone']} & {row['aggregation']} & {_number(row['high_var_mse'])} & {_number(row['low_var_mse'])}"
            for row in rows
        ),
    )


def _temperature_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    tokens = {("CLAP-Baseline", "MI"): "ClapMITau", ("CLAP-Baseline", "TA"): "ClapTATau", ("CLAP+MERT", "MI"): "FusionMITau", ("CLAP+MERT", "TA"): "FusionTATau", ("MERT-audio", "MI"): "MertMITau", ("MERT-audio", "TA"): "MertTATau"}
    lines = ["% Generated by scripts/render_results.py; do not edit manually."]
    for row in rows:
        command = tokens[(str(row["backbone"]), str(row["dimension"]))]
        lines.append(f"\\newcommand{{\\{command}}}{{{_number(row['selected_tau'], digits=1)}}}")
    return "\n".join(lines) + "\n"


def _validate_portable(text: str, *, filename: str) -> None:
    if re.search(r"\b(?:TBD|TODO|PLACEHOLDER)\b", text, re.IGNORECASE):
        raise ResultSchemaError(f"generated {filename} contains an unresolved placeholder")
    if _WINDOWS_ABSOLUTE_PATH.search(text) or _POSIX_ABSOLUTE_PATH.search(text):
        raise ResultSchemaError(f"generated {filename} contains an absolute path")


def _write_text(path: Path, text: str) -> None:
    _validate_portable(text, filename=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def render_result_tables(
    *,
    calibration_path: Path,
    diagnostic_path: Path,
    main_aggregation_path: Path,
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    calibration = _load_json(calibration_path, label="calibration")
    diagnostic = _load_json(diagnostic_path, label="diagnostic")
    main = _load_json(main_aggregation_path, label="main aggregation")

    table_i = table_i_rows(calibration)
    table_ii, table_ii_causal = table_ii_rows(diagnostic)
    table_iii, uncertainty = table_iii_rows(main)
    table_iv = table_iv_rows(main)
    temperatures = selected_temperature_rows(main)

    artifacts = {
        "table_i_calibration.csv": _csv_text(table_i, ("backbone", "dimension", "mean_difference", "pearson_r", "bland_altman_lower", "bland_altman_upper", "calibration_triggered")),
        "table_i_calibration.tex": _table_i_tex(table_i),
        "table_ii_diagnostic.csv": _csv_text(table_ii, ("group", "sigma_rule", "mi_mae", "mi_signed_error", "ta_mae", "ta_signed_error", "n")),
        "table_ii_diagnostic.tex": _table_ii_tex(table_ii),
        "table_ii_causal_control.csv": _csv_text(table_ii_causal, ("group", "dimension", "full_clip_mae", "full_clip_signed_error", "segment_mean_mae", "segment_mean_signed_error", "n")),
        "table_ii_causal_control.tex": _table_ii_causal_tex(table_ii_causal),
        "table_iii_full_test.csv": _csv_text(table_iii, ("backbone", "aggregation", "mi_mse", "mi_srcc", "ta_mse", "ta_srcc")),
        "table_iii_full_test.tex": _table_iii_tex(table_iii),
        "table_iii_bootstrap.csv": _csv_text(uncertainty, ("backbone", "aggregation", "dimension", "metric", "estimate", "ci_lower", "ci_upper", "confidence_level", "valid_resamples")),
        "table_iv_variance_groups.csv": _csv_text(table_iv, ("backbone", "aggregation", "high_var_mse", "low_var_mse")),
        "table_iv_variance_groups.tex": _table_iv_tex(table_iv),
        "selected_temperatures.csv": _csv_text(temperatures, ("backbone", "dimension", "selected_tau")),
        "selected_temperatures.tex": _temperature_tex(temperatures),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for filename, text in artifacts.items():
        path = output_dir / filename
        _write_text(path, text)
        output_paths[filename] = path

    manifest = {
        "schema_version": "1.0",
        "generator": "scripts/render_results.py",
        "inputs": {
            "calibration.json": {"path": _repository_path(calibration_path, root=project_root), "sha256": _sha256(calibration_path)},
            "diagnostic.json": {"path": _repository_path(diagnostic_path, root=project_root), "sha256": _sha256(diagnostic_path)},
            "main_aggregation.json": {"path": _repository_path(main_aggregation_path, root=project_root), "sha256": _sha256(main_aggregation_path)},
        },
        "outputs": {
            filename: {"path": _repository_path(path, root=project_root), "sha256": _sha256(path)}
            for filename, path in sorted(output_paths.items())
        },
        "row_counts": {
            "table_i": len(table_i),
            "table_ii": len(table_ii),
            "table_ii_causal_control": len(table_ii_causal),
            "table_iii": len(table_iii),
            "table_iii_bootstrap": len(uncertainty),
            "table_iv": len(table_iv),
            "selected_temperatures": len(temperatures),
        },
        "status": "complete",
    }
    manifest_text = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest_path = output_dir / "render_manifest.json"
    _write_text(manifest_path, manifest_text)
    return manifest


def main() -> int:
    args = _args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _project_root(config_path)
    outputs = config.get("outputs", {})
    if args.run_dir is not None:
        run_dir = _resolve(args.run_dir, root=root)
        default_calibration = run_dir / "calibration.json"
        default_diagnostic = run_dir / "diagnostic.json"
        default_main = run_dir / "main_aggregation.json"
    else:
        default_calibration = root / "results" / "p0" / "calibration.json"
        default_diagnostic = root / "results" / "p0" / "diagnostic.json"
        default_main = _resolve(outputs.get("table_dir", "results/tables"), root=root) / "main_aggregation.json"
    calibration_path = _resolve(args.calibration, root=root) if args.calibration is not None else default_calibration
    diagnostic_path = _resolve(args.diagnostic, root=root) if args.diagnostic is not None else default_diagnostic
    main_path = _resolve(args.main_aggregation, root=root) if args.main_aggregation is not None else default_main
    output_dir = _resolve(args.output_dir, root=root) if args.output_dir is not None else _resolve(outputs.get("table_dir", "results/tables"), root=root)
    manifest = render_result_tables(
        calibration_path=calibration_path,
        diagnostic_path=diagnostic_path,
        main_aggregation_path=main_path,
        output_dir=output_dir,
        project_root=root,
    )
    print(json.dumps(manifest["row_counts"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ResultSchemaError",
    "render_result_tables",
    "selected_temperature_rows",
    "table_i_rows",
    "table_ii_rows",
    "table_iii_rows",
    "table_iv_rows",
]
