"""Build the paper's data figures from results/p1 JSONs.

  fig_tcad.pdf      — localisation evidence (layer profile + score spectrum)
  fig_tradeoff.pdf  — the ASR-vs-clean-MSE Pareto picture (main figure)
  fig_closedloop.pdf— score distributions before/after mitigation

All figures are written to submission/ as vector PDFs.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("results/p1")
OUT_DIR = Path("submission")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)

E1 = json.loads((RESULTS / "e1_localization.json").read_text(encoding="utf-8"))
E2 = json.loads((RESULTS / "e2_tradeoff.json").read_text(encoding="utf-8"))
E4 = json.loads((RESULTS / "e4_baselines.json").read_text(encoding="utf-8"))
E5 = json.loads((RESULTS / "e5_closedloop.json").read_text(encoding="utf-8"))
E10 = json.loads((RESULTS / "e10_calibration_pathology.json").read_text(encoding="utf-8"))


def _study():
    from src.mitigation.data import load_clap_study, load_head_for_seed
    from src.mitigation.tcad import tcad_scores

    data = load_clap_study()
    head = load_head_for_seed(20260907)
    raw = tcad_scores(head, data.dev_clean, data.dev_trig)
    return data, head, raw


def fig_tcad() -> None:
    data, head, raw = _study()

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6))
    ax = axes[0]
    layers = E1["layers"]
    labels = [f"L{row['layer']}\n({row['neurons']}u)" for row in layers]
    x = np.arange(len(layers))
    ax.bar(x - 0.19, [row["tcad_fraction"] for row in layers], width=0.38, label="TCAD mass", color="#2166ac")
    ax.bar(x + 0.19, [row["share_of_neurons"] for row in layers], width=0.38, label="neuron share", color="#bdbdbd")
    ax2 = ax.twinx()
    ax2.plot(x, [row["top30_count"] / 30.0 for row in layers], "o--", ms=3, lw=0.9, color="#b2182b", label="top-30 share")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of total")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    ax.set_title("(a) Layer profile (CLAP)")

    ax = axes[1]
    for layer, scores in enumerate(raw.per_layer):
        sorted_scores = np.sort(scores)[::-1]
        ax.plot(np.arange(1, len(sorted_scores) + 1), sorted_scores, lw=1.0, label=f"L{layer}")
    ax.set_xlabel("neuron rank within layer")
    ax.set_ylabel("TCAD score")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("(b) Score spectrum")
    fig.savefig(OUT_DIR / "fig_tcad.pdf")
    plt.close(fig)


def fig_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    prune = [row for row in E2["prune_sweep"] if row["k"] > 0]
    ax.plot(
        [row["clean_mse"] for row in prune],
        [100 * row["asr"] for row in prune],
        "o-", ms=3, lw=1.0, color="#2166ac", label="z-TCAD pruning (k sweep)",
    )
    dampen = [row for row in E2["dampen_sweep"] if row["alpha"] in (0.2, 0.3, 0.5)]
    ax.plot(
        [row["clean_mse"] for row in dampen],
        [100 * row["asr"] for row in dampen],
        "s-", ms=3, lw=1.0, color="#4393c3", label=r"z-TCAD dampening ($k\times\alpha$)",
    )
    combined = [row for row in E2["combined"] if row["calibration"] != "none"]
    ax.scatter(
        [row["clean_mse"] for row in combined],
        [100 * row["asr"] for row in combined],
        marker="^", s=10, color="#d6604d", label="combined (+calib.)",
    )
    markers = {
        "afp": ("v", "#e08214", "AFP"),
        "anc": ("D", "#8073ac", "ANC"),
        "gradient": ("P", "#5aae61", "grad-sens"),
        "random": ("X", "#878787", "random"),
    }
    for name, (marker, color, label) in markers.items():
        row = E4["budgets"]["0.28"].get(name)
        if row:
            ax.scatter(
                [row["clean_mse"]], [100 * row["asr"]], marker=marker, s=22, color=color, label=label,
            )
    upper = E4["upper_bounds"]
    ax.scatter(
        [upper["full_100ep"]["clean_mse"]], [100 * upper["full_100ep"]["asr"]],
        marker="*", s=80, color="#1a9850", label="full retrain (100 ep)", zorder=5,
    )
    ax.scatter(
        [upper["partial_10ep"]["clean_mse"]], [100 * upper["partial_10ep"]["asr"]],
        marker="d", s=25, color="#66c2a5", label="partial retrain (10 ep)", zorder=5,
    )
    bd_mse = E4["backdoored_mse"]
    ax.scatter([bd_mse], [100 * 0.708], marker="x", s=30, color="black", label="backdoored (3 seeds)", zorder=5)
    ours = E4["ours_aggregate"]
    ax.errorbar(
        [ours["clean_mse"]["mean"]], [100 * ours["asr"]["mean"]],
        xerr=[ours["clean_mse"]["std"]], yerr=[100 * ours["asr"]["std"]],
        fmt="o", ms=5, color="#b2182b", capsize=2, label="z-TCAD dampen $k{=}10$, $\\alpha{=}0.2$ (3 seeds)", zorder=6,
    )

    ax.set_xlabel("clean-test MSE (lower is better)")
    ax.set_ylabel("attack success rate (%)")
    ax.set_ylim(-4, 100)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(OUT_DIR / "fig_tradeoff.pdf")
    plt.close(fig)


def fig_closedloop() -> None:
    from src.mitigation.data import CLAP_AUDIO, load_clap_study, load_head_for_seed, system_of
    from src.mitigation.evaluation import predict_scores
    from src.mitigation.strategies import dampen_neurons
    from src.mitigation.tcad import normalize_per_layer, tcad_scores
    from src.features.extraction import read_manifest

    data = load_clap_study()
    head = load_head_for_seed(20260907)
    zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
    mitigated = copy.deepcopy(head)
    dampen_neurons(mitigated, zrank.top_k(10), 0.2)

    test_rows = sorted(
        (r for r in read_manifest("cache/manifest.jsonl", split="test") if system_of(r["clip_id"]) == "026"),
        key=lambda r: r["clip_id"],
    )
    clean24 = np.stack(
        [np.load(CLAP_AUDIO / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in test_rows]
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6), sharey=True)
    for ax, model, title in (
        (axes[0], head, f"(a) backdoored (AUC {E5['backdoored']['gmm_auc']:.2f})"),
        (axes[1], mitigated, f"(b) mitigated (AUC {E5['mitigated_variants']['dampen_k10_a02']['gmm_auc']:.2f})"),
    ):
        trig = predict_scores(model, data.trig_test)
        clean = predict_scores(model, clean24)
        ax.hist(clean, bins=12, range=(1, 6), color="#4393c3", alpha=0.85, label="clean")
        ax.hist(trig, bins=12, range=(1, 6), color="#b2182b", alpha=0.85, label="triggered")
        ax.axvline(5.0, color="black", lw=0.7, ls="--")
        ax.set_title(title)
        ax.set_xlabel("predicted MOS")
    axes[0].set_ylabel("clips")
    axes[0].legend(frameon=False)
    fig.savefig(OUT_DIR / "fig_closedloop.pdf")
    plt.close(fig)


def fig_calibration() -> None:
    recipes = ["none", "bias_shift", "affine", "gd_full_1step", "anchored_adam", "trigger_aware"]
    labels = ["none", "bias", "affine", "GD\nfull head", "anchored\nAdam", "trigger\naware"]
    panels = [
        ("raw_dampen_k10_a3", "raw ranking, $k{=}10$"),
        ("z_dampen_k10_a3", "z ranking, $k{=}10$"),
        ("raw_dampen_k20_a3", "raw ranking, $k{=}20$"),
        ("z_dampen_k20_a3", "z ranking, $k{=}20$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.7), sharey=True)
    x = np.arange(len(recipes))
    for ax, (key, title) in zip(axes, panels):
        rows = E10[key]
        asr = [100 * rows[r]["asr"] for r in recipes]
        mse = [rows[r]["clean_mse"] for r in recipes]
        colors = ["#4393c3" if rows[r]["clean_mse"] < 1.0 else "#cccccc" for r in recipes]
        ax.bar(x, asr, color=colors, width=0.62)
        ax2 = ax.twinx()
        ax2.plot(x, mse, "o--", color="#b2182b", ms=3, lw=0.9)
        ax2.set_yscale("log")
        ax2.set_ylim(0.2, 100)
        if ax is axes[-1]:
            ax2.set_ylabel("clean MSE", color="#b2182b")
            ax2.tick_params(axis="y", colors="#b2182b")
        else:
            ax2.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("ASR (%)")
    fig.savefig(OUT_DIR / "fig_calibration.pdf")
    plt.close(fig)


def main() -> int:
    fig_tcad()
    fig_tradeoff()
    fig_calibration()
    fig_closedloop()
    print("figures written to", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
