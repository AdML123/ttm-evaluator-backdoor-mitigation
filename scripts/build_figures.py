"""Build the paper's data figures from results/p1 JSONs.

Style contract (IEEE): at most three colors -- IEEE blue #00629B for the
proposed/defender quantities, IEEE red #C8102E for backdoor/triggered
quantities, gray #7F7F7F for neutral references -- with line style and
marker shape as the secondary encoding.  Legends sit ABOVE each panel body
never inside it.  Panel layouts: fig_tcad and fig_closedloop are 2x1
stacked, fig_tradeoff carries an inset zoom over the crowded low-MSE
region, fig_calibration uses short horizontal x labels.

  fig_tcad.pdf       -- localization evidence (layer profile + spectrum)
  fig_tradeoff.pdf   -- security-accuracy plane with inset zoom (main figure)
  fig_calibration.pdf-- calibration pathology, raw vs normalized ranking
  fig_closedloop.pdf -- score distributions before/after mitigation
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

# IEEE three-color palette + style encodings
BLUE, RED, GRAY = "#00629B", "#C8102E", "#7F7F7F"
BLACK = "#000000"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)


def _save(fig, stem: str) -> None:
    """Journal export set: vector PDF for the manuscript plus SVG/PNG."""
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600)


def _legend_above(ax, ncol=3, title=None):
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.10), ncol=ncol,
        mode=None, borderaxespad=0.0, handlelength=1.6, columnspacing=0.9,
        handletextpad=0.4, title=title, title_fontsize=7,
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


# ------------------------------------------------------------------ Fig 2
def fig_tcad() -> None:
    data, head, raw = _study()

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 3.4))
    fig.subplots_adjust(hspace=0.62, top=0.86, bottom=0.10, left=0.15, right=0.96)

    layers = E1["layers"]
    x = np.arange(len(layers))
    ax_a.bar(x - 0.19, [row["tcad_fraction"] for row in layers], width=0.38,
             color=BLUE, edgecolor=BLUE, label="TCAD mass")
    ax_a.bar(x + 0.19, [row["share_of_neurons"] for row in layers], width=0.38,
             color="white", edgecolor=GRAY, linewidth=0.8, hatch="////",
             label="neuron share")
    ax_a2 = ax_a.twinx()
    ax_a2.spines["top"].set_visible(False)
    ax_a2.plot(x, [row["top30_count"] / 30.0 for row in layers], linestyle="-.",
               marker="D", markersize=3.5, color=RED, linewidth=0.9, label="top-30 share")
    ax_a2.set_ylim(0, 1)
    ax_a2.set_yticks([0, 0.5, 1.0])
    ax_a2.tick_params(axis="y", colors=RED, labelsize=6.5)
    ax_a2.set_ylabel("top-30 share", color=RED, fontsize=7)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"$h^{{(1)}}$ (256u)", "$h^{(2)}$ (128u)"])
    ax_a.set_ylabel("fraction of total")
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(
        h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.12),
        ncol=3, borderaxespad=0.0, handlelength=1.5, columnspacing=0.8,
    )
    ax_a.set_title("(a) Layer profile", loc="left", pad=14)

    for layer, (color, marker, lsty) in enumerate(
        ((BLUE, "o", "-"), (GRAY, "s", "--"))
    ):
        scores = np.sort(raw.per_layer[layer])[::-1]
        ax_b.plot(
            np.arange(1, len(scores) + 1), scores, linestyle=lsty, marker=marker,
            markersize=2.6, markevery=max(1, len(scores) // 12), linewidth=0.9,
            color=color, label=f"$h^{{({layer + 1})}}$",
        )
    ax_b.set_xlabel("neuron rank within layer")
    ax_b.set_ylabel("TCAD score")
    ax_b.set_yscale("log")  # scores are strictly positive, no pseudocount needed
    _legend_above(ax_b, ncol=2)
    ax_b.set_title("(b) Score spectrum", loc="left", pad=14)

    _save(fig, "fig_tcad")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 3
def fig_tradeoff() -> None:
    """Security-accuracy plane, decomposed: one panel per strategy family.

    Each panel holds a single series so no legend crowd interferes with
    reading it, and every panel repeats two common references, the
    backdoored model (x) and the proposed default (red dot with three-seed
    error bars).  Panel (d) compares all budget-matched landmarks on its
    own zoomed MSE axis, which spreads the tight x-cluster.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.7), sharey=True)
    fig.subplots_adjust(hspace=0.52, wspace=0.10, top=0.80, bottom=0.13,
                        left=0.075, right=0.985)
    (ax_a, ax_c), (ax_b, ax_d) = axes[0], axes[1]  # row-major a,b top; c,d bottom

    prune = [row for row in E2["prune_sweep"] if row["k"] > 0]
    upper = E4["upper_bounds"]
    ours = E4["ours_aggregate"]
    bd = (E4["backdoored_mse"], 75.0)

    def references(ax):
        ax.scatter([bd[0]], [bd[1]], marker="x", s=34, color=BLACK, zorder=5)
        ax.errorbar([ours["clean_mse"]["mean"]], [100 * ours["asr"]["mean"]],
                    xerr=[ours["clean_mse"]["std"]],
                    yerr=[100 * ours["asr"]["std"]],
                    fmt="o", ms=5, color=RED, capsize=2, zorder=6)

    # ---------------- (a) prune dose-response ----------------
    references(ax_a)
    ax_a.plot([r["clean_mse"] for r in prune], [100 * r["asr"] for r in prune],
              linestyle="-", marker="o", markersize=3.2, linewidth=0.9,
              color=BLUE, zorder=4)
    by_k = {row["k"]: row for row in prune}
    for k, off in ((5, (3, 4)), (8, (3, 4)), (10, (4, -9)), (20, (0, 6)), (30, (-6, 6))):
        row = by_k.get(k)
        if row:
            ax_a.annotate(f"$k$={k}", (row["clean_mse"], 100 * row["asr"]),
                          textcoords="offset points", xytext=off, fontsize=6,
                          color=GRAY)
    ax_a.set_xlim(0.26, 0.85)
    ax_a.set_xticks([0.3, 0.5, 0.7])
    ax_a.set_title("(a) z-TCAD prune", loc="left", pad=3)

    # ---------------- (b) dampen over the (k, alpha) grid ----------------
    references(ax_b)
    markers = {0.1: "^", 0.2: "o", 0.3: "s", 0.5: "D"}
    for alpha, marker in markers.items():
        rows = [r for r in E2["dampen_sweep"] if r["alpha"] == alpha]
        rows.sort(key=lambda r: r["k"])
        ax_b.plot([r["clean_mse"] for r in rows], [100 * r["asr"] for r in rows],
                  linestyle="-", marker=marker, markersize=2.8, linewidth=0.7,
                  color=BLUE, alpha=0.85, zorder=4)
        last = rows[-1]
        ax_b.annotate("$\\alpha$=" + str(alpha), (last["clean_mse"], 100 * last["asr"]),
                      textcoords="offset points", xytext=(4, -2), fontsize=6,
                      color=BLUE)
    ax_b.set_xlim(0.265, 0.435)
    ax_b.set_xticks([0.28, 0.33, 0.38])
    ax_b.set_title("(b) z-TCAD dampen", loc="left", pad=3)

    # ---------------- (c) dampen + calibration ----------------
    references(ax_c)
    aff = [r for r in E2["combined"] if r["calibration"] == "affine"]
    tw = [r for r in E2["combined"] if r["calibration"] == "trigger_aware"]
    ax_c.scatter([r["clean_mse"] for r in aff], [100 * r["asr"] for r in aff],
                 marker="o", s=12, facecolors="none", edgecolors=BLUE,
                 linewidths=0.8, zorder=4, label="+ affine")
    ax_c.scatter([r["clean_mse"] for r in tw], [100 * r["asr"] for r in tw],
                 marker="^", s=12, color=BLUE, zorder=4, label="+ trigger-aware")
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.10), ncol=2,
                borderaxespad=0.0, handlelength=1.2, fontsize=6,
                columnspacing=0.8, handletextpad=0.3)
    ax_c.set_xlim(0.265, 0.475)
    ax_c.set_xticks([0.28, 0.33, 0.38, 0.43])
    ax_c.set_title("(c) dampen + calibration", loc="left", pad=10)
    ax_c.set_xlabel("clean-test MSE")
    ax_c.set_ylabel("ASR (%)")

    # ---------------- (d) budget-matched landmarks, zoomed ----------------
    references(ax_d)
    marks = [
        ("afp", "v", "AFP", (3, -2)),
        ("anc", "D", "ANC", (3, 0)),
        ("gradient", "P", "grad", (3, 3)),
        ("random", "X", "random", (3, 0)),
        ("unpaired", "*", "unpaired", (4, -3)),
        ("raw_tcad", "P", "raw", (3, 6)),
    ]
    for name, marker, label, off in marks:
        row = E4["budgets"]["0.28"].get(name)
        if row:
            ax_d.scatter([row["clean_mse"]], [100 * row["asr"]], marker=marker,
                         s=30, facecolors="none", edgecolors=RED, linewidths=0.9,
                         zorder=5)
            ax_d.annotate(label, (row["clean_mse"], 100 * row["asr"]),
                          textcoords="offset points", xytext=off, fontsize=6,
                          color=RED)
    ax_d.scatter([upper["full_100ep"]["clean_mse"]], [0], marker="*", s=60,
                 color=GRAY, zorder=5)
    ax_d.annotate("full retrain", (upper["full_100ep"]["clean_mse"], 0),
                  textcoords="offset points", xytext=(5, 3), fontsize=6,
                  color=GRAY)
    ax_d.scatter([upper["partial_10ep"]["clean_mse"]], [0], marker="d", s=24,
                 facecolors="none", edgecolors=GRAY, linewidths=0.9, zorder=5)
    ax_d.annotate("partial", (upper["partial_10ep"]["clean_mse"], 0),
                  textcoords="offset points", xytext=(5, 3), fontsize=6,
                  color=GRAY)
    ax_d.annotate("ours", (ours["clean_mse"]["mean"], 100 * ours["asr"]["mean"]),
                  textcoords="offset points", xytext=(6, -7), fontsize=6.5,
                  color=RED)
    ax_d.annotate("backdoored", bd, textcoords="offset points",
                  xytext=(-2, 4), fontsize=6, color=BLACK, ha="right")
    ax_d.set_xlim(0.2655, 0.3005)
    ax_d.set_xticks([0.27, 0.28, 0.29, 0.30])
    ax_d.tick_params(axis="x", labelsize=6)
    ax_d.set_title("(d) budget-matched comparison", loc="left", pad=3)
    ax_d.set_xlabel("clean-test MSE (zoom)")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_ylim(-4, 84)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_a.set_ylabel("ASR (%)")

    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", ms=3, lw=0.9,
                   label="this panel's series"),
        plt.Line2D([], [], color=RED, marker="o", ms=5, lw=0,
                   label="default $(10,0.2)$, 3 seeds"),
        plt.Line2D([], [], color=BLACK, marker="x", ms=5, lw=0,
                   label="backdoored"),
    ]
    fig.legend(handles=handles, loc="lower left",
               bbox_to_anchor=(0.10, 0.925, 0.85, 0.05), ncol=3,
               handlelength=1.4, columnspacing=1.2)
    _save(fig, "fig_tradeoff")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 4
def fig_calibration() -> None:
    recipes = ["none", "bias_shift", "affine", "gd_full_1step", "anchored_adam", "trigger_aware"]
    labels = ["none", "bias", "affine", "GD", "anch.", "trig."]
    panels = [
        ("raw_dampen_k10_a3", "raw ranking, $k{=}10$"),
        ("z_dampen_k10_a3", "z ranking, $k{=}10$"),
        ("raw_dampen_k20_a3", "raw ranking, $k{=}20$"),
        ("z_dampen_k20_a3", "z ranking, $k{=}20$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.9), sharey=True)
    fig.subplots_adjust(top=0.70, bottom=0.22, left=0.07, right=0.985, wspace=0.12)

    x = np.arange(len(recipes))
    for ax, (key, title) in zip(axes, panels):
        rows = E10[key]
        asr = [100 * rows[r]["asr"] for r in recipes]
        broken = [rows[r]["clean_mse"] > 1.0 for r in recipes]
        colors = [GRAY if b else BLUE for b in broken]
        ax.bar(x, asr, color=colors, width=0.62, linewidth=0)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        mse = [rows[r]["clean_mse"] for r in recipes]
        ax2.plot(x, mse, linestyle="--", marker="o", markersize=2.6,
                 linewidth=0.8, color=RED)
        ax2.set_yscale("log")  # MSE values span 0.26-62, strictly positive
        ax2.set_ylim(0.2, 100)
        ax2.tick_params(axis="y", colors=RED, labelsize=6)
        if ax is axes[-1]:
            ax2.set_ylabel("clean MSE (log)", color=RED, fontsize=7)
        else:
            ax2.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=7.5, pad=5)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("ASR (%)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=GRAY),
        plt.Line2D([], [], color=RED, linestyle="--", marker="o", markersize=3, linewidth=0.8),
    ]
    fig.legend(
        handles, ["ASR (head intact)", "ASR (head destroyed, MSE>1)", "clean MSE, log scale (right)"],
        loc="lower left", bbox_to_anchor=(0.06, 0.90, 0.90, 0.06), ncol=3,
        handlelength=1.5, columnspacing=1.0,
    )
    _save(fig, "fig_calibration")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 5
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

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.4, 3.6))
    fig.subplots_adjust(hspace=0.62, top=0.85, bottom=0.11, left=0.13, right=0.97)

    for ax, model, title in (
        (ax_a, head, f"(a) backdoored  (GMM AUC {E5['backdoored']['gmm_auc']:.2f})"),
        (ax_b, mitigated, f"(b) mitigated  (GMM AUC {E5['mitigated_variants']['dampen_k10_a02']['gmm_auc']:.2f})"),
    ):
        trig = predict_scores(model, data.trig_test)
        clean = predict_scores(model, clean24)
        ax.hist(clean, bins=12, range=(1, 6), color="white", edgecolor=BLUE,
                linewidth=0.9, hatch="////", label="clean")
        ax.hist(trig, bins=12, range=(1, 6), color=RED, linewidth=0,
                alpha=0.75, label="triggered")
        ax.axvline(5.0, color=BLACK, lw=0.7, ls="--")
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("predicted MOS")
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_ylim(0, 11)
    ax_a.set_ylabel("clips")
    ax_b.set_ylabel("clips")
    # one shared legend above the whole figure, never inside either panel
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=BLUE, hatch="////"),
        plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor=RED, alpha=0.75),
        plt.Line2D([], [], color=BLACK, lw=0.7, ls="--"),
    ]
    fig.legend(
        handles, ["clean (24)", "triggered (24)", "target $y_t$"],
        loc="lower left", bbox_to_anchor=(0.10, 0.925, 0.85, 0.05), ncol=3,
        handlelength=1.4, columnspacing=1.0,
    )
    _save(fig, "fig_closedloop")
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
