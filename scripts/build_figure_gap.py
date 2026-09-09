"""检测缺口图（fig_gap.pdf）：GMM vs MC Dropout 在不同攻击强度下的 AUC。

数据来自 results/p0/ 下 mc_dropout*.json（与正文 Table II 一致）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "fig_gap.pdf"

BLUE = "#0072BD"
ORANGE = "#D95319"
GRAY = "#666666"


def _load() -> tuple[list[str], list[float], list[float]]:
    p0 = ROOT / "results" / "p0"

    def rd(name):
        return json.loads((p0 / name).read_text(encoding="utf-8"))

    strong = rd("mc_dropout.json")
    weak = rd("mc_dropout_weak.json")
    multi = rd("mc_dropout_multitarget.json")
    lowrho = rd("mc_dropout_lowrho.json")

    # weak 文件内按 target 升序；lowrho 按 rho
    weak_by_target = {w["target"]: w for w in weak}
    lowrho_by_rho = {w["rho"]: w for w in lowrho}
    settings = [
        "Strong\n(5.0)",
        "Weak\n(4.0)",
        "Weak\n(3.5)",
        "Weak\n(3.0)",
        "Multi-\ntarget",
        "Low $\\rho$\n0.5%",
        "Low $\\rho$\n1%",
    ]
    gmm = [
        strong["gmm_auc"],
        weak_by_target[4.0]["gmm_auc"],
        weak_by_target[3.5]["gmm_auc"],
        weak_by_target[3.0]["gmm_auc"],
        multi["gmm_auc"],
        lowrho_by_rho[0.005]["gmm_auc"],
        lowrho_by_rho[0.01]["gmm_auc"],
    ]
    mc = [
        strong["mc_dropout_auc"],
        weak_by_target[4.0]["mc_dropout_auc"],
        weak_by_target[3.5]["mc_dropout_auc"],
        weak_by_target[3.0]["mc_dropout_auc"],
        multi["mc_dropout_auc"],
        lowrho_by_rho[0.005]["mc_dropout_auc"],
        lowrho_by_rho[0.01]["mc_dropout_auc"],
    ]
    return settings, gmm, mc


def main() -> int:
    settings, gmm, mc = _load()
    x = np.arange(len(settings))
    width = 0.36

    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.15)

    ax.bar(x - width / 2, gmm, width, label="GMM", color=BLUE, alpha=0.7, edgecolor="white", linewidth=0.5, hatch="//")
    ax.bar(x + width / 2, mc, width, label="MC dropout", color=ORANGE, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=0.8)

    for i in range(len(settings)):
        ax.text(x[i] - width / 2, gmm[i] + 0.02, f"{gmm[i]:.2f}", ha="center", va="bottom", fontsize=4.5, color="#333333")
        ax.text(x[i] + width / 2, mc[i] + 0.02, f"{mc[i]:.2f}", ha="center", va="bottom", fontsize=4.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=5.5, color="black")
    ax.set_ylabel("AUC", fontsize=6, color=GRAY)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5\n(chance)", "1.0"], fontsize=5.5, color=GRAY)
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
