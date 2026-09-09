"""Build the black-and-white temporal aggregation schematic for Fig. 1.

The figure is intentionally data-free.  It can therefore be regenerated on a
machine that has only the public Python environment and does not disclose
dataset paths, cache names, or author metadata.  Both outputs are generated
from the same vector primitives: ``.pdf`` remains vector artwork while the
``.png`` is a high-resolution raster preview.

Example
-------
``python scripts/build_figure1.py --output-prefix submission/fig1_pipeline``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

# Select a non-interactive backend before importing pyplot.  This keeps the
# script usable in headless CI and on submission-build machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches


_BLACK = "#111111"
_MID = "#666666"
_LIGHT = "#F2F2F2"
_PALE = "#FAFAFA"
_WHITE = "#FFFFFF"


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str = _WHITE,
    hatch: str | None = None,
    fontsize: float = 8.2,
    linewidth: float = 0.9,
    radius: float = 0.04,
):
    """Draw a compact labelled box in axes coordinates."""

    artist = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=patches.BoxStyle("Round", pad=0.015, rounding_size=radius),
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=_BLACK,
        linewidth=linewidth,
        hatch=hatch,
        joinstyle="miter",
        zorder=2,
    )
    ax.add_patch(artist)
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=_BLACK,
        fontsize=fontsize,
        linespacing=1.05,
        zorder=3,
    )
    return artist


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *, dashed=False):
    """Draw a directional connector in axes coordinates."""

    style = (0, (2.0, 2.0)) if dashed else "-"
    arrow = patches.FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=0.85,
        linestyle=style,
        color=_BLACK,
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=1,
    )
    ax.add_patch(arrow)
    return arrow


def _connector_label(ax, x: float, y: float, label: str, *, fontsize: float = 6.8):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=_MID,
        fontsize=fontsize,
        zorder=4,
    )


def _draw_standard(ax) -> None:
    """Draw panel (a), the full-clip mean-pooling path."""

    ax.text(
        0.012,
        0.91,
        "(a) Standard full-clip evaluation",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=_BLACK,
    )

    y = 0.43
    h = 0.25
    boxes = [
        (0.03, 0.12, "full clip\n$\\mathbf{x}$", _PALE, None, 7.8),
        (0.22, 0.13, "audio\nencoder", _LIGHT, "///", 7.8),
        (0.41, 0.14, "frame states\n$\\mathbf{h}_1,\\ldots,\\mathbf{h}_L$", _PALE, None, 7.2),
        (0.61, 0.13, "temporal mean\n$L^{-1}\\sum_l\\mathbf{h}_l$", _LIGHT, "\\\\", 7.1),
        (0.80, 0.10, "MLP\n$g_\\phi$", _LIGHT, "...", 7.8),
    ]
    for x, width, label, face, hatch, fontsize in boxes:
        _box(ax, x, y, width, h, label, facecolor=face, hatch=hatch, fontsize=fontsize)
    for left, right in zip(boxes, boxes[1:]):
        _arrow(ax, (left[0] + left[1], y + h / 2), (right[0], y + h / 2))
    _arrow(ax, (0.89, y + h / 2), (0.915, y + h / 2))
    _box(ax, 0.915, y + 0.035, 0.075, 0.18, r"$\hat{y}_{\rm full}$", facecolor=_WHITE, fontsize=7.2, radius=0.025)

    _connector_label(ax, 0.125, y - 0.065, "one input")
    _connector_label(ax, 0.685, y - 0.065, "uniform over frames")
    ax.text(
        0.50,
        0.13,
        "A single embedding summarizes the entire waveform before score prediction.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        color=_MID,
    )


def _draw_proposed(ax) -> None:
    """Draw panel (b), segment scoring followed by soft-min pooling."""

    ax.text(
        0.012,
        0.91,
        "(b) Proposed segment-level soft-min evaluation",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=_BLACK,
    )

    # Input and segmenter.
    y = 0.45
    h = 0.22
    _box(ax, 0.025, y, 0.12, h, "full clip\n$\\mathbf{x}$", facecolor=_PALE, fontsize=7.8)
    _box(ax, 0.18, y, 0.14, h, "overlapping\nsegmenter", facecolor=_LIGHT, hatch="///", fontsize=7.3)
    _arrow(ax, (0.145, y + h / 2), (0.18, y + h / 2))

    # A compact stack makes the overlap explicit without implying a fixed N.
    stack_x = 0.36
    stack_y = 0.31
    seg_w = 0.145
    seg_h = 0.10
    for idx, (dy, hatch) in enumerate(((0.16, ""), (0.08, "\\\\"), (0.0, "///"))):
        _box(
            ax,
            stack_x,
            stack_y + dy,
            seg_w,
            seg_h,
            rf"segment $x_{{{idx + 1}}}$" if idx < 2 else r"$\cdots\;x_N$",
            facecolor=_WHITE,
            hatch=hatch or None,
            fontsize=7.1,
            radius=0.025,
            linewidth=0.8,
        )
    _arrow(ax, (0.32, y + h / 2), (stack_x, stack_y + 0.21))
    _connector_label(ax, 0.43, 0.18, r"$N$ windows, 50\% overlap")

    # Shared evaluator path represented once, with a brace-like label to make
    # clear that each segment is encoded and scored independently.
    eval_x = 0.54
    _box(ax, eval_x, 0.57, 0.14, 0.18, "shared\nencoder", facecolor=_LIGHT, hatch="///", fontsize=7.4)
    _box(ax, eval_x, 0.28, 0.14, 0.18, "shared\nMLP", facecolor=_LIGHT, hatch="...", fontsize=7.4)
    _arrow(ax, (stack_x + seg_w, stack_y + 0.21), (eval_x, 0.67), dashed=True)
    _arrow(ax, (eval_x + 0.07, 0.57), (eval_x + 0.07, 0.46))
    ax.text(
        eval_x + 0.165,
        0.51,
        "repeat for each segment",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.7,
        color=_MID,
        rotation=90,
    )

    _box(ax, 0.72, 0.43, 0.115, 0.18, "scores\n$s_1,\\ldots,s_N$", facecolor=_PALE, fontsize=7.5)
    _arrow(ax, (eval_x + 0.14, 0.37), (0.72, 0.52))
    _box(ax, 0.855, 0.43, 0.075, 0.18, "soft-min\n$\\tau$", facecolor=_LIGHT, hatch="\\\\", fontsize=7.2, radius=0.025)
    _arrow(ax, (0.835, 0.52), (0.855, 0.52))
    _arrow(ax, (0.93, 0.52), (0.94, 0.52))
    _box(ax, 0.94, 0.455, 0.045, 0.13, r"$\hat{y}$", facecolor=_WHITE, fontsize=7.2, radius=0.02)
    _connector_label(ax, 0.89, 0.34, r"$-\tau^{-1}\log\,\mathrm{mean}\,e^{-\tau s_i}$", fontsize=6.0)

    ax.text(
        0.50,
        0.10,
        "Low scores receive larger weights, preserving localized degradation.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        color=_MID,
    )


def build_figure(output_prefix: str | Path, *, dpi: int = 300) -> tuple[Path, Path]:
    """Render Fig. 1 and return ``(pdf_path, png_path)``.

    ``output_prefix`` may include a suffix; any suffix is replaced with
    ``.pdf`` and ``.png`` so callers can pass either a stem or a nominal file
    name.  Parent directories are created as needed.
    """

    prefix = Path(output_prefix)
    if prefix.suffix.lower() in {".pdf", ".png"}:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = prefix.with_suffix(".pdf")
    png_path = prefix.with_suffix(".png")

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.16, 3.35),
            gridspec_kw={"height_ratios": (1.0, 1.15)},
        )
        fig.patch.set_facecolor(_WHITE)
        for ax in axes:
            ax.set_xlim(0, 1.08)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_facecolor(_WHITE)
        _draw_standard(axes[0])
        _draw_proposed(axes[1])
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, hspace=0.02)

        metadata = {
            "Title": "Temporal aggregation in text-to-music evaluation",
            "Subject": "Standard mean pooling versus segment-level soft-min pooling",
            "Creator": "build_figure1.py",
        }
        # Keep a visible white border around the terminal output boxes when
        # the figure is embedded at the narrow IEEE column width.
        fig.savefig(pdf_path, format="pdf", metadata=metadata, bbox_inches="tight", pad_inches=0.12)
        fig.savefig(png_path, format="png", dpi=dpi, metadata=metadata, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)

    return pdf_path, png_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/figures/fig1_pipeline"),
        help="output stem; .pdf and .png are written next to it",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    pdf_path, png_path = build_figure(args.output_prefix, dpi=args.dpi)
    print(f"pdf={pdf_path}")
    print(f"png={png_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke test
    raise SystemExit(main())
