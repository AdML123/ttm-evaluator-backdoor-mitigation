"""Focused artifact checks for the data-free Fig. 1 builder."""

from __future__ import annotations

import matplotlib.image as mpimg

from scripts.build_figure1 import build_figure


def test_figure1_builds_vector_pdf_and_grayscale_png(tmp_path):
    pdf_path, png_path = build_figure(tmp_path / "fig1_pipeline", dpi=120)

    assert pdf_path.suffix == ".pdf"
    assert png_path.suffix == ".png"
    assert pdf_path.read_bytes().startswith(b"%PDF-")

    image = mpimg.imread(png_path)
    assert image.ndim == 3
    assert image.shape[2] in (3, 4)
    assert image.shape[0] > 200 and image.shape[1] > 500
    # The schematic is deliberately monochrome.  Ignore alpha, if present.
    assert (image[..., :3].max(axis=2) - image[..., :3].min(axis=2)).max() < 1e-6
    assert image[..., :3].min() < 0.2
    assert image[..., :3].max() > 0.8

    pdf_text = pdf_path.read_bytes()
    assert b"D:\\paper49" not in pdf_text
    assert b"worktrees" not in pdf_text
