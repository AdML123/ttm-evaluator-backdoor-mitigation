"""Deterministic duration-aware audio segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Segment:
    """One fixed-size model window and its source/padding boundaries."""

    clip_id: str | None
    start_sample: int
    end_sample: int
    pad_right_samples: int
    window_samples: int
    stride_samples: int
    sample_rate_hz: int

    @property
    def start_s(self) -> float:
        return self.start_sample / self.sample_rate_hz

    @property
    def end_s(self) -> float:
        return self.end_sample / self.sample_rate_hz

    @property
    def pad_right_s(self) -> float:
        return self.pad_right_samples / self.sample_rate_hz

    @property
    def requested_end_sample(self) -> int:
        return self.end_sample + self.pad_right_samples

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "pad_right_samples": self.pad_right_samples,
            "window_samples": self.window_samples,
            "stride_samples": self.stride_samples,
            "sample_rate_hz": self.sample_rate_hz,
        }


def _sample_count(seconds: float, sample_rate_hz: int) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration/window must be finite and non-negative")
    return int(round(seconds * sample_rate_hz))


def make_segments(
    duration_s: float,
    window_s: float,
    overlap: float,
    *,
    sample_rate_hz: int = 16000,
    clip_id: str | None = None,
) -> tuple[Segment, ...]:
    """Return a reproducible stride-grid plan.

    A clip shorter than the window gets one zero-padded window.  Otherwise the
    plan contains ``ceil(duration / stride)`` starts on the stride grid.  The
    final source interval is clipped at the true end and its right-padding is
    recorded explicitly.  This rule preserves the four-window protocol example
    for a 21.8-second clip (starts at 0, 5.5, 11.0, and 16.5 seconds).
    """

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    duration_samples = _sample_count(duration_s, sample_rate_hz)
    window_samples = _sample_count(window_s, sample_rate_hz)
    if duration_samples <= 0 or window_samples <= 0:
        raise ValueError("duration and window must be greater than zero")
    stride_samples = int(round(window_samples * (1.0 - overlap)))
    if stride_samples <= 0:
        raise ValueError("overlap leaves no positive stride")

    if duration_samples <= window_samples:
        starts = (0,)
    else:
        count = int(math.ceil(duration_samples / stride_samples))
        starts = tuple(index * stride_samples for index in range(count))

    segments: list[Segment] = []
    for start in starts:
        end = min(start + window_samples, duration_samples)
        # A stride-grid start is always inside the clip under the count rule.
        if start >= duration_samples or end <= start:
            raise RuntimeError("internal segmentation boundary error")
        pad = max(0, start + window_samples - duration_samples)
        segments.append(
            Segment(
                clip_id=clip_id,
                start_sample=start,
                end_sample=end,
                pad_right_samples=pad,
                window_samples=window_samples,
                stride_samples=stride_samples,
                sample_rate_hz=sample_rate_hz,
            )
        )
    return tuple(segments)


def materialize_segment(waveform: np.ndarray, segment: Segment) -> np.ndarray:
    """Extract a segment and zero-pad it to the encoder window length."""

    values = np.asarray(waveform)
    if values.ndim != 1:
        raise ValueError("waveform must be a one-dimensional array")
    if segment.end_sample > values.shape[0]:
        raise ValueError("segment end exceeds waveform length")
    output = np.zeros(segment.window_samples, dtype=values.dtype)
    source = values[segment.start_sample : segment.end_sample]
    output[: source.shape[0]] = source
    return output
