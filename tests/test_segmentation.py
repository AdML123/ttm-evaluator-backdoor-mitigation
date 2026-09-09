import numpy as np

from src.features.segmentation import make_segments, materialize_segment


def test_primary_window_has_four_segments_for_average_clip():
    segments = make_segments(duration_s=21.8, window_s=11.0, overlap=0.5)

    assert len(segments) == 4
    assert segments[-1].end_s == 21.8
    assert segments[-1].pad_right_s == 5.7


def test_short_clip_uses_one_tail_aligned_segment():
    segments = make_segments(duration_s=8.0, window_s=11.0, overlap=0.5)

    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].pad_right_s == 3.0


def test_segment_plan_is_reproducible():
    assert make_segments(21.8, 11.0, 0.5) == make_segments(21.8, 11.0, 0.5)


def test_materialize_segment_pads_without_changing_window_length():
    waveform = np.arange(8, dtype=np.float32)
    segment = make_segments(8.0, 11.0, 0.5, sample_rate_hz=1)[0]

    result = materialize_segment(waveform, segment)

    np.testing.assert_array_equal(result, np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0], dtype=np.float32))
