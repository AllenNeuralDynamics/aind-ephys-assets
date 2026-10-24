"""Tests for aind_ephys_assets.motion's SpikeInterface helpers.

Run with ``uv run --extra si --no-group zarr3 pytest tests/test_motion_si.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

si = pytest.importorskip("spikeinterface")
pytest.importorskip("scipy")

from spikeinterface.preprocessing import astype  # noqa: E402

from aind_ephys_assets.motion import correct_recording, load_motion  # noqa: E402
from tests.test_motion import REC, write_motion  # noqa: E402

DURATION_S, T0 = 60.0, 5000.0


@pytest.fixture
def recording():
    rec = si.generate_recording(num_channels=16, durations=[DURATION_S], seed=0)
    rec.shift_times(T0)
    return rec


def test_ks4_on_recording_clock(tmp_path, recording):
    bins = T0 + np.arange(1, DURATION_S, 2.0)
    write_motion(tmp_path / "spikesorted/motion" / REC, np.full((bins.size, 2), 7.0), bins, [0, 300])
    motion, info = load_motion(tmp_path, REC, recording)
    np.testing.assert_array_equal(motion.displacement[0], -7.0)
    np.testing.assert_array_equal(motion.temporal_bins_s[0], bins)
    assert info["time_offset_s"] == 0.0 and info["ks4_sign_flipped"]


def test_dredge_fast_elapsed_and_filtered(tmp_path, recording):
    bins = np.arange(1, DURATION_S, 2.0)
    disp = np.zeros((bins.size, 1))
    disp[10] = 50.0
    write_motion(tmp_path / "preprocessed/motion" / REC / "motion", disp, bins, [0])
    motion, info = load_motion(tmp_path, REC, recording, "dredge_fast")
    np.testing.assert_array_equal(motion.temporal_bins_s[0], bins + T0)
    np.testing.assert_array_equal(motion.displacement[0], 0.0)
    assert info["median_filter_bins"] == 15


def test_correct_int16_keeps_channels(tmp_path, recording):
    rec16 = astype(recording, "int16")
    bins = T0 + np.arange(1, DURATION_S, 2.0)
    write_motion(tmp_path / "spikesorted/motion" / REC, np.full((bins.size, 1), -10.0), bins, [0])
    motion, _ = load_motion(tmp_path, REC, rec16)
    corrected = correct_recording(rec16, motion)
    assert corrected.get_dtype() == np.float32
    assert list(corrected.channel_ids) == list(rec16.channel_ids)
    assert corrected.get_traces(start_frame=0, end_frame=100).shape == (100, 16)
