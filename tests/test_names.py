"""Tests for aind_ephys_assets.names."""

from __future__ import annotations

import pytest

from aind_ephys_assets.names import (
    extract_device_name,
    match_analyzer_to_raw,
    sanitize_stream_name,
)

RAW_STREAM = "experiment1_Record Node 111#Neuropix-PXI-100.50213-1"
ANALYZER = f"{RAW_STREAM}_recording1"


class TestSanitizeStreamName:
    def test_real_aind_stream_name(self):
        assert sanitize_stream_name(RAW_STREAM) == ("experiment1_Record_Node_111_Neuropix-PXI-100.50213-1")

    def test_preserves_alnum_dot_underscore_dash(self):
        assert sanitize_stream_name("a-b_c.d") == "a-b_c.d"

    def test_collapses_runs_of_bad_chars(self):
        assert sanitize_stream_name("a   #  b") == "a_b"

    def test_strips_leading_trailing_underscores(self):
        assert sanitize_stream_name("# name #") == "name"

    def test_empty_string(self):
        assert sanitize_stream_name("") == ""

    def test_all_bad_chars(self):
        assert sanitize_stream_name("###") == ""


class TestMatchAnalyzerToRaw:
    def test_strips_recording_with_zarr_suffix(self):
        assert match_analyzer_to_raw(f"{ANALYZER}.zarr") == f"{RAW_STREAM}.zarr"

    def test_strips_recording_without_zarr_suffix(self):
        assert match_analyzer_to_raw(ANALYZER) == RAW_STREAM

    def test_idempotent_when_no_recording_suffix(self):
        assert match_analyzer_to_raw(RAW_STREAM) == RAW_STREAM
        assert match_analyzer_to_raw(f"{RAW_STREAM}.zarr") == f"{RAW_STREAM}.zarr"

    def test_handles_multi_digit_recording_number(self):
        assert match_analyzer_to_raw(f"{RAW_STREAM}_recording42.zarr") == (f"{RAW_STREAM}.zarr")

    def test_does_not_strip_recording_midstring(self):
        # A ``_recording<N>`` that is not at end-of-name (or before
        # ``.zarr$``) is data, not a suffix, and must be preserved.
        name = "foo_recording1_bar"
        assert match_analyzer_to_raw(name) == name

    def test_does_not_strip_without_number(self):
        assert match_analyzer_to_raw("foo_recording") == "foo_recording"


class TestExtractDeviceName:
    def test_real_analyzer_name(self):
        assert extract_device_name(ANALYZER) == "50213-1"

    def test_multi_part_serial(self):
        name = "experiment1_Record Node 111#Neuropix-PXI-100.50213-3_recording1"
        assert extract_device_name(name) == "50213-3"

    def test_raises_on_ni_daqmx_stream(self):
        with pytest.raises(ValueError, match="cannot extract device name"):
            extract_device_name("experiment1_Record Node 111#NI-DAQmx-103.PXIe-6341")

    def test_raises_on_unrelated_string(self):
        with pytest.raises(ValueError):
            extract_device_name("not a recording name")

    def test_different_record_node_number(self):
        name = "experiment1_Record Node 104#Neuropix-PXI-100.12345-0_recording1"
        assert extract_device_name(name) == "12345-0"
