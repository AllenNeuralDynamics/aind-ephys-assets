"""Tests for aind_ephys_assets.motion's numpy-only helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aind_ephys_assets.motion import align_bins, motion_folder, read_motion

REC = "experiment1_Record Node 104#Neuropix-PXI-100.50213-1_recording1"


def write_motion(folder: Path, displacement, bins, depths) -> None:
    folder.mkdir(parents=True)
    np.save(folder / "displacement_seg0.npy", np.asarray(displacement, dtype=float))
    np.save(folder / "temporal_bins_s_seg0.npy", np.asarray(bins, dtype=float))
    np.save(folder / "spatial_bins_um.npy", np.asarray(depths, dtype=float))


@pytest.fixture
def sort_root(tmp_path):
    root = tmp_path / "mount" / "sorted_asset"
    (root / "postprocessed").mkdir(parents=True)
    disp = np.arange(12, dtype=float).reshape(4, 3)
    write_motion(root / "spikesorted/motion" / REC, disp, [1, 3, 5, 7], [100, 400, 700])
    write_motion(root / "preprocessed/motion" / REC / "motion", disp, [1, 3, 5, 7], [100, 400, 700])
    return root


class TestMotionFolder:
    def test_ks4(self):
        assert motion_folder("s3://b/sorted", REC) == f"s3://b/sorted/spikesorted/motion/{REC}"

    def test_dredge_fast(self):
        assert motion_folder("s3://b/sorted/", REC, "dredge_fast") == f"s3://b/sorted/preprocessed/motion/{REC}/motion"

    def test_strips_zarr_suffix(self):
        assert motion_folder("s3://b/s", f"{REC}.zarr").endswith(f"motion/{REC}")

    def test_descends_into_nested_local_sort(self, sort_root):
        assert motion_folder(sort_root.parent, REC) == f"{sort_root}/spikesorted/motion/{REC}"

    def test_unknown_source(self):
        with pytest.raises(ValueError, match="unknown motion source"):
            motion_folder("s3://b/s", REC, "kilosort")  # type: ignore[arg-type]


class TestReadMotion:
    def test_ks4_sign_flipped(self, sort_root):
        m = read_motion(sort_root, REC)
        np.testing.assert_array_equal(m.displacement, -np.arange(12).reshape(4, 3))
        assert m.source == "ks4"

    def test_dredge_fast_sign_kept(self, sort_root):
        m = read_motion(sort_root, REC, "dredge_fast")
        np.testing.assert_array_equal(m.displacement, np.arange(12).reshape(4, 3))

    def test_missing_estimate_does_not_fall_back(self, tmp_path):
        (tmp_path / "postprocessed").mkdir()
        write_motion(tmp_path / "preprocessed/motion" / REC / "motion", np.zeros((2, 1)), [0, 1], [0])
        with pytest.raises(FileNotFoundError):
            read_motion(tmp_path, REC)

    @pytest.mark.parametrize(
        ("disp", "bins", "depths", "match"),
        [
            (np.zeros((3, 2)), [0, 1], [0, 1], "does not match"),
            (np.full((2, 1), np.nan), [0, 1], [0], "non-finite"),
            (np.zeros((2, 2)), [1, 0], [0, 1], "increase strictly"),
            (np.zeros((2, 2)), [0, 1], [1, 0], "increase strictly"),
        ],
    )
    def test_rejects_bad_arrays(self, tmp_path, disp, bins, depths, match):
        write_motion(tmp_path / "spikesorted/motion" / REC, disp, bins, depths)
        with pytest.raises(ValueError, match=match):
            read_motion(tmp_path, REC)


class TestAlignBins:
    def test_recording_clock_unchanged(self):
        aligned, offset = align_bins([1001, 1003, 1005], 1000.0, 1006.0)
        np.testing.assert_array_equal(aligned, [1001, 1003, 1005])
        assert offset == 0.0

    def test_elapsed_time_shifted(self):
        aligned, offset = align_bins([1, 3, 5], 1000.0, 1006.0)
        np.testing.assert_array_equal(aligned, [1001, 1003, 1005])
        assert offset == 1000.0

    def test_other_block_rejected(self):
        with pytest.raises(ValueError, match="neither clock"):
            align_bins([1, 3, 5], 1000.0, 2000.0)

    def test_needs_two_bins(self):
        with pytest.raises(ValueError, match="two finite"):
            align_bins([1.0], 0.0, 2.0)
