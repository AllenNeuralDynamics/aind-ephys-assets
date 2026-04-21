"""Tests for aind_ephys_assets.zarr_io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from aind_ephys_assets.zarr_io import (
    SIHeader,
    open_raw_traces,
    read_si_zarr_header,
)


def _make_si_zarr(
    path: Path,
    *,
    sampling_frequency: float | None = 30_000.0,
    shape: tuple[int, int] = (1024, 384),
    dtype: str = "int16",
    include_traces: bool = True,
) -> Path:
    """Write a hand-built SI-format v2 zarr under *path*.

    Mirrors the AIND layout: group root with ``sampling_frequency`` attr
    and a ``traces_seg0`` array of shape ``(n_samples, n_channels)``.

    Parameters
    ----------
    path : Path
        Directory path for the zarr store.
    sampling_frequency : float, optional
        If ``None``, the attr is omitted (simulates a malformed store).
    shape, dtype :
        Shape and dtype of ``traces_seg0``.
    include_traces : bool
        If False, the group has no ``traces_seg0`` (simulates a store
        that is a valid zarr group but not SI-format).
    """
    grp = zarr.open_group(str(path), mode="w", zarr_format=2)
    if sampling_frequency is not None:
        grp.attrs["sampling_frequency"] = sampling_frequency
    if include_traces:
        arr = grp.create_array("traces_seg0", shape=shape, dtype=dtype)
        arr[:] = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return path


class TestReadSIZarrHeader:
    def test_valid_store(self, tmp_path: Path):
        store = _make_si_zarr(tmp_path / "stream.zarr")
        header = read_si_zarr_header(store)
        assert header == SIHeader(sampling_frequency=30_000.0, n_samples=1024, n_channels=384)

    def test_accepts_str(self, tmp_path: Path):
        store = _make_si_zarr(tmp_path / "stream.zarr")
        header = read_si_zarr_header(str(store))
        assert header is not None
        assert header.sampling_frequency == 30_000.0

    def test_accepts_path(self, tmp_path: Path):
        store = _make_si_zarr(tmp_path / "stream.zarr")
        header = read_si_zarr_header(store)
        assert header is not None
        assert header.n_samples == 1024

    def test_nonexistent_path_returns_none(self, tmp_path: Path):
        assert read_si_zarr_header(tmp_path / "does-not-exist.zarr") is None

    def test_missing_traces_returns_none(self, tmp_path: Path):
        _make_si_zarr(tmp_path / "no-traces.zarr", include_traces=False)
        assert read_si_zarr_header(tmp_path / "no-traces.zarr") is None

    def test_missing_sampling_frequency_attr_is_zero(self, tmp_path: Path):
        # Missing sf is a "real" AIND edge case on analyzer zarrs — the
        # sort asset doesn't carry sf at root. Contract: return 0.0
        # rather than None, so the caller's AP-band filter ("sf >=
        # threshold OR sf == 0") still includes analyzers by default.
        _make_si_zarr(tmp_path / "no-sf.zarr", sampling_frequency=None)
        header = read_si_zarr_header(tmp_path / "no-sf.zarr")
        assert header is not None
        assert header.sampling_frequency == 0.0
        assert header.n_samples == 1024

    def test_not_a_zarr_returns_none(self, tmp_path: Path):
        bad = tmp_path / "not-a-zarr"
        bad.mkdir()
        (bad / "random.txt").write_text("hello")
        assert read_si_zarr_header(bad) is None

    def test_siheader_is_immutable(self):
        header = SIHeader(sampling_frequency=30_000.0, n_samples=10, n_channels=2)
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            header.sampling_frequency = 0.0  # type: ignore[misc]


class TestOpenRawTraces:
    def test_returns_array_and_sf(self, tmp_path: Path):
        store = _make_si_zarr(tmp_path / "stream.zarr")
        traces, sf = open_raw_traces(store)
        assert sf == 30_000.0
        assert traces.shape == (1024, 384)
        assert str(traces.dtype) == "int16"

    def test_traces_are_readable(self, tmp_path: Path):
        store = _make_si_zarr(tmp_path / "stream.zarr", shape=(100, 4), dtype="int16")
        traces, _ = open_raw_traces(store)
        window = np.asarray(traces[10:20, :])
        assert window.shape == (10, 4)
        assert window.dtype == np.int16

    def test_accepts_str_and_path(self, tmp_path: Path):
        store = _make_si_zarr(tmp_path / "stream.zarr")
        t1, sf1 = open_raw_traces(str(store))
        t2, sf2 = open_raw_traces(store)
        assert sf1 == sf2 == 30_000.0
        assert t1.shape == t2.shape

    def test_raises_on_nonexistent_path(self, tmp_path: Path):
        # open_raw_traces is the *strict* opener — unlike
        # read_si_zarr_header, it must propagate errors rather than
        # swallowing them, so callers using it as "I know this exists"
        # don't get silent None-returns for typos.
        with pytest.raises(Exception):
            open_raw_traces(tmp_path / "does-not-exist.zarr")

    def test_raises_on_missing_sampling_frequency(self, tmp_path: Path):
        # Differs from read_si_zarr_header (which treats missing sf as
        # 0.0) — strict opener: if the caller is about to *use* the sf,
        # absence is a bug.
        _make_si_zarr(tmp_path / "no-sf.zarr", sampling_frequency=None)
        with pytest.raises(KeyError):
            open_raw_traces(tmp_path / "no-sf.zarr")
