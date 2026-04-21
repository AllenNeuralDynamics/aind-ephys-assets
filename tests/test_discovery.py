"""Tests for aind_ephys_assets.discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aind_ephys_assets import discovery
from aind_ephys_assets.discovery import (
    AnalyzerSpec,
    discover_analyzers,
    discover_raw_streams,
    find_nwb,
    find_raw_zarr,
)
from aind_ephys_assets.zarr_io import SIHeader

from ._zarr_compat import create_array, open_group_w_v2


def _make_si_zarr(
    path: Path,
    *,
    sampling_frequency: float = 30_000.0,
    shape: tuple[int, int] = (1024, 384),
    dtype: str = "int16",
) -> None:
    grp = open_group_w_v2(str(path))
    grp.attrs["sampling_frequency"] = sampling_frequency
    arr = create_array(grp, "traces_seg0", shape=shape, dtype=dtype)
    arr[:] = np.zeros(shape, dtype=dtype)


def _make_analyzer_zarr(path: Path) -> None:
    """Analyzer zarrs don't carry ``sampling_frequency`` or ``traces_seg0`` at the
    root — they're SortingAnalyzer containers.  For discover_analyzers's
    purposes only the presence of ``.zarr`` dirs matters; we don't open
    them as SI stores."""
    grp = open_group_w_v2(str(path))
    grp.attrs["dummy"] = True


class TestDiscoverRawStreamsLocal:
    def test_flat_layout(self, tmp_path: Path):
        _make_si_zarr(tmp_path / "stream1.zarr")
        _make_si_zarr(tmp_path / "stream2.zarr", sampling_frequency=2500.0, shape=(100, 8))

        streams = discover_raw_streams(tmp_path)
        assert [s.name for s in streams] == ["stream1.zarr", "stream2.zarr"]
        assert streams[0].sampling_frequency == 30_000.0
        assert streams[0].n_samples == 1024
        assert streams[0].n_channels == 384
        assert streams[1].sampling_frequency == 2500.0
        assert streams[1].n_channels == 8

    def test_nested_layout_descends(self, tmp_path: Path):
        nested = tmp_path / "ecephys" / "ecephys_compressed"
        nested.mkdir(parents=True)
        _make_si_zarr(nested / "stream.zarr")
        streams = discover_raw_streams(tmp_path)
        assert len(streams) == 1
        assert streams[0].source == str(nested / "stream.zarr")

    def test_accepts_str_and_path(self, tmp_path: Path):
        _make_si_zarr(tmp_path / "stream.zarr")
        assert discover_raw_streams(str(tmp_path)) == discover_raw_streams(tmp_path)

    def test_skips_non_zarr_dirs(self, tmp_path: Path):
        _make_si_zarr(tmp_path / "good.zarr")
        (tmp_path / "random").mkdir()  # no .zgroup
        (tmp_path / "not-a-dir.zarr").touch()
        streams = discover_raw_streams(tmp_path)
        assert [s.name for s in streams] == ["good.zarr"]

    def test_skips_unreadable_si_stores(self, tmp_path: Path):
        _make_si_zarr(tmp_path / "good.zarr")
        # Build a zarr group without traces_seg0 — read_si_zarr_header
        # returns None; discovery silently skips so one corrupt store
        # doesn't break enumeration for the asset as a whole.
        bad = tmp_path / "bad.zarr"
        grp = open_group_w_v2(str(bad))
        grp.attrs["sampling_frequency"] = 30_000.0
        streams = discover_raw_streams(tmp_path)
        assert [s.name for s in streams] == ["good.zarr"]

    def test_empty_asset_returns_empty_list(self, tmp_path: Path):
        # Valid directory with no zarrs — return [] rather than raise.
        # "Asset exists but has no probes" is a legitimate state; the
        # failure mode we want to catch loudly is "asset path doesn't
        # exist" (that raises, see below).
        assert discover_raw_streams(tmp_path) == []

    def test_nonexistent_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="raw streams root"):
            discover_raw_streams(tmp_path / "does-not-exist")


class TestDiscoverRawStreamsS3:
    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        entries: list[dict],
        existing: set[str],
        header_for: dict[str, SIHeader | None] | None = None,
    ) -> None:
        """Install a fake fsspec filesystem + stub header reader.

        *header_for* optionally maps stream URL → header (or None for
        "unreadable"); when omitted, every URL returns the same default.
        """
        from aind_ephys_assets import paths

        class FakeFS:
            def ls(self, prefix: str, detail: bool = False) -> list[dict]:
                assert detail
                return entries

            def exists(self, path: str) -> bool:
                return path in existing

        monkeypatch.setattr(discovery, "_s3_filesystem", lambda _opts: FakeFS())
        monkeypatch.setattr(paths, "_s3_filesystem", lambda _opts: FakeFS())

        def fake_header(source: str, **_kwargs: object) -> SIHeader | None:
            if header_for is not None:
                return header_for.get(source)
            return SIHeader(sampling_frequency=30_000.0, n_samples=10, n_channels=2)

        monkeypatch.setattr(discovery, "read_si_zarr_header", fake_header)

    def test_s3_enumeration(self, monkeypatch: pytest.MonkeyPatch):
        # Cover the s3 branch end-to-end with a fake fsspec filesystem
        # and a stubbed header reader — isolates the branch's logic
        # (prefix construction, .zgroup check, skip-on-None-header)
        # without pulling in moto / aiobotocore.
        key_root = "mybucket/asset/ecephys/ecephys_compressed"
        self._setup(
            monkeypatch,
            entries=[
                {"name": f"{key_root}/stream1.zarr", "type": "directory"},
                {"name": f"{key_root}/stream2.zarr", "type": "directory"},
                {"name": f"{key_root}/README.txt", "type": "file"},
            ],
            existing={
                f"{key_root}/stream1.zarr/.zgroup",
                f"{key_root}/stream2.zarr/.zgroup",
                f"{key_root}",  # for resolve_raw_root's nested-layout check
            },
        )
        streams = discover_raw_streams("s3://mybucket/asset")
        assert [s.name for s in streams] == ["stream1.zarr", "stream2.zarr"]
        assert all(s.source.startswith("s3://mybucket/asset/") for s in streams)
        assert streams[0].n_samples == 10

    def test_s3_skips_dirs_without_zgroup(self, monkeypatch: pytest.MonkeyPatch):
        # A directory entry that isn't actually a zarr (no ``.zgroup``)
        # must be silently skipped — S3 assets occasionally contain
        # stray dirs (e.g. ``.compression/`` metadata drops).
        key_root = "mybucket/asset/ecephys/ecephys_compressed"
        self._setup(
            monkeypatch,
            entries=[
                {"name": f"{key_root}/stream.zarr", "type": "directory"},
                {"name": f"{key_root}/not_a_zarr_dir", "type": "directory"},
            ],
            existing={
                f"{key_root}/stream.zarr/.zgroup",
                f"{key_root}",
            },
        )
        streams = discover_raw_streams("s3://mybucket/asset")
        assert [s.name for s in streams] == ["stream.zarr"]

    def test_s3_skips_unreadable_streams(self, monkeypatch: pytest.MonkeyPatch):
        # Directory has ``.zgroup`` but the stored contents are
        # corrupt / not SI-format — ``read_si_zarr_header`` returns
        # None and discovery skips rather than raising.
        key_root = "mybucket/asset/ecephys/ecephys_compressed"
        self._setup(
            monkeypatch,
            entries=[
                {"name": f"{key_root}/good.zarr", "type": "directory"},
                {"name": f"{key_root}/broken.zarr", "type": "directory"},
            ],
            existing={
                f"{key_root}/good.zarr/.zgroup",
                f"{key_root}/broken.zarr/.zgroup",
                f"{key_root}",
            },
            header_for={
                f"s3://{key_root}/good.zarr": SIHeader(30_000.0, 10, 2),
                f"s3://{key_root}/broken.zarr": None,
            },
        )
        streams = discover_raw_streams("s3://mybucket/asset")
        assert [s.name for s in streams] == ["good.zarr"]


class TestDiscoverAnalyzers:
    def test_flat_layout(self, tmp_path: Path):
        pp = tmp_path / "postprocessed"
        pp.mkdir()
        _make_analyzer_zarr(pp / "expA_recording1.zarr")
        _make_analyzer_zarr(pp / "expB_recording1.zarr")

        analyzers = discover_analyzers(tmp_path, raw_root="/raw")
        assert [a.name for a in analyzers] == ["expA_recording1", "expB_recording1"]
        assert analyzers[0].analyzer_path == str(pp / "expA_recording1.zarr")
        assert analyzers[0].raw_stream_name == "expA"
        assert analyzers[0].raw_path == "/raw/expA.zarr"

    def test_nested_sort_layout(self, tmp_path: Path):
        nested = tmp_path / "asset_name"
        pp = nested / "postprocessed"
        pp.mkdir(parents=True)
        _make_analyzer_zarr(pp / "stream_recording1.zarr")
        analyzers = discover_analyzers(tmp_path, raw_root="/raw")
        assert len(analyzers) == 1
        assert analyzers[0].analyzer_path == str(pp / "stream_recording1.zarr")

    def test_raw_root_trailing_slash_stripped(self, tmp_path: Path):
        pp = tmp_path / "postprocessed"
        pp.mkdir()
        _make_analyzer_zarr(pp / "stream_recording1.zarr")
        analyzers = discover_analyzers(tmp_path, raw_root="/raw/")
        assert analyzers[0].raw_path == "/raw/stream.zarr"

    def test_s3_raw_root(self, tmp_path: Path):
        pp = tmp_path / "postprocessed"
        pp.mkdir()
        _make_analyzer_zarr(pp / "stream_recording1.zarr")
        analyzers = discover_analyzers(tmp_path, raw_root="s3://bucket/ecephys/ecephys_compressed")
        assert analyzers[0].raw_path == ("s3://bucket/ecephys/ecephys_compressed/stream.zarr")

    def test_accepts_str_and_path_for_both_args(self, tmp_path: Path):
        pp = tmp_path / "postprocessed"
        pp.mkdir()
        _make_analyzer_zarr(pp / "stream_recording1.zarr")
        a1 = discover_analyzers(tmp_path, raw_root=Path("/raw"))
        a2 = discover_analyzers(str(tmp_path), raw_root="/raw")
        assert a1 == a2

    def test_raises_when_no_postprocessed(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="postprocessed"):
            discover_analyzers(tmp_path, raw_root="/raw")

    def test_skips_non_zarr_dirs(self, tmp_path: Path):
        pp = tmp_path / "postprocessed"
        pp.mkdir()
        _make_analyzer_zarr(pp / "good_recording1.zarr")
        (pp / "random").mkdir()
        (pp / "log.txt").touch()
        analyzers = discover_analyzers(tmp_path, raw_root="/raw")
        assert [a.name for a in analyzers] == ["good_recording1"]

    def test_returns_analyzer_spec_dataclass(self, tmp_path: Path):
        pp = tmp_path / "postprocessed"
        pp.mkdir()
        _make_analyzer_zarr(pp / "stream_recording1.zarr")
        analyzers = discover_analyzers(tmp_path, raw_root="/raw")
        assert isinstance(analyzers[0], AnalyzerSpec)


class TestFindRawZarr:
    def test_finds_in_nested_ecephys_compressed(self, tmp_path: Path):
        raw = tmp_path / "raw_asset" / "ecephys" / "ecephys_compressed"
        raw.mkdir(parents=True)
        (raw / "stream.zarr").mkdir()
        found = find_raw_zarr(tmp_path, "stream_recording1")
        assert found == raw / "stream.zarr"

    def test_strips_recording_suffix(self, tmp_path: Path):
        raw = tmp_path / "ecephys_compressed"
        raw.mkdir()
        (raw / "experiment1_foo.zarr").mkdir()
        found = find_raw_zarr(tmp_path, "experiment1_foo_recording3")
        assert found == raw / "experiment1_foo.zarr"

    def test_accepts_name_with_zarr_suffix(self, tmp_path: Path):
        raw = tmp_path / "ecephys_compressed"
        raw.mkdir()
        (raw / "foo.zarr").mkdir()
        assert find_raw_zarr(tmp_path, "foo_recording1.zarr") == raw / "foo.zarr"

    def test_ignores_zarrs_outside_ecephys_compressed(self, tmp_path: Path):
        # The sort asset's ``postprocessed/<name>_recording1.zarr`` can
        # share the bare basename of the raw zarr — making sure we pin
        # the search to ``ecephys_compressed/`` is exactly the point of
        # this helper (pl-oversplitting's docstring calls this out).
        pp = tmp_path / "sorted" / "postprocessed"
        pp.mkdir(parents=True)
        (pp / "foo.zarr").mkdir()  # would be a false match without the filter
        assert find_raw_zarr(tmp_path, "foo_recording1") is None

    def test_returns_none_when_not_found(self, tmp_path: Path):
        assert find_raw_zarr(tmp_path, "missing_recording1") is None

    def test_follows_symlinks(self, tmp_path: Path):
        # CO pipeline workers stage mounts via symlinks; the helper
        # must traverse them or it misses the data.
        real = tmp_path / "real" / "ecephys_compressed"
        real.mkdir(parents=True)
        (real / "stream.zarr").mkdir()

        link_parent = tmp_path / "link_parent"
        link_parent.mkdir()
        (link_parent / "ecephys_compressed").symlink_to(real, target_is_directory=True)

        found = find_raw_zarr(link_parent, "stream_recording1")
        assert found is not None
        assert found.name == "stream.zarr"

    def test_accepts_str_data_dir(self, tmp_path: Path):
        raw = tmp_path / "ecephys_compressed"
        raw.mkdir()
        (raw / "foo.zarr").mkdir()
        assert find_raw_zarr(str(tmp_path), "foo_recording1") == raw / "foo.zarr"


class TestFindNwb:
    def test_flat_layout(self, tmp_path: Path):
        nwb = tmp_path / "session.nwb"
        nwb.write_bytes(b"not a real nwb, just a marker")
        assert find_nwb(tmp_path) == nwb

    def test_nested_layout(self, tmp_path: Path):
        nested = tmp_path / "asset" / "ecephys"
        nested.mkdir(parents=True)
        nwb = nested / "session.nwb"
        nwb.write_bytes(b"marker")
        assert find_nwb(tmp_path) == nwb

    def test_missing_returns_none(self, tmp_path: Path):
        # Empty dir, other files — still no NWB.
        (tmp_path / "notes.txt").write_text("hi")
        assert find_nwb(tmp_path) is None

    def test_follows_symlinks(self, tmp_path: Path):
        # Simulate CO's pipeline staging, which bind-mounts assets under
        # /tmp and symlinks them into /data. Without followlinks, a naive
        # recursion would miss the NWB entirely.
        real = tmp_path / "real_asset"
        real.mkdir()
        nwb = real / "session.nwb"
        nwb.write_bytes(b"marker")
        link = tmp_path / "data" / "asset_link"
        link.parent.mkdir()
        link.symlink_to(real, target_is_directory=True)
        found = find_nwb(tmp_path / "data")
        assert found is not None
        assert found.name == "session.nwb"

    def test_accepts_str(self, tmp_path: Path):
        nwb = tmp_path / "session.nwb"
        nwb.write_bytes(b"marker")
        assert find_nwb(str(tmp_path)) == nwb
