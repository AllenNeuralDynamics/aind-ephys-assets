"""Tests for aind_ephys_assets.paths resolve_raw_root / resolve_sort_root.

S3 branch of resolve_raw_root is tested via a monkeypatched
``_s3_filesystem`` rather than real S3 / moto — keeps the test suite
fast and free of s3fs / moto dependencies. The logic being verified
(prefix construction, trailing-slash handling, nested-key existence
check) is local to this module; a fake fs with an ``exists`` method
covers it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aind_ephys_assets import paths
from aind_ephys_assets.paths import resolve_raw_root, resolve_sort_root


class _FakeS3FS:
    """Minimal fsspec-like fake that only implements ``exists``."""

    def __init__(self, existing: set[str]) -> None:
        self._existing = existing

    def exists(self, path: str) -> bool:
        return path in self._existing


def _patch_s3(monkeypatch: pytest.MonkeyPatch, existing: set[str]) -> None:
    monkeypatch.setattr(paths, "_s3_filesystem", lambda _opts: _FakeS3FS(existing))


class TestResolveRawRootLocal:
    def test_flat_layout_returns_input(self, tmp_path: Path):
        (tmp_path / "stream1.zarr").mkdir()
        assert resolve_raw_root(tmp_path) == str(tmp_path)

    def test_nested_layout_descends(self, tmp_path: Path):
        nested = tmp_path / "ecephys" / "ecephys_compressed"
        nested.mkdir(parents=True)
        assert resolve_raw_root(tmp_path) == str(nested)

    def test_already_at_streams_root_is_noop(self, tmp_path: Path):
        # If the caller passes the nested path itself, we don't double-descend.
        nested = tmp_path / "ecephys" / "ecephys_compressed"
        nested.mkdir(parents=True)
        assert resolve_raw_root(nested) == str(nested)

    def test_accepts_str_and_path(self, tmp_path: Path):
        (tmp_path / "ecephys" / "ecephys_compressed").mkdir(parents=True)
        assert resolve_raw_root(str(tmp_path)) == resolve_raw_root(tmp_path)

    def test_trailing_slash_stripped(self, tmp_path: Path):
        (tmp_path / "ecephys" / "ecephys_compressed").mkdir(parents=True)
        expected = str(tmp_path / "ecephys" / "ecephys_compressed")
        assert resolve_raw_root(f"{tmp_path}/") == expected

    def test_partial_nesting_does_not_descend(self, tmp_path: Path):
        # Only the ``ecephys/`` half of the suffix exists — must not
        # claim the nested layout.
        (tmp_path / "ecephys").mkdir()
        assert resolve_raw_root(tmp_path) == str(tmp_path)


class TestResolveRawRootS3:
    def test_nested_layout_descends(self, monkeypatch: pytest.MonkeyPatch):
        _patch_s3(monkeypatch, {"mybucket/asset/ecephys/ecephys_compressed"})
        assert resolve_raw_root("s3://mybucket/asset") == ("s3://mybucket/asset/ecephys/ecephys_compressed")

    def test_flat_layout_returns_input(self, monkeypatch: pytest.MonkeyPatch):
        _patch_s3(monkeypatch, set())
        assert resolve_raw_root("s3://mybucket/asset") == "s3://mybucket/asset"

    def test_trailing_slash_stripped(self, monkeypatch: pytest.MonkeyPatch):
        _patch_s3(monkeypatch, {"mybucket/asset/ecephys/ecephys_compressed"})
        assert resolve_raw_root("s3://mybucket/asset/") == ("s3://mybucket/asset/ecephys/ecephys_compressed")

    def test_bucket_root(self, monkeypatch: pytest.MonkeyPatch):
        # Asset at bucket root — key is empty, nested suffix goes directly.
        _patch_s3(monkeypatch, {"mybucket/ecephys/ecephys_compressed"})
        assert resolve_raw_root("s3://mybucket") == ("s3://mybucket/ecephys/ecephys_compressed")

    def test_storage_options_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        captured: list[dict | None] = []

        def fake_fs(opts: dict | None) -> _FakeS3FS:
            captured.append(opts)
            return _FakeS3FS(set())

        monkeypatch.setattr(paths, "_s3_filesystem", fake_fs)
        resolve_raw_root("s3://b/k", storage_options={"anon": True})
        assert captured == [{"anon": True}]


class TestS3FilesystemMissingExtra:
    def test_friendly_error_when_s3fs_missing(self, monkeypatch: pytest.MonkeyPatch):
        # Simulate fsspec being installed but s3fs not.  fsspec's own
        # ImportError for a missing protocol backend must be wrapped to
        # point users at the ``[s3]`` extra.
        import fsspec as _fsspec

        def fake_filesystem(protocol: str, **_kwargs: object) -> object:
            raise ImportError("Install s3fs to access S3")

        monkeypatch.setattr(_fsspec, "filesystem", fake_filesystem)
        with pytest.raises(ImportError, match=r"\[s3\] extra"):
            paths._s3_filesystem(None)


class TestResolveSortRoot:
    def test_root_has_postprocessed(self, tmp_path: Path):
        (tmp_path / "postprocessed").mkdir()
        assert resolve_sort_root(tmp_path) == str(tmp_path)

    def test_nested_one_level(self, tmp_path: Path):
        # Pipeline-mount layout: /data/sorted/<asset_name>/postprocessed/
        nested = tmp_path / "asset_name"
        (nested / "postprocessed").mkdir(parents=True)
        assert resolve_sort_root(tmp_path) == str(nested)

    def test_picks_first_child_with_postprocessed(self, tmp_path: Path):
        # If multiple children exist, pick the first (sorted) one that
        # has postprocessed/. Deterministic behavior matters — avoids
        # flaky tests and surprises on reruns.
        (tmp_path / "a").mkdir()  # no postprocessed
        (tmp_path / "b" / "postprocessed").mkdir(parents=True)
        (tmp_path / "c" / "postprocessed").mkdir(parents=True)
        assert resolve_sort_root(tmp_path) == str(tmp_path / "b")

    def test_returns_input_when_no_match(self, tmp_path: Path):
        # Neither layout applies — return input unchanged so the caller
        # gets a meaningful "no postprocessed/" failure with the path
        # they actually passed.
        (tmp_path / "random").mkdir()
        assert resolve_sort_root(tmp_path) == str(tmp_path)

    def test_returns_input_on_nonexistent_path(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        assert resolve_sort_root(missing) == str(missing)

    def test_accepts_str_and_path(self, tmp_path: Path):
        (tmp_path / "postprocessed").mkdir()
        assert resolve_sort_root(str(tmp_path)) == resolve_sort_root(tmp_path)
