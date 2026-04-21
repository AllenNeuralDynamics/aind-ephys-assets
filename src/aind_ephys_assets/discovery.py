"""Enumerate AIND raw streams and sorted analyzers.

Composes the pure helpers in :mod:`aind_ephys_assets.names`,
:mod:`aind_ephys_assets.paths`, and :mod:`aind_ephys_assets.zarr_io` to
produce the layout-aware stream / analyzer inventories that capsules
need up front.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aind_ephys_assets.names import match_analyzer_to_raw
from aind_ephys_assets.paths import (
    _s3_filesystem,
    resolve_raw_root,
    resolve_sort_root,
    split_s3_url,
)
from aind_ephys_assets.zarr_io import read_si_zarr_header

__all__ = [
    "AnalyzerSpec",
    "RawStreamSpec",
    "discover_analyzers",
    "discover_raw_streams",
    "find_raw_zarr",
]


@dataclass(frozen=True, slots=True)
class RawStreamSpec:
    """One raw AP/LFP stream under an AIND raw asset.

    ``source`` is a string so local paths and ``s3://`` URIs share the
    same type. See CLAUDE.md for why this is normalized to ``str``.
    """

    name: str
    source: str
    sampling_frequency: float
    n_samples: int
    n_channels: int


@dataclass(frozen=True, slots=True)
class AnalyzerSpec:
    """One SortingAnalyzer zarr under ``postprocessed/``.

    ``raw_path`` is constructed by joining ``raw_root`` + the analyzer's
    corresponding raw stream name — it is **not** checked for existence
    here. Callers that need verification should ``os.path.exists`` /
    ``fs.exists`` themselves.
    """

    name: str
    analyzer_path: str
    raw_stream_name: str
    raw_path: str


def discover_raw_streams(
    asset_path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
) -> list[RawStreamSpec]:
    """Enumerate per-stream zarrs under an AIND raw asset.

    Auto-descends into ``ecephys/ecephys_compressed/`` via
    :func:`resolve_raw_root`; reads each zarr's SI-format header via
    :func:`read_si_zarr_header`. Entries that can't be opened or aren't
    SI-format are silently skipped (logged by the caller if needed).

    Parameters
    ----------
    asset_path : str or Path
        Asset root or already-descended streams root. Local dir or
        ``s3://`` URI.
    storage_options : dict, optional
        Passed to fsspec for S3 access.

    Returns
    -------
    list[RawStreamSpec]
        Sorted by stream name. Empty list if the root has no readable
        SI-format zarrs (not an error — may happen on probe-less assets).

    Raises
    ------
    FileNotFoundError
        Local: if ``asset_path`` (after resolution) isn't a directory.
        S3: if the prefix doesn't exist (fsspec's error, not wrapped).
    """
    root = resolve_raw_root(asset_path, storage_options=storage_options)
    if root.startswith("s3://"):
        return _discover_raw_s3(root, storage_options)
    return _discover_raw_local(Path(root))


def _discover_raw_local(streams_root: Path) -> list[RawStreamSpec]:
    if not streams_root.is_dir():
        msg = f"raw streams root is not a directory: {streams_root}"
        raise FileNotFoundError(msg)
    out: list[RawStreamSpec] = []
    for entry in sorted(streams_root.iterdir()):
        if not entry.is_dir() or not (entry / ".zgroup").exists():
            continue
        header = read_si_zarr_header(entry)
        if header is None:
            continue
        out.append(
            RawStreamSpec(
                name=entry.name,
                source=str(entry),
                sampling_frequency=header.sampling_frequency,
                n_samples=header.n_samples,
                n_channels=header.n_channels,
            )
        )
    return out


def _discover_raw_s3(
    streams_root: str,
    storage_options: dict[str, Any] | None,
) -> list[RawStreamSpec]:
    fs = _s3_filesystem(storage_options)
    bucket, key = split_s3_url(streams_root)
    prefix = f"{bucket}/{key}" if key else bucket
    out: list[RawStreamSpec] = []
    for entry in fs.ls(prefix, detail=True):
        if entry.get("type") != "directory":
            continue
        entry_key = entry["name"].rstrip("/")
        name = entry_key.rsplit("/", 1)[-1]
        stream_url = f"s3://{entry_key}"
        if not fs.exists(f"{entry_key}/.zgroup"):
            continue
        header = read_si_zarr_header(stream_url, storage_options=storage_options)
        if header is None:
            continue
        out.append(
            RawStreamSpec(
                name=name,
                source=stream_url,
                sampling_frequency=header.sampling_frequency,
                n_samples=header.n_samples,
                n_channels=header.n_channels,
            )
        )
    out.sort(key=lambda s: s.name)
    return out


def discover_analyzers(
    sort_path: str | Path,
    *,
    raw_root: str | Path,
    storage_options: dict[str, Any] | None = None,
) -> list[AnalyzerSpec]:
    """Enumerate SortingAnalyzer zarrs under ``<sort_root>/postprocessed/``.

    Auto-descends via :func:`resolve_sort_root`. ``raw_root`` is used
    only to construct ``AnalyzerSpec.raw_path`` — no existence check is
    performed, so the caller can point at an S3 raw asset without any
    S3 round-trip from this function.

    Parameters
    ----------
    sort_path : str or Path
        Sort asset root or its parent (pipeline-mount layout).
    raw_root : str or Path
        Path or URI that directly contains the raw per-stream zarrs
        (i.e. already resolved via :func:`resolve_raw_root` or equivalent).
    storage_options : dict, optional
        Currently unused — reserved for future S3-hosted sort assets.

    Returns
    -------
    list[AnalyzerSpec]
        Sorted by analyzer name.

    Raises
    ------
    FileNotFoundError
        If the resolved sort root has no ``postprocessed/`` directory.
    """
    del storage_options  # Reserved — sort assets are local in practice.
    resolved = Path(resolve_sort_root(sort_path))
    postprocessed = resolved / "postprocessed"
    if not postprocessed.is_dir():
        msg = f"sort asset has no postprocessed/ dir: {postprocessed}"
        raise FileNotFoundError(msg)

    raw_root_str = str(raw_root).rstrip("/")
    out: list[AnalyzerSpec] = []
    for entry in sorted(postprocessed.iterdir()):
        if entry.suffix != ".zarr" or not entry.is_dir():
            continue
        name = entry.stem  # without .zarr
        raw_stream_name = match_analyzer_to_raw(name)
        raw_path = f"{raw_root_str}/{raw_stream_name}.zarr"
        out.append(
            AnalyzerSpec(
                name=name,
                analyzer_path=str(entry),
                raw_stream_name=raw_stream_name,
                raw_path=raw_path,
            )
        )
    return out


def find_raw_zarr(data_dir: str | Path, recording_name: str) -> Path | None:
    """Locate the raw zarr for *recording_name* under *data_dir*.

    Differs from :func:`resolve_raw_root` + name concat: this walks
    *data_dir* looking for *any* nested ``ecephys_compressed/<name>.zarr``
    rather than assuming a single known asset at the root. This matches
    the pl-oversplitting capsule's need to find the raw zarr under a
    data_dir that may contain multiple mounted assets, or where the
    raw asset landed at an unexpected sub-path.

    Strips ``_recording<N>`` from *recording_name* to get the raw
    basename (AIND's pipeline drops the suffix between analyzer and raw
    zarr names). Walks following symlinks — Code Ocean pipeline workers
    stage mounts via ``/tmp`` symlinks that ``Path.rglob`` misses.

    Parameters
    ----------
    data_dir : str or Path
        Directory to search. Typically ``/data`` on Code Ocean.
    recording_name : str
        Either the bare recording name (``<name>_recording<N>``) or the
        zarr dir name (``<name>_recording<N>.zarr``). Suffix is stripped.

    Returns
    -------
    Path or None
        Path to the first matching ``ecephys_compressed/<basename>.zarr``,
        or ``None`` if not found.
    """
    basename = match_analyzer_to_raw(recording_name).removesuffix(".zarr")
    target = f"{basename}.zarr"
    root = Path(str(data_dir))
    for dirpath, dirs, _files in os.walk(root, followlinks=True):
        if Path(dirpath).name == "ecephys_compressed" and target in dirs:
            return Path(dirpath) / target
    return None
