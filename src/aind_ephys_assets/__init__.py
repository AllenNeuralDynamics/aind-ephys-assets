"""AIND ephys asset layout knowledge and helpers."""

from importlib.metadata import PackageNotFoundError, version

from aind_ephys_assets.discovery import (
    AnalyzerSpec,
    RawStreamSpec,
    discover_analyzers,
    discover_raw_streams,
    find_nwb,
    find_raw_zarr,
)
from aind_ephys_assets.names import (
    extract_device_name,
    match_analyzer_to_raw,
    sanitize_stream_name,
)
from aind_ephys_assets.paths import resolve_raw_root, resolve_sort_root, split_s3_url
from aind_ephys_assets.zarr_io import SIHeader, open_raw_traces, read_si_zarr_header

try:
    __version__ = version("aind-ephys-assets")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "AnalyzerSpec",
    "RawStreamSpec",
    "SIHeader",
    "__version__",
    "discover_analyzers",
    "discover_raw_streams",
    "extract_device_name",
    "find_nwb",
    "find_raw_zarr",
    "match_analyzer_to_raw",
    "open_raw_traces",
    "read_si_zarr_header",
    "resolve_raw_root",
    "resolve_sort_root",
    "sanitize_stream_name",
    "split_s3_url",
]
