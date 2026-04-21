"""AIND ephys asset path resolution and S3-URL parsing.

All path-resolution functions accept ``str | Path`` and return canonical
``str``. Normalization is required because ``s3://`` URIs do not survive
a ``Path`` round-trip (``str(Path("s3://b/k")) == "s3:/b/k"`` on POSIX);
keeping one canonical type avoids inconsistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["resolve_raw_root", "resolve_sort_root", "split_s3_url"]

# Nested path that AIND places per-stream zarrs under, relative to the
# asset root. Both nested and flat layouts appear in practice.
_RAW_NESTED_SUFFIX = "ecephys/ecephys_compressed"


def split_s3_url(url: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URL into ``(bucket, key)``.

    Parameters
    ----------
    url : str
        An S3 URL beginning with ``s3://``. The bucket-only form
        ``"s3://bucket"`` (with or without a trailing slash) is
        accepted and returns an empty key.

    Returns
    -------
    tuple[str, str]
        ``(bucket, key)``. The key has any trailing ``/`` stripped so
        callers can safely ``f"{bucket}/{key}/..."`` without doubling up.

    Raises
    ------
    ValueError
        If *url* does not start with ``s3://``.
    """
    if not url.startswith("s3://"):
        msg = f"expected s3:// URL, got {url!r}"
        raise ValueError(msg)
    parts = url[len("s3://") :].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key.rstrip("/")


def _s3_filesystem(storage_options: dict[str, Any] | None) -> Any:
    """Return an ``s3`` fsspec filesystem, with a friendly error if s3fs missing.

    fsspec's own ``ImportError`` for a missing protocol backend is terse
    and doesn't mention our ``[s3]`` extra — wrap it so users see the
    install hint that makes the fix obvious.
    """
    import fsspec as _fsspec

    try:
        return _fsspec.filesystem("s3", **(storage_options or {}))
    except ImportError as exc:
        msg = (
            "s3fs is required for s3:// paths — install the [s3] extra, e.g. "
            "`pip install aind-ephys-assets[s3]` or `uv add 'aind-ephys-assets[s3]'`."
        )
        raise ImportError(msg) from exc


def resolve_raw_root(
    asset_path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
) -> str:
    """Return the directory directly containing per-stream ``.zarr`` dirs.

    AIND raw assets nest streams under ``ecephys/ecephys_compressed/``.
    Callers may pass either that nested path or the asset root — this
    auto-descends one level when the suffix exists, and is a no-op
    otherwise.

    Parameters
    ----------
    asset_path : str or Path
        Local directory path or ``s3://bucket/key`` URI to the raw asset.
    storage_options : dict, optional
        Passed to fsspec when *asset_path* is an S3 URI.

    Returns
    -------
    str
        Canonical path (local absolute-or-relative str, or ``s3://...``
        URI) to the streams root. Never ends with a trailing slash.
    """
    s = str(asset_path).rstrip("/")
    if s.startswith("s3://"):
        bucket, key = split_s3_url(s)
        nested_key = f"{key}/{_RAW_NESTED_SUFFIX}" if key else _RAW_NESTED_SUFFIX
        fs = _s3_filesystem(storage_options)
        if fs.exists(f"{bucket}/{nested_key}"):
            return f"s3://{bucket}/{nested_key}"
        return s
    local = Path(s)
    nested = local / _RAW_NESTED_SUFFIX
    return str(nested) if nested.is_dir() else s


def resolve_sort_root(sort_path: str | Path) -> str:
    """Return the directory that directly contains ``postprocessed/``.

    Sort assets live at one of two depths depending on how the asset was
    mounted: either the passed path *is* the sort root (it has
    ``postprocessed/`` as a direct child), or the sort asset is nested
    one level deeper (e.g. ``<asset_name>/postprocessed/``) as happens
    under pipeline mounts. Check the passed path first, then descend one
    level looking for the first child that has ``postprocessed/``; if
    neither layout matches, return the input unchanged (caller will see
    the "no postprocessed/" failure with a clear path).

    Parameters
    ----------
    sort_path : str or Path
        Local directory path — sort assets are not S3-hosted in practice.

    Returns
    -------
    str
        Canonical path (str, no trailing slash) to the resolved sort root.
    """
    p = Path(str(sort_path))
    if (p / "postprocessed").is_dir():
        return str(p)
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_dir() and (child / "postprocessed").is_dir():
                return str(child)
    return str(p)
