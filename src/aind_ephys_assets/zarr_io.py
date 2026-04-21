"""Open SpikeInterface-format v2 zarrs on local disk or S3.

AIND raw ephys streams are stored as SI-format v2 zarrs: a top-level
group carrying ``sampling_frequency`` as a group attr and a
``traces_seg0`` array (``(n_samples, n_channels)``) holding the traces.
This module provides a header reader for enumeration (``None`` on
unreadable stores) and a strict opener for consumption (propagates
errors).

Supports both ``zarr>=3`` (which reads v2-format data via the
``zarr_format=2`` kwarg on ``open_group``) and ``zarr>=2.16,<3`` (which
only reads v2-format data, and doesn't have the newer
``storage_options`` / ``zarr_format`` kwargs). Feature detection lives
in :func:`_open_group_r`; call sites just use that helper.

Codec-level perf tuning — e.g. ``traces.compressor.num_decoding_threads``
for wavpack — is deliberately *not* applied here. It's a function of how
the caller intends to consume the traces, not of the asset layout.
Apply it on the returned array in the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import zarr

__all__ = ["SIHeader", "open_raw_traces", "read_si_zarr_header"]


# zarr 3.x changed the open_group signature (added ``zarr_format`` and
# ``storage_options`` kwargs); on 2.x, neither kwarg exists and remote
# URIs must be wrapped explicitly via ``zarr.storage.FSStore``. We pick
# the right path at import time so the hot read path doesn't re-check.
_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _open_group_r(
    source: str,
    storage_options: dict[str, Any] | None,
) -> Any:
    """Open a v2-format SI zarr for reading, portable across zarr 2.x and 3.x.

    On zarr 3.x, passes ``zarr_format=2`` and forwards ``storage_options``
    directly. On zarr 2.x (no ``zarr_format`` kwarg; reads v2 natively),
    wraps ``s3://`` / ``gs://`` / ``http(s)://`` URIs (and any local path
    with storage_options) in an explicit ``FSStore`` so ``storage_options``
    reaches fsspec.
    """
    if _ZARR_V3:
        return zarr.open_group(
            source,
            mode="r",
            zarr_format=2,
            storage_options=storage_options,
        )
    # zarr 2.x: no zarr_format kwarg (only reads v2 format anyway), and
    # storage_options must go through FSStore rather than open_group.
    is_uri = source.startswith(("s3://", "gs://", "az://", "http://", "https://"))
    if is_uri or storage_options:
        store: Any = zarr.storage.FSStore(source, **(storage_options or {}))
        return zarr.open_group(store=store, mode="r")
    return zarr.open_group(source, mode="r")


@dataclass(frozen=True, slots=True)
class SIHeader:
    """Header metadata for a SpikeInterface-format v2 zarr store."""

    sampling_frequency: float
    n_samples: int
    n_channels: int


def _as_store_arg(source: str | Path) -> str:
    """Return *source* as the string form zarr / fsspec accept.

    ``Path`` is always local — ``str(Path(...))`` is safe. ``str`` inputs
    (local paths or ``s3://`` URIs) pass through unchanged. Crucially we
    do **not** run ``s3://`` inputs through ``Path``: ``str(Path("s3://a/b"))``
    collapses to ``"s3:/a/b"`` on POSIX.
    """
    return str(source)


def read_si_zarr_header(
    source: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
) -> SIHeader | None:
    """Open an SI-format v2 zarr and return its header, or ``None`` on failure.

    Intended for enumerating streams under an asset — callers iterate
    directory entries and use a ``None`` return to silently skip anything
    that isn't a readable SI store (missing ``.zgroup``, missing
    ``traces_seg0``, permissions, etc.). For opening a store that is
    known to be valid, use :func:`open_raw_traces` instead.

    Parameters
    ----------
    source : str or Path
        Local directory path or ``s3://bucket/key`` URI to the zarr.
    storage_options : dict, optional
        Passed to fsspec when *source* is a URI (e.g. ``{"anon": True}``
        for public AIND Open Data buckets). Ignored for local paths.

    Returns
    -------
    SIHeader or None
        Populated header, or ``None`` if the store cannot be opened /
        doesn't look like SI format. Missing ``sampling_frequency`` attr
        is treated as ``0.0`` (matches capsule behavior; callers filter
        AP vs LFP by threshold so zero is a safe "unknown" sentinel).
    """
    try:
        grp = _open_group_r(_as_store_arg(source), storage_options)
        # zarr's attrs.get and grp[key] are typed with permissive unions
        # (JSON-like / Array|Group); the SI format contract narrows them
        # to "number" and "Array", so casts are safe here.
        sf = float(cast(Any, grp.attrs.get("sampling_frequency", 0)))
        traces = cast(Any, grp["traces_seg0"])
        n_samples, n_channels = traces.shape
    except Exception:
        return None
    return SIHeader(
        sampling_frequency=sf,
        n_samples=int(n_samples),
        n_channels=int(n_channels),
    )


def open_raw_traces(
    source: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    """Open an SI-format v2 zarr and return ``(traces_seg0, sampling_frequency)``.

    Strict opener — propagates any error from zarr / fsspec rather than
    swallowing it. Use this once you've already decided to consume a
    particular stream (e.g. after :func:`read_si_zarr_header` selected
    it during enumeration).

    The returned ``traces_seg0`` is the live zarr array handle:
    ``traces[t0:t1, :]`` reads a window without materializing the full
    recording. Codec-level tuning (e.g. wavpack
    ``num_decoding_threads``) is the caller's responsibility — mutate
    ``traces.compressor`` directly; the mutation propagates because
    zarr reuses one codec instance across chunk reads.

    Parameters
    ----------
    source : str or Path
        Local directory path or ``s3://bucket/key`` URI to the zarr.
    storage_options : dict, optional
        Passed to fsspec when *source* is a URI.

    Returns
    -------
    tuple
        ``(traces_seg0_array, sampling_frequency)``.
    """
    grp = _open_group_r(_as_store_arg(source), storage_options)
    traces = cast(Any, grp["traces_seg0"])
    sf = float(cast(Any, grp.attrs["sampling_frequency"]))
    return traces, sf
