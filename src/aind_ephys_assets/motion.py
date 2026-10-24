"""Read the drift estimates saved in AIND sorted assets.

A sorted asset holds two estimates per recording, both SpikeInterface ``Motion`` folders:

- ``ks4``: ``spikesorted/motion/<recording>/``, the estimate Kilosort 4 applied before
  sorting. Its sign is the opposite of SpikeInterface's. Sorting capsules before
  ``cdef667`` (2026-04-16) did not save it.
- ``dredge_fast``: ``preprocessed/motion/<recording>/motion/``, computed in preprocessing
  and never applied. It has single-bin outliers of 20-80 um, so :func:`load_motion`
  median-filters it.

Time bins are on the recording's clock or on elapsed time, depending on the writer;
:func:`align_bins` accepts either. :func:`read_motion` needs only numpy;
:func:`load_motion` and :func:`correct_recording` need the ``[si]`` extra.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from aind_ephys_assets.paths import resolve_sort_root

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording
    from spikeinterface.core.motion import Motion

__all__ = [
    "MotionSource",
    "SavedMotion",
    "align_bins",
    "correct_recording",
    "load_motion",
    "motion_folder",
    "read_motion",
]

MotionSource = Literal["ks4", "dredge_fast"]

DREDGE_FAST_SMOOTH_S = 30.0


@dataclass(frozen=True, slots=True)
class SavedMotion:
    """One saved estimate, with ``displacement`` already in SpikeInterface's sign.

    ``displacement`` is ``(time bins, depth bins)`` in um; ``temporal_bins_s`` is as
    saved, not yet aligned to a recording.
    """

    source: MotionSource
    folder: str
    displacement: npt.NDArray[np.float64]
    temporal_bins_s: npt.NDArray[np.float64]
    spatial_bins_um: npt.NDArray[np.float64]


def motion_folder(sort_root: str | Path, recording_name: str, source: MotionSource = "ks4") -> str:
    """Return the ``Motion`` folder of *recording_name* in a sorted asset.

    Parameters
    ----------
    sort_root : str or Path
        Sorted asset root (local, or ``s3://``). Local paths go through
        :func:`~aind_ephys_assets.paths.resolve_sort_root`.
    recording_name : str
        The analyzer's name, e.g. ``experiment1_Record Node 104#Neuropix-PXI-100.50213-1_recording1``;
        a trailing ``.zarr`` is dropped.
    source : {"ks4", "dredge_fast"}
        Which estimate.

    Returns
    -------
    str
        Folder path or URI, not checked for existence.
    """
    root = str(sort_root).rstrip("/")
    if not root.startswith("s3://"):
        root = resolve_sort_root(root)
    name = recording_name.removesuffix(".zarr")
    if source == "ks4":
        return f"{root}/spikesorted/motion/{name}"
    if source == "dredge_fast":
        return f"{root}/preprocessed/motion/{name}/motion"
    msg = f"unknown motion source {source!r}"
    raise ValueError(msg)


def _load_npy(url: str, storage_options: dict[str, Any] | None) -> npt.NDArray[np.float64]:
    if url.startswith("s3://"):
        from aind_ephys_assets.paths import _s3_filesystem

        with _s3_filesystem(storage_options).open(url, "rb") as f:
            buf = io.BytesIO(f.read())
        return np.asarray(np.load(buf, allow_pickle=False), dtype=float)
    return np.asarray(np.load(url, allow_pickle=False), dtype=float)


def read_motion(
    sort_root: str | Path,
    recording_name: str,
    source: MotionSource = "ks4",
    *,
    storage_options: dict[str, Any] | None = None,
) -> SavedMotion:
    """Read and validate one saved estimate, flipping KS4's sign to SpikeInterface's.

    Parameters
    ----------
    sort_root : str or Path
        Sorted asset root (local, or ``s3://``).
    recording_name : str
        The analyzer's name; see :func:`motion_folder`.
    source : {"ks4", "dredge_fast"}
        Which estimate. There is no fallback: a sort without a saved KS4 estimate
        raises rather than silently returning the unapplied one.
    storage_options : dict, optional
        Passed to s3fs for ``s3://`` roots, e.g. ``{"anon": True}``.

    Returns
    -------
    SavedMotion
        Single-segment estimate.

    Raises
    ------
    FileNotFoundError
        If the estimate was not saved.
    ValueError
        If the arrays are inconsistent, non-finite or non-monotonic.
    """
    folder = motion_folder(sort_root, recording_name, source)
    displacement = _load_npy(f"{folder}/displacement_seg0.npy", storage_options)
    bins = _load_npy(f"{folder}/temporal_bins_s_seg0.npy", storage_options)
    depths = _load_npy(f"{folder}/spatial_bins_um.npy", storage_options)
    if displacement.shape != (bins.size, depths.size):
        msg = f"{folder}: displacement {displacement.shape} does not match {bins.size} time x {depths.size} depth bins"
        raise ValueError(msg)
    if not (np.isfinite(displacement).all() and np.isfinite(bins).all() and np.isfinite(depths).all()):
        msg = f"{folder}: motion contains non-finite values"
        raise ValueError(msg)
    if np.any(np.diff(bins) <= 0) or np.any(np.diff(depths) <= 0):
        msg = f"{folder}: time and depth bins must increase strictly"
        raise ValueError(msg)
    if source == "ks4":
        displacement = -displacement
    return SavedMotion(source, folder, displacement, bins, depths)


def align_bins(bins: npt.ArrayLike, start_s: float, end_s: float) -> tuple[npt.NDArray[np.float64], float]:
    """Put time bins on the recording's clock, whichever clock they were saved on.

    Parameters
    ----------
    bins : array_like
        Bin centers in s, on the recording's clock or as time since its start.
    start_s, end_s : float
        Times of the recording's first and last samples.

    Returns
    -------
    aligned : ndarray
        Bin centers on the recording's clock.
    offset_s : float
        What was added: 0, or *start_s*.

    Raises
    ------
    ValueError
        If neither clock spans the recording to within two bins, which means the
        estimate belongs to another recording or block.
    """
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all():
        msg = "motion needs at least two finite time bins"
        raise ValueError(msg)
    if np.any(np.diff(bins) <= 0):
        msg = "motion time bins must increase strictly"
        raise ValueError(msg)
    tolerance = 2 * float(np.max(np.diff(bins)))
    for offset in (0.0, start_s):
        aligned = bins + offset
        if max(abs(aligned[0] - start_s), abs(aligned[-1] - end_s)) <= tolerance:
            return aligned, offset
    msg = f"motion spans {bins[0]:.3f}..{bins[-1]:.3f} s, recording {start_s:.3f}..{end_s:.3f} s on neither clock"
    raise ValueError(msg)


def _require_si() -> None:
    try:
        import scipy  # noqa: F401
        import spikeinterface  # noqa: F401
    except ImportError as exc:
        msg = (
            "spikeinterface and scipy are required here — install the [si] extra, e.g. "
            "`uv add 'aind-ephys-assets[si]'`."
        )
        raise ImportError(msg) from exc


def load_motion(
    sort_root: str | Path,
    recording_name: str,
    recording: BaseRecording,
    source: MotionSource = "ks4",
    *,
    storage_options: dict[str, Any] | None = None,
) -> tuple[Motion, dict[str, Any]]:
    """Load a saved estimate as a SpikeInterface ``Motion`` aligned to *recording*.

    ``dredge_fast`` is median-filtered over 30 s against its outliers; KS4's is used
    as applied, glitches included.

    Parameters
    ----------
    sort_root : str or Path
        Sorted asset root (local, or ``s3://``).
    recording_name : str
        The analyzer's name; see :func:`motion_folder`.
    recording : BaseRecording
        Single-segment recording the estimate was made on; supplies the clock.
    source : {"ks4", "dredge_fast"}
        Which estimate.
    storage_options : dict, optional
        Passed to s3fs for ``s3://`` roots.

    Returns
    -------
    motion : Motion
        In SpikeInterface's sign, on the recording's clock.
    info : dict
        Provenance worth recording next to anything corrected with it.
    """
    _require_si()
    from scipy.ndimage import median_filter
    from spikeinterface.core.motion import Motion

    saved = read_motion(sort_root, recording_name, source, storage_options=storage_options)
    start = float(recording.sample_index_to_time(0))
    end = float(recording.sample_index_to_time(recording.get_num_samples() - 1))
    bins, offset = align_bins(saved.temporal_bins_s, start, end)
    displacement = saved.displacement
    smooth_bins = 1
    if source == "dredge_fast":
        smooth_bins = max(1, round(DREDGE_FAST_SMOOTH_S / float(np.median(np.diff(bins)))))
        displacement = median_filter(displacement, size=(smooth_bins, 1), mode="nearest")
    motion = Motion(displacement, bins, saved.spatial_bins_um, direction="y")
    info = {
        "source": source,
        "folder": saved.folder,
        "time_offset_s": offset,
        "ks4_sign_flipped": source == "ks4",
        "median_filter_bins": smooth_bins,
    }
    return motion, info


def correct_recording(recording: BaseRecording, motion: Motion) -> BaseRecording:
    """Interpolate *recording* by *motion*, keeping every channel in its order.

    The result is float32 (``interpolate_motion`` rejects integer input), and like any
    ``InterpolateMotionRecording`` it does not round-trip through JSON.

    Parameters
    ----------
    recording : BaseRecording
        Recording to correct.
    motion : Motion
        Estimate in SpikeInterface's sign, e.g. from :func:`load_motion`.

    Returns
    -------
    BaseRecording
        Lazy corrected recording.
    """
    _require_si()
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    corrected: BaseRecording = interpolate_motion(
        astype(recording, "float32"),
        motion,
        spatial_interpolation_method="kriging",
        border_mode="force_extrapolate",
    )
    return corrected
