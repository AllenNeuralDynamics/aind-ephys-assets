"""Pure regex parsing of AIND ephys stream / recording names.

No I/O — each function takes a string and returns a string (or raises
on non-matching input). See :mod:`aind_ephys_assets.paths` for the
companion S3-URL parser.
"""

from __future__ import annotations

import re

__all__ = [
    "extract_device_name",
    "match_analyzer_to_raw",
    "sanitize_stream_name",
]

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Strip a trailing ``_recording<N>`` while preserving an optional ``.zarr``
# suffix.  The lookahead anchors the match either at end-of-string or
# immediately before ``.zarr$``.
_RECORDING_SUFFIX_RE = re.compile(r"_recording\d+(?=(?:\.zarr)?$)")

_DEVICE_NAME_RE = re.compile(r"Neuropix-PXI-\d+\.([^\s_]+)_recording")


def sanitize_stream_name(name: str) -> str:
    """Return a filesystem-safe basename for *name*.

    AIND recording names contain spaces, ``#``, and ``:`` which break some
    filesystems; this replaces any non-``[A-Za-z0-9._-]`` run with a single
    ``_`` and trims leading/trailing underscores.

    Parameters
    ----------
    name : str
        Raw stream or recording name, e.g.
        ``"experiment1_Record Node 111#Neuropix-PXI-100.50213-1"``.

    Returns
    -------
    str
        Sanitized basename suitable for use as a filesystem path
        component, e.g. ``"experiment1_Record_Node_111_Neuropix-PXI-100.50213-1"``.
    """
    return _SANITIZE_RE.sub("_", name).strip("_")


def match_analyzer_to_raw(analyzer_name: str) -> str:
    """Strip a trailing ``_recording<N>`` from an analyzer name.

    AIND's sort asset stores analyzer zarrs as
    ``postprocessed/<name>_recording<N>.zarr`` but the matching raw zarr
    is ``ecephys_compressed/<name>.zarr`` (no recording suffix). Accepts
    both the with- and without-``.zarr`` forms and preserves whichever
    was passed in.

    Parameters
    ----------
    analyzer_name : str
        Analyzer zarr dir name, either bare (``"<name>_recording1"``) or
        with suffix (``"<name>_recording1.zarr"``).

    Returns
    -------
    str
        ``analyzer_name`` with the ``_recording<N>`` segment removed.
        Returns the input unchanged if no recording suffix is present
        (idempotent).
    """
    return _RECORDING_SUFFIX_RE.sub("", analyzer_name)


def extract_device_name(recording_name: str) -> str:
    """Extract the probe serial from an AIND recording name.

    Parses the ``Neuropix-PXI-<N>.<serial>_recording<M>`` segment of a
    recording directory name.

    Parameters
    ----------
    recording_name : str
        Recording name, e.g.
        ``"experiment1_Record Node 111#Neuropix-PXI-100.50213-3_recording1"``.

    Returns
    -------
    str
        The probe device serial, e.g. ``"50213-3"``.

    Raises
    ------
    ValueError
        If *recording_name* does not contain a
        ``Neuropix-PXI-<N>.<serial>_recording`` segment — this is the
        expected behavior for non-Neuropixels streams (e.g. NI-DAQmx),
        which callers should filter out.
    """
    m = _DEVICE_NAME_RE.search(recording_name)
    if m is None:
        msg = f"cannot extract device name from: {recording_name!r}"
        raise ValueError(msg)
    return m.group(1)
