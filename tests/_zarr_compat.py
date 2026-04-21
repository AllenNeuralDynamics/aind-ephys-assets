"""Shared test helper: write v2-format zarrs portably across zarr 2.x / 3.x.

The package reads v2-format zarrs on both zarr major versions. Tests
must also *write* v2-format fixtures regardless of the installed zarr
version — which requires branching, because the ``zarr_format=2`` kwarg
is v3-only (v2 has no such kwarg and only writes v2 format anyway).
"""

from __future__ import annotations

from typing import Any

import zarr

_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def open_group_w_v2(path: str) -> Any:
    """Open a zarr group for writing in v2 format, portable across zarr versions."""
    if _ZARR_V3:
        return zarr.open_group(path, mode="w", zarr_format=2)
    return zarr.open_group(path, mode="w")


def create_array(grp: Any, name: str, *, shape: tuple[int, ...], dtype: str) -> Any:
    """Create an Array in *grp*, portable across zarr 2.x (``create_dataset``)
    and zarr 3.x (``create_array``).
    """
    if _ZARR_V3:
        return grp.create_array(name, shape=shape, dtype=dtype)
    return grp.create_dataset(name, shape=shape, dtype=dtype)
