# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.5.0 (2026-10-24)

### Feat

- read and apply the drift estimates saved in sorted assets

### Fix

- type-check the zarr 2 FSStore branch against zarr 3

## v0.4.0 (2026-04-23)

### Fix

- `find_nwb` now matches Zarr NWB directories (e.g.
  `session.nwb/` with zarr chunks, read via
  `hdmf-zarr.NWBZarrIO`), not just HDF5 `*.nwb` files. Previously
  directory-format NWB assets mounted under `/data` were silently
  skipped and callers got `nwb_path=None`. Files are still preferred
  over directories when both exist in the same parent, so we never
  descend into a `.nwb` directory looking for a nested file.

## v0.3.0 (2026-04-21)

### Feat

- add `find_nwb(data_dir)` for locating AIND NWB files under
  pipeline-staged data directories (symlink-safe recursion).

### Refactor

- broaden zarr support to `zarr>=2.16` (previously `zarr>=3`). The
  package still reads only SI-format v2 zarrs; internally, a small
  shim in `zarr_io._open_group_r` branches on `zarr.__version__` to
  handle the different `open_group` signatures (v3 accepts
  `zarr_format=` and `storage_options=` kwargs; v2 requires an
  explicit `FSStore` wrap for URIs / storage options).

## v0.2.0 (2026-04-21)

### Feat

- implement v0.1 asset layout helpers
