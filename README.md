# AIND ephys assets

![CI](https://github.com/AllenNeuralDynamics/aind-ephys-assets/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-ephys-assets)](https://pypi.org/project/aind-ephys-assets/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

AIND ephys asset layout knowledge and helpers — stream discovery, path
resolution, and name parsing for capsules that consume AIND raw +
sorted electrophysiology assets. Pure filesystem / S3 structure
reasoning; does not wrap SpikeInterface, does not build mipmaps.

## Installation

```bash
pip install aind-ephys-assets          # local paths only
pip install 'aind-ephys-assets[s3]'    # adds s3fs for s3:// paths
pip install 'aind-ephys-assets[si]'    # adds SpikeInterface for load_motion / correct_recording
```

Or from a clone:

```bash
pip install .            # or `pip install '.[s3]'`
uv sync                  # dev
```

## Usage

```python
from aind_ephys_assets import (
    discover_raw_streams,
    discover_analyzers,
    resolve_raw_root,
    resolve_sort_root,
    find_raw_zarr,
)

# Enumerate raw AP/LFP streams (local or s3://).  Auto-descends into
# ecephys/ecephys_compressed/ when the caller passes the asset root.
streams = discover_raw_streams("/data/raw_asset")
for s in streams:
    print(s.name, s.sampling_frequency, s.n_channels)

# Enumerate sorted analyzers and map each to its raw counterpart.
analyzers = discover_analyzers(
    "/data/sorted_asset",
    raw_root="s3://my-bucket/raw_asset/ecephys/ecephys_compressed",
)
for a in analyzers:
    print(a.name, "→ raw:", a.raw_path)

# S3 with anonymous auth (e.g. AIND Open Data):
streams = discover_raw_streams(
    "s3://aind-open-data/ecephys_...",
    storage_options={"anon": True},
)

# Stand-alone helpers
from aind_ephys_assets import (
    extract_device_name,
    match_analyzer_to_raw,
    sanitize_stream_name,
    read_si_zarr_header,
    open_raw_traces,
)

assert match_analyzer_to_raw("foo_recording1.zarr") == "foo.zarr"
assert extract_device_name("..._Neuropix-PXI-100.50213-1_recording1") == "50213-1"

header = read_si_zarr_header("/path/to/stream.zarr")
traces, sf = open_raw_traces("/path/to/stream.zarr")
```

### Drift estimates

A sorted asset saves Kilosort 4's applied estimate (`spikesorted/motion/<recording>/`, opposite
sign to SpikeInterface) and preprocessing's unapplied `dredge_fast`
(`preprocessed/motion/<recording>/motion/`). Both come back in SpikeInterface's sign; there is no
fallback from one to the other.

```python
from aind_ephys_assets import read_motion, load_motion, correct_recording

m = read_motion(sort_root, analyzer_name)              # numpy only
motion, info = load_motion(sort_root, analyzer_name, recording)   # [si]: Motion on the recording's clock
corrected = correct_recording(recording, motion)        # [si]: float32, kriging, all channels kept
```

SpikeInterface pins `zarr<3` below Python 3.14, so its tests run outside the default zarr 3
environment: `uv run --extra si --no-group zarr3 pytest`.

## Development

Please test your changes using the full linting and testing suite:

```bash
./scripts/run_linters_and_checks.sh -c
```

Or run individual commands:
```bash
uv run --frozen ruff format          # Code formatting
uv run --frozen ruff check           # Linting
uv run --frozen mypy                 # Type checking
uv run --frozen interrogate -v src   # Documentation coverage
uv run --frozen codespell --check-filenames  # Spell checking
uv run --frozen pytest --cov aind_ephys_assets # Tests with coverage
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
