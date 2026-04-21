"""AIND ephys asset layout knowledge and helpers"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-ephys-assets")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
