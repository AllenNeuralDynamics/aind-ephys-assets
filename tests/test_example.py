"""Example test module."""

import aind_ephys_assets


def test_version():
    """Test that version is defined."""
    assert aind_ephys_assets.__version__ is not None
    assert isinstance(aind_ephys_assets.__version__, str)
