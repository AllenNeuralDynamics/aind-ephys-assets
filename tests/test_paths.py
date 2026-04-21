"""Tests for aind_ephys_assets.paths."""

from __future__ import annotations

import pytest

from aind_ephys_assets.paths import split_s3_url


class TestSplitS3Url:
    def test_bucket_and_key(self):
        assert split_s3_url("s3://bucket/path/to/thing") == ("bucket", "path/to/thing")

    def test_bucket_only(self):
        assert split_s3_url("s3://bucket") == ("bucket", "")

    def test_bucket_with_trailing_slash(self):
        assert split_s3_url("s3://bucket/") == ("bucket", "")

    def test_strips_trailing_slash_from_key(self):
        assert split_s3_url("s3://bucket/key/") == ("bucket", "key")

    def test_key_with_multiple_slashes(self):
        assert split_s3_url("s3://bucket/a/b/c") == ("bucket", "a/b/c")

    def test_raises_on_non_s3_url(self):
        with pytest.raises(ValueError, match="expected s3:// URL"):
            split_s3_url("/local/path")

    def test_raises_on_https_url(self):
        with pytest.raises(ValueError, match="expected s3:// URL"):
            split_s3_url("https://example.com/bucket/key")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="expected s3:// URL"):
            split_s3_url("")
