# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import current_platform
from vllm.utils.import_utils import PlaceholderModule, _has_module, has_tilelang


def _raises_module_not_found():
    return pytest.raises(ModuleNotFoundError, match="No module named")


def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule("placeholder_1234")

    with _raises_module_not_found():
        int(placeholder)

    with _raises_module_not_found():
        placeholder()

    with _raises_module_not_found():
        _ = placeholder.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __name attribute
        _ = placeholder.name

    # OK to print the placeholder or use it in a f-string
    _ = repr(placeholder)
    _ = str(placeholder)

    # No error yet; only error when it is used downstream
    placeholder_attr = placeholder.placeholder_attr("attr")

    with _raises_module_not_found():
        int(placeholder_attr)

    with _raises_module_not_found():
        placeholder_attr()

    with _raises_module_not_found():
        _ = placeholder_attr.some_attr

    with _raises_module_not_found():
        # Test conflict with internal __module attribute
        _ = placeholder_attr.module


class TestHasModule:
    """Tests for _has_module with trial import verification."""

    def setup_method(self):
        # Clear the @cache between tests so each test gets a fresh call
        _has_module.cache_clear()

    def test_returns_true_for_importable_stdlib_module(self):
        assert _has_module("json") is True

    def test_returns_false_for_nonexistent_module(self):
        assert _has_module("nonexistent_module_xyz_12345") is False

    def test_returns_false_when_find_spec_succeeds_but_import_fails(self):
        """Simulate a native extension whose shared library is missing.

        ``find_spec`` finds the package on disk, but the actual import
        raises ``ImportError`` (e.g. missing ``libcudart.so``).
        """
        fake_spec = MagicMock()

        with (
            patch(
                "vllm.utils.import_utils.importlib.util.find_spec",
                return_value=fake_spec,
            ),
            patch(
                "vllm.utils.import_utils.importlib.import_module",
                side_effect=ImportError(
                    "libcudart.so.12: cannot open shared object file"
                ),
            ),
        ):
            assert _has_module("fake_native_ext") is False

    def test_returns_false_when_find_spec_raises(self):
        """``find_spec`` itself can raise for dotted names whose parent package
        fails to import. This should be treated as the module being unavailable.
        """
        with patch(
            "vllm.utils.import_utils.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'fake_parent'"),
        ):
            assert _has_module("fake_parent.child") is False

    def test_result_is_cached(self):
        """Verify the @cache decorator prevents repeated imports."""
        _has_module("json")  # prime the cache

        with patch("vllm.utils.import_utils.importlib.util.find_spec") as mock_spec:
            result = _has_module("json")  # should hit cache
            mock_spec.assert_not_called()
            assert result is True


class TestHasTilelang:
    def setup_method(self):
        has_tilelang.cache_clear()

    def test_availability_check_does_not_import_tilelang(self):
        with (
            patch(
                "vllm.utils.import_utils.importlib.util.find_spec",
                return_value=MagicMock(),
            ),
            patch("vllm.utils.import_utils.importlib.import_module") as mock_import,
            patch.object(current_platform, "is_rocm", return_value=False),
        ):
            assert has_tilelang() is True
            mock_import.assert_not_called()

    def test_returns_false_when_tilelang_is_not_installed(self):
        with patch(
            "vllm.utils.import_utils.importlib.util.find_spec", return_value=None
        ):
            assert has_tilelang() is False
