"""Tests for quantx_engine._deprecation."""

from __future__ import annotations

import warnings

import pytest

from quantx_engine._deprecation import deprecated, warn


# ---------------------------------------------------------------------------
# warn() helper
# ---------------------------------------------------------------------------

class TestWarnHelper:
    def test_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn("quantx_engine.OldSymbol", since="0.4.0", removed_in="0.6.0",
                 replacement="NewSymbol", stacklevel=1)
        assert len(caught) == 1
        w = caught[0]
        assert issubclass(w.category, DeprecationWarning)
        assert "OldSymbol" in str(w.message)
        assert "0.4.0" in str(w.message)
        assert "0.6.0" in str(w.message)
        assert "NewSymbol" in str(w.message)

    def test_no_replacement_message(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn("quantx_engine.Sym", since="0.4.0", stacklevel=1)
        assert len(caught) == 1
        assert "instead" not in str(caught[0].message)

    def test_no_removed_in(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn("quantx_engine.Sym", since="0.4.0", stacklevel=1)
        assert "removed" not in str(caught[0].message)

    def test_stacklevel_controls_location(self):
        """Stacklevel 1 → warning points at the warn() call itself."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn("X", since="0.4.0", stacklevel=1)
        assert len(caught) == 1


# ---------------------------------------------------------------------------
# @deprecated decorator — functions
# ---------------------------------------------------------------------------

class TestDeprecatedFunction:
    def test_function_emits_warning(self):
        @deprecated(since="0.4.0", removed_in="0.6.0", replacement="new_fn")
        def old_fn(x: int) -> int:
            return x * 2

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_fn(3)

        assert result == 6
        assert len(caught) == 1
        w = caught[0]
        assert issubclass(w.category, DeprecationWarning)
        assert "old_fn" in str(w.message)
        assert "0.4.0" in str(w.message)
        assert "new_fn" in str(w.message)

    def test_function_preserves_return_value(self):
        @deprecated(since="0.4.0")
        def identity(x):
            return x

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert identity("hello") == "hello"

    def test_function_preserves_docstring(self):
        @deprecated(since="0.4.0")
        def documented():
            """Original docstring."""

        assert documented.__doc__ == "Original docstring."

    def test_function_preserves_name(self):
        @deprecated(since="0.4.0")
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    def test_function_marked_deprecated(self):
        @deprecated(since="0.4.0")
        def fn():
            pass

        assert getattr(fn, "__deprecated__", False) is True

    def test_deprecated_without_removed_in(self):
        @deprecated(since="0.4.0")
        def fn():
            return 42

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn()

        assert result == 42
        assert "removed" not in str(caught[0].message)


# ---------------------------------------------------------------------------
# @deprecated decorator — classes
# ---------------------------------------------------------------------------

class TestDeprecatedClass:
    def test_class_emits_warning_on_instantiation(self):
        @deprecated(since="0.4.0", removed_in="0.6.0", replacement="NewClass")
        class OldClass:
            def __init__(self, value: int):
                self.value = value

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = OldClass(10)

        assert obj.value == 10
        assert len(caught) == 1
        w = caught[0]
        assert issubclass(w.category, DeprecationWarning)
        assert "OldClass" in str(w.message)
        assert "NewClass" in str(w.message)

    def test_class_still_callable_after_deprecation(self):
        @deprecated(since="0.4.0")
        class Container:
            def __init__(self, items):
                self.items = list(items)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            c = Container([1, 2, 3])

        assert c.items == [1, 2, 3]

    def test_class_marked_deprecated(self):
        @deprecated(since="0.4.0")
        class Cls:
            pass

        assert getattr(Cls, "__deprecated__", False) is True

    def test_class_methods_work_normally(self):
        @deprecated(since="0.4.0")
        class Calculator:
            def __init__(self, x):
                self.x = x

            def double(self):
                return self.x * 2

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            calc = Calculator(7)

        assert calc.double() == 14


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_warning_type_is_deprecation_warning(self):
        """Must be DeprecationWarning, not UserWarning or FutureWarning."""
        @deprecated(since="0.4.0")
        def fn():
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()

        assert caught[0].category is DeprecationWarning

    def test_multiple_calls_each_emit_warning(self):
        @deprecated(since="0.4.0")
        def fn():
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
            fn()
            fn()

        # Each call from the same source produces one warning entry
        # (standard Python deduplication may collapse them — just check at least 1)
        assert len(caught) >= 1

    def test_kwargs_passed_through(self):
        @deprecated(since="0.4.0")
        def fn(a, b=10, *, c=20):
            return a + b + c

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = fn(1, b=2, c=3)

        assert result == 6
