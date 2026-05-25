"""QuantX Engine — deprecation utilities.

Usage
-----
Emit a deprecation warning from library code::

    from quantx_engine._deprecation import warn, deprecated

    # Inline warning
    warn("OldClass", replacement="NewClass", since="0.4.0", removed_in="0.6.0")

    # Decorator for a deprecated function or class
    @deprecated(since="0.4.0", removed_in="0.6.0", replacement="new_function")
    def old_function(*args, **kwargs):
        ...

Policy
------
- A symbol is marked **deprecated** with a ``DeprecationWarning`` (visible with
  ``python -W default``) in the minor version it is superseded.
- It is **removed** no earlier than two minor versions later (e.g. deprecated in
  0.4.x → removed in 0.6.0 at the earliest).
- The deprecation warning message always includes:
    1. The deprecated symbol name.
    2. The version it was deprecated in (``since``).
    3. The planned removal version (``removed_in``).
    4. The recommended replacement (``replacement``), when one exists.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar

__all__ = ["warn", "deprecated"]

F = TypeVar("F", bound=Callable[..., Any])

# Module-level registry of already-warned symbols to avoid duplicate spam
# within the same process (one warning per call site is enough).
_WARNED: set[str] = set()


def warn(
    symbol: str,
    *,
    replacement: Optional[str] = None,
    since: str = "unknown",
    removed_in: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Emit a ``DeprecationWarning`` for *symbol*.

    Parameters
    ----------
    symbol:
        Fully-qualified name of the deprecated symbol
        (e.g. ``"quantx_engine.Engine.run_old"``).
    replacement:
        Recommended alternative, e.g. ``"Engine.run_new"``.
    since:
        Version string in which the symbol was deprecated, e.g. ``"0.4.0"``.
    removed_in:
        Version in which the symbol *will* be removed, e.g. ``"0.6.0"``.
    stacklevel:
        Passed to :func:`warnings.warn`; ``2`` points at the *caller's* site.
    """
    msg_parts = [
        f"``{symbol}`` is deprecated since v{since}",
    ]
    if removed_in:
        msg_parts.append(f"and will be removed in v{removed_in}")
    msg_parts[-1] += "."
    if replacement:
        msg_parts.append(f"Use ``{replacement}`` instead.")

    message = " ".join(msg_parts)
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def deprecated(
    *,
    since: str,
    removed_in: Optional[str] = None,
    replacement: Optional[str] = None,
) -> Callable[[F], F]:
    """Class/function decorator that emits a deprecation warning on each call.

    Example
    -------
    >>> from quantx_engine._deprecation import deprecated
    >>> @deprecated(since="0.4.0", removed_in="0.6.0", replacement="new_fn")
    ... def old_fn(x):
    ...     return x
    >>> old_fn(1)  # emits DeprecationWarning
    1
    """

    def decorator(obj: F) -> F:
        qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", str(obj))
        module = getattr(obj, "__module__", "")
        full_name = f"{module}.{qualname}" if module else qualname

        if isinstance(obj, type):
            # Class decorator — wrap __init__
            orig_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(orig_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn(
                    full_name,
                    since=since,
                    removed_in=removed_in,
                    replacement=replacement,
                    stacklevel=2,
                )
                orig_init(self, *args, **kwargs)

            obj.__init__ = new_init  # type: ignore[method-assign]
            obj.__deprecated__ = True  # type: ignore[attr-defined]
            return obj  # type: ignore[return-value]

        # Function / method decorator
        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn(
                full_name,
                since=since,
                removed_in=removed_in,
                replacement=replacement,
                stacklevel=2,
            )
            return obj(*args, **kwargs)

        wrapper.__deprecated__ = True  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
