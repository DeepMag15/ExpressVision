"""HTTP API for the operator console.

Imported lazily by the CLI so that `exv run`, `exv review` and the tests keep
working on a machine with no FastAPI installed — the collector is the product,
the console is an optional surface on top of it.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
