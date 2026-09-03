"""Import-string entry point for ``uvicorn --reload``.

Reload works by re-importing the app in a fresh process, so it needs a module
path rather than an object. The settings therefore travel through the
environment; ``exv serve --reload`` sets them. Nothing outside development
should use this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from .app import create_app


def _path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


app = create_app(
    db_path=_path("EXV_DB") or Path("out/expressvision.db"),
    media_root=_path("EXV_MEDIA_ROOT"),
    static_dir=_path("EXV_STATIC_DIR"),
    dev=os.environ.get("EXV_DEV") == "1",
)
