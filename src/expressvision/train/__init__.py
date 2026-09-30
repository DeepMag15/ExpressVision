"""Detector training and size-stratified evaluation.

Kept out of the base install. Training needs torch, transformers and a GPU;
the collector, the CLI, the console and the tests all run without any of them,
because an edge node that only collects must not carry a training stack.

    uv pip install -e ".[train]"

The evaluation here is deliberately not mAP. mAP averages over object sizes and
so hides the only thing this project needs to know: whether the detector works
at the pixel size inherited CCTV actually delivers. See :mod:`.evaluate`.
"""

from __future__ import annotations

__all__ = ["data", "evaluate"]
