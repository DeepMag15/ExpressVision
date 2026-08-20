"""GPU environment probing and telemetry.

Two jobs:

* Report what the machine actually has, so a CUDA setup either works or fails
  loudly rather than falling back to CPU and quietly costing a day.
* Sample utilisation and memory during a benchmark, so throughput numbers come
  with the context needed to interpret them — a run at 30% GPU utilisation is
  bottlenecked somewhere else, and knowing that changes what to fix next.

Everything degrades gracefully: no torch, no CUDA, no NVML all produce a report
rather than an exception.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GpuSnapshot:
    """One instantaneous reading."""

    utilisation_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None


@dataclass
class EnvReport:
    python: str = ""
    platform: str = ""
    torch_version: str | None = None
    torch_cuda_build: str | None = None      # CUDA version torch was built with
    cuda_available: bool = False
    device_count: int = 0
    device_name: str | None = None
    device_capability: str | None = None
    total_vram_mb: float | None = None
    driver_version: str | None = None
    nvml_available: bool = False
    cudnn_version: str | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_ready(self) -> bool:
        return self.cuda_available and self.device_count > 0


def probe_environment() -> EnvReport:
    """Everything worth knowing before trusting a benchmark number."""
    import platform
    import sys

    report = EnvReport(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
    )

    try:
        import torch
    except ImportError:
        report.problems.append(
            "torch is not installed — install the ml extra: uv sync --extra ml"
        )
        _add_driver_info(report)
        return report

    report.torch_version = torch.__version__
    report.torch_cuda_build = getattr(torch.version, "cuda", None)
    report.cuda_available = bool(torch.cuda.is_available())

    if report.torch_cuda_build is None:
        report.problems.append(
            "This is a CPU-only build of torch. Reinstall from the CUDA index:\n"
            "    uv pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu124"
        )

    if report.cuda_available:
        report.device_count = torch.cuda.device_count()
        try:
            props = torch.cuda.get_device_properties(0)
            report.device_name = props.name
            report.device_capability = f"{props.major}.{props.minor}"
            report.total_vram_mb = props.total_memory / 1e6
        except Exception as exc:                        # pragma: no cover
            report.problems.append(f"Could not read device properties: {exc}")
        try:
            report.cudnn_version = str(torch.backends.cudnn.version())
        except Exception:                               # pragma: no cover
            pass
    elif report.torch_cuda_build is not None:
        report.problems.append(
            "torch has CUDA support compiled in but reports no usable device. "
            "Check the NVIDIA driver, and that the GPU is visible "
            "(CUDA_VISIBLE_DEVICES)."
        )

    report.nvml_available = _nvml_handle() is not None
    if not report.nvml_available and report.gpu_ready:
        report.notes.append(
            "pynvml not installed — utilisation and temperature will fall back to "
            "nvidia-smi, or be omitted. Install with: uv pip install nvidia-ml-py"
        )

    _add_driver_info(report)
    return report


def _add_driver_info(report: EnvReport) -> None:
    handle = _nvml_handle()
    if handle is not None:
        try:
            import pynvml

            report.driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(report.driver_version, bytes):
                report.driver_version = report.driver_version.decode()
            return
        except Exception:                               # pragma: no cover
            pass

    smi = _nvidia_smi(["--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        report.driver_version = smi[0]


_NVML_STATE: dict[str, object] = {}


def _nvml_handle():
    """Initialise NVML once; return a device handle or None."""
    if "handle" in _NVML_STATE:
        return _NVML_STATE["handle"]

    handle = None
    try:
        import pynvml

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        handle = None

    _NVML_STATE["handle"] = handle
    return handle


def _nvidia_smi(args: list[str]) -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=10, check=False
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:                                   # pragma: no cover
        return []


def sample_gpu() -> GpuSnapshot:
    """One reading of utilisation, memory and temperature.

    NVML when available (fast enough to poll during a run), nvidia-smi otherwise
    (a subprocess per call, so only for occasional sampling).
    """
    handle = _nvml_handle()
    if handle is not None:
        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                temp = None
            return GpuSnapshot(
                utilisation_pct=float(util.gpu),
                memory_used_mb=mem.used / 1e6,
                memory_total_mb=mem.total / 1e6,
                temperature_c=temp,
            )
        except Exception:                               # pragma: no cover
            pass

    rows = _nvidia_smi(
        [
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if rows:
        try:
            util, used, total, temp = (p.strip() for p in rows[0].split(","))
            return GpuSnapshot(
                utilisation_pct=float(util),
                memory_used_mb=float(used),
                memory_total_mb=float(total),
                temperature_c=float(temp),
            )
        except (ValueError, IndexError):                # pragma: no cover
            pass

    return GpuSnapshot()


def torch_vram_mb() -> tuple[float | None, float | None]:
    """Torch's own allocated / peak-reserved VRAM, in MB.

    Distinct from the NVML reading: this is what *our process* holds, which is
    what determines how many model instances fit on one card.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.max_memory_reserved() / 1e6,
        )
    except Exception:                                   # pragma: no cover
        return None, None


def reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                   # pragma: no cover
        pass


def synchronize() -> None:
    """Block until queued CUDA work finishes.

    Mandatory around timed sections: CUDA calls are asynchronous, so without it
    a benchmark measures how fast Python can enqueue work, not how fast the GPU
    completes it — which produces impossibly good numbers.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                                   # pragma: no cover
        pass
