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
class DeviceInfo:
    """One GPU, including what is already running on it.

    Free memory matters as much as total on a shared machine. A DGX Station has
    four GPUs and no scheduler by default, so another user's job routinely
    occupies one — and torch defaults to device 0 regardless. Without this, a
    training run either fights for memory or OOMs on a box with three idle GPUs.
    """

    index: int
    name: str = ""
    capability: str = ""
    total_mb: float | None = None
    free_mb: float | None = None
    used_mb: float | None = None
    utilisation_pct: float | None = None

    @property
    def is_busy(self) -> bool:
        """Occupied by someone else's work.

        Judged on memory rather than utilisation: a job between batches shows
        0% for an instant but still holds its allocation, and picking that GPU
        would collide the moment it resumes.
        """
        if self.used_mb is None:
            return False
        return self.used_mb > 1024

    @property
    def supports_bf16(self) -> bool:
        """bfloat16 needs Ampere (SM 8.0) or newer.

        Volta and Turing do fp16 only. This is worth surfacing because a
        training config copied from an A100 recipe will specify bf16 and fail,
        or silently fall back to fp32 and run several times slower.
        """
        try:
            major = int(self.capability.split(".")[0])
        except (ValueError, IndexError):
            return False
        return major >= 8


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
    devices: list[DeviceInfo] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_ready(self) -> bool:
        return self.cuda_available and self.device_count > 0

    def free_devices(self) -> list[DeviceInfo]:
        return [d for d in self.devices if not d.is_busy]

    def recommended_devices(self) -> str | None:
        """A CUDA_VISIBLE_DEVICES value that avoids other people's jobs."""
        free = self.free_devices()
        return ",".join(str(d.index) for d in free) if free else None


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
        report.devices = _probe_devices(torch, report.device_count)

        busy = [d for d in report.devices if d.is_busy]
        if busy and len(busy) < len(report.devices):
            names = ", ".join(str(d.index) for d in busy)
            free = report.recommended_devices()
            report.notes.append(
                f"GPU {names} already in use by another process. torch defaults to "
                f"device 0 regardless — pin the free ones:\n"
                f"    export CUDA_VISIBLE_DEVICES={free}"
            )
        elif busy:
            report.problems.append(
                "Every GPU is already occupied. Training now will contend for "
                "memory or fail to allocate."
            )

        if report.devices and not any(d.supports_bf16 for d in report.devices):
            report.notes.append(
                "These GPUs predate Ampere, so bfloat16 is unavailable — use fp16 "
                "mixed precision. Flash-Attention 2 is unsupported here too; the "
                "default attention path is correct."
            )
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


def _probe_devices(torch, count: int) -> list[DeviceInfo]:
    """Per-GPU properties and current occupancy.

    torch reports capability and total memory; free memory comes from
    ``mem_get_info``, which reflects the whole device rather than this process,
    so another user's allocation shows up. nvidia-smi fills in utilisation and
    covers the case where mem_get_info is unavailable.
    """
    devices: list[DeviceInfo] = []
    for index in range(count):
        info = DeviceInfo(index=index)
        try:
            props = torch.cuda.get_device_properties(index)
            info.name = props.name
            info.capability = f"{props.major}.{props.minor}"
            info.total_mb = props.total_memory / 1e6
        except Exception:                               # pragma: no cover
            pass
        try:
            free, total = torch.cuda.mem_get_info(index)
            info.free_mb = free / 1e6
            info.total_mb = total / 1e6
            info.used_mb = (total - free) / 1e6
        except Exception:                               # pragma: no cover
            pass
        devices.append(info)

    rows = _nvidia_smi(
        ["--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"]
    )
    by_index = {d.index: d for d in devices}
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        device = by_index.get(index)
        if device is None:
            continue
        try:
            device.used_mb = float(parts[1])
            device.total_mb = float(parts[2])
            device.free_mb = device.total_mb - device.used_mb
            device.utilisation_pct = float(parts[3])
        except ValueError:                              # pragma: no cover
            pass

    return devices


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
