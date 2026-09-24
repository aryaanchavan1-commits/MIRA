"""Hardware detection and resource-aware profiling (spec §3-6, §38).

Detects CPU, RAM, GPU/VRAM, CUDA, disk without hard-coding any machine.
Everything degrades gracefully: a machine with no NVIDIA GPU gets
ULTRA_LOW/LOW/MEDIUM profiles and CPU-only inference; nothing crashes.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from core.types import parse_float, parse_int

logger = logging.getLogger("mira.hardware")


# ---------------------------------------------------------------------------
# Performance profiles (spec §39). Fractions apply to *usable* memory, where
# usable = total - safety headroom. Never 100% of anything.
# ---------------------------------------------------------------------------
PROFILE_FRACTIONS: Dict[str, Dict[str, float]] = {
    #                 share of usable RAM   share of usable VRAM
    "safe":     dict(ram=0.50, vram=0.60),
    "balanced": dict(ram=0.70, vram=0.80),
    "fast":     dict(ram=0.85, vram=0.95),
    "research": dict(ram=0.95, vram=1.00),  # warnings shown in UI
}


@dataclass
class HardwareProfile:
    os: str = "unknown"
    os_version: str = ""
    python_version: str = ""
    cpu_name: str = "unknown"
    cpu_arch: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    ram_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_name: str = "none"
    vram_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: str = ""
    driver_version: str = ""
    free_disk_gb: float = 0.0
    project_disk: str = ""
    errors: list = field(default_factory=list)

    # ---- dynamic budget calculation (spec §4) ----
    def usable_ram_gb(self, mode: str = "balanced") -> float:
        frac = PROFILE_FRACTIONS.get(mode, PROFILE_FRACTIONS["balanced"])["ram"]
        return round(max(0.5, self.ram_available_gb * frac), 2)

    def usable_vram_gb(self, mode: str = "balanced") -> float:
        if not self.cuda_available or self.vram_free_gb <= 0:
            return 0.0
        frac = PROFILE_FRACTIONS.get(mode, PROFILE_FRACTIONS["balanced"])["vram"]
        return round(max(0.0, self.vram_free_gb * frac), 2)

    def tier(self) -> str:
        """ULTRA_LOW / LOW / MEDIUM / HIGH / GPU_HIGH — from memory, not GPU name."""
        eff_vram = self.vram_gb if self.cuda_available else 0.0
        if self.ram_gb < 4:
            return "ULTRA_LOW"
        if self.ram_gb < 8 or (eff_vram == 0 and self.ram_gb < 12):
            return "LOW"
        if eff_vram >= 8 and self.ram_gb >= 16:
            return "GPU_HIGH"
        if eff_vram >= 4 or self.ram_gb >= 16:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        gpu = f"{self.gpu_name} ({self.vram_gb:.1f} GB, {self.vram_free_gb:.1f} free)" \
            if self.gpu_name != "none" else "none (CPU-only)"
        return (
            f"{self.os} {self.os_version} | Python {self.python_version} | "
            f"{self.cpu_name} ({self.physical_cores}C/{self.logical_cores}T) | "
            f"RAM {self.ram_gb:.1f} GB ({self.ram_available_gb:.1f} free) | GPU: {gpu} | "
            f"disk {self.free_disk_gb:.1f} GB free on {self.project_disk} | tier {self.tier()}"
        )


def _ram_windows() -> Dict[str, float]:
    """GlobalMemoryStatusEx — works without psutil, returns GB."""
    import ctypes
    import ctypes.wintypes as wt

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD),
            ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return {
            "total": stat.ullTotalPhys / 2**30,
            "available": stat.ullAvailPhys / 2**30,
        }
    return {}


def _ram_psutil() -> Dict[str, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total": vm.total / 2**30, "available": vm.available / 2**30}
    except Exception:
        return {}


def _ram_stdlib() -> Dict[str, float]:
    try:
        import sys
        if sys.platform == "linux":
            with open("/proc/meminfo", "r", encoding="ascii") as fh:
                info = {}
                for line in fh:
                    key, _, val = line.partition(":")
                    info[key.strip()] = float(val.strip().split()[0]) / 2**20  # KiB→GB
            return {
                "total": info.get("MemTotal", 0.0),
                "available": info.get("MemAvailable", info.get("MemFree", 0.0)),
            }
    except Exception:
        pass
    return {}


def detect_ram() -> Dict[str, float]:
    for fn in (_ram_windows, _ram_psutil, _ram_stdlib):
        try:
            info = fn()
            if info.get("total", 0) > 0:
                return info
        except Exception:
            continue
    return {}


def detect_gpu() -> Dict[str, Any]:
    """VRAM via NVML → nvidia-smi → torch. Returns {} when no NVIDIA GPU."""
    # 1) NVML (cleanest, gives free VRAM too)
    try:
        import pynvml  # noqa
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            drv = pynvml.nvmlSystemGetDriverVersion()
            drv = drv.decode() if isinstance(drv, bytes) else str(drv)
            return {
                "name": name,
                "total_gb": mem.total / 2**30,
                "free_gb": mem.free / 2**30,
                "driver": drv,
                "source": "nvml",
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # 2) nvidia-smi subprocess
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            if out and "," in out[0]:
                name, total, free, drv = [p.strip() for p in out[0].split(",", 3)]
                return {
                    "name": name,
                    "total_gb": float(total) / 1024.0,
                    "free_gb": float(free) / 1024.0,
                    "driver": drv,
                    "source": "nvidia-smi",
                }
        except Exception:
            pass

    # 3) torch (if installed for CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, _total_b = torch.cuda.mem_get_info(0)
            return {
                "name": props.name,
                "total_gb": props.total_memory / 2**30,
                "free_gb": free_b / 2**30,
                "driver": "",
                "source": "torch",
            }
    except Exception:
        pass
    return {}


def detect_cuda_version(gpu_info: Dict[str, Any]) -> str:
    if not gpu_info:
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if "CUDA Version" in line:
                return line.split("CUDA Version:")[1].split(")")[0].strip()
    except Exception:
        pass
    return ""


def detect_hardware(project_root: Optional[str] = None) -> HardwareProfile:
    project_root = project_root or os.getcwd()
    prof = HardwareProfile()

    prof.os = platform.system() or "unknown"
    prof.os_version = platform.release()
    prof.python_version = platform.python_version()
    prof.cpu_arch = platform.machine()

    # CPU name
    if prof.os == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                prof.cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            prof.cpu_name = platform.processor() or "unknown"
    else:
        try:
            with open("/proc/cpuinfo", "r", encoding="ascii") as fh:
                for line in fh:
                    if "model name" in line:
                        prof.cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            prof.cpu_name = platform.processor() or "unknown"

    prof.logical_cores = os.cpu_count() or 1
    try:
        prof.physical_cores = os.cpu_count() or 1  # cheap approximation
        if prof.os == "Windows":
            import ctypes
            num = ctypes.c_ulong()
            ctypes.windll.kernel32.GetLogicalProcessorInformationEx(1, None, ctypes.byref(num))
            # (physical-core refinement skipped; logical count is what tuning needs)
            prof.physical_cores = max(1, prof.logical_cores // 2) if prof.logical_cores > 2 else prof.logical_cores
        elif prof.os == "Linux":
            import re
            with open("/proc/cpuinfo", "r", encoding="ascii") as fh:
                ids = {m.group(1) for m in re.finditer(r"physical id\s*:\s*(\d+)", fh.read())}
            cores = set()
            # count unique (physical, core) pairs
            txt = open("/proc/cpuinfo", "r", encoding="ascii").read()
            pairs = set(re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", txt, re.S))
            prof.physical_cores = len(pairs) or prof.logical_cores
    except Exception:
        prof.physical_cores = prof.logical_cores

    ram = detect_ram()
    prof.ram_gb = round(ram.get("total", 0.0), 2)
    prof.ram_available_gb = round(ram.get("available", prof.ram_gb * 0.5), 2)

    gpu = detect_gpu()
    if gpu:
        prof.gpu_name = gpu["name"]
        prof.vram_gb = round(gpu["total_gb"], 2)
        prof.vram_free_gb = round(gpu["free_gb"], 2)
        prof.driver_version = gpu.get("driver", "")
        prof.cuda_available = True
        prof.cuda_version = detect_cuda_version(gpu)
    else:
        # torch may still be CPU-only but present — CUDA unusable either way
        prof.cuda_available = False

    try:
        disk = shutil.disk_usage(project_root)
        prof.free_disk_gb = round(disk.free / 2**30, 2)
        prof.project_disk = os.path.splitdrive(project_root)[0] or "/"
    except Exception as exc:
        prof.errors.append(f"disk: {exc}")

    logger.info(
        "hardware detected",
        extra={"tier": prof.tier(), "ram_gb": prof.ram_gb, "gpu": prof.gpu_name,
               "cuda": prof.cuda_available, "disk_gb": prof.free_disk_gb},
    )
    return prof


# ---------------------------------------------------------------------------
# Auto-configuration (spec §38): turn a HardwareProfile + user config into a
# concrete, safe runtime configuration. If a candidate doesn't fit, step down
# until one does. Never raises for insufficient hardware — degrades instead.
# ---------------------------------------------------------------------------
@dataclass
class RuntimeConfig:
    performance_mode: str = "balanced"
    llm_backend: str = "none"          # llama_cpp | none
    llm_model: str = ""                # gguf filename once known
    llm_params_b: float = 0.0
    llm_quant: str = ""
    n_gpu_layers: int = 0
    n_threads: int = 2
    n_ctx: int = 2048
    n_batch: int = 128
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    usable_ram_gb: float = 0.0
    usable_vram_gb: float = 0.0
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gguf_ram_need_gb(params_b: float, quant: str) -> float:
    """Rough GGUF file size + KV/scratch estimate, GB."""
    bits_per_weight = {
        "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q4_K_M": 4.8,
        "Q4_K_S": 4.6, "Q4_0": 4.5, "Q3_K_M": 3.9, "Q2_K": 3.4,
    }.get(quant.upper(), 5.0)
    file_gb = params_b * bits_per_weight / 8.0
    # KV cache + compute buffers scale with context; assume 4k ctx default tune
    overhead_gb = 0.4 + (params_b * 0.08)
    return file_gb + overhead_gb


def auto_configure(hw: HardwareProfile, cfg: Dict[str, Any]) -> RuntimeConfig:
    """Pick the largest acceptable model config that fits usable memory.

    Selection ladder (spec §5): try candidates in preference order; the first
    that fits *within the active performance profile's budget* wins. If none
    fit, fall back to the smallest candidate with heavy downgrades — never
    raise, never crash (spec §24/§38).
    """
    rc = RuntimeConfig()
    mode = str(cfg.get("performance_mode", "balanced")).lower()
    if mode not in PROFILE_FRACTIONS:
        mode = "balanced"
    rc.performance_mode = mode

    usable_ram = hw.usable_ram_gb(mode)
    usable_vram = hw.usable_vram_gb(mode)
    rc.usable_ram_gb = usable_ram
    rc.usable_vram_gb = usable_vram

    # threads: half the logical cores, capped (hyperthreads rarely help llama.cpp)
    rc.n_threads = max(2, min(8, (hw.logical_cores or 4) // 2))

    # ---- embedding model ----
    emb_pref = (cfg.get("models", {}).get("embeddings", {}) or {}).get("prefer",
                ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "paraphrase-MiniLM-L3-v2"])
    rc.embedding_model = emb_pref[0]
    # sentence-transformers on GPU only if it clearly fits
    rc.embedding_device = "cuda" if usable_vram >= 1.5 else "cpu"
    if rc.embedding_device == "cuda":
        usable_vram = max(0.0, usable_vram - 0.6)  # reserve for embeddings
        rc.notes.append("embeddings on GPU (-0.6 GB VRAM reserve)")

    # ---- LLM selection ----
    models_cfg = cfg.get("models", {}).get("llm", {}) or {}
    prefer_b = models_cfg.get("prefer_params_b", [0.5, 1.0, 1.5, 2.0, 3.0])
    prefer_q = models_cfg.get("prefer_quant", ["Q4_K_M", "Q5_K_M", "Q4_0", "Q8_0"])

    max_disk_cfg = cfg.get("resources", {}).get("max_model_disk_gb", "auto")
    # 10% of free disk (min 2.5 GB): a 3B Q4_K_M (~1.9 GB) is small next to any
    # real corpus; the old 5% cap rejected models that fit memory comfortably
    max_disk = parse_float(max_disk_cfg, 4.0) if str(max_disk_cfg).lower() != "auto" else max(2.5, hw.free_disk_gb * 0.10)

    budget_gb = usable_vram + usable_ram  # hybrid: weights can split GPU/CPU

    for params_b in prefer_b:
        for quant in prefer_q:
            need = _gguf_ram_need_gb(params_b, quant)
            file_need = params_b * ({"Q4_K_M": 4.8, "Q5_K_M": 5.7, "Q4_0": 4.5, "Q8_0": 8.5}
                                    .get(quant, 5.0)) / 8.0
            if need <= budget_gb and file_need <= max_disk:
                rc.llm_backend = "llama_cpp"
                rc.llm_params_b = params_b
                rc.llm_quant = quant
                # GPU offload: fit as many layers as VRAM allows (spec §24)
                if usable_vram >= need:
                    rc.n_gpu_layers = -1  # all layers
                elif usable_vram > file_need * 0.4:
                    # proportion of weights that fit on GPU; llama.cpp maps
                    # n_gpu_layers=-0.6 equivalent via negative values, but
                    # integer layer counts are safer: estimate 32 layers scale
                    rc.n_gpu_layers = max(0, min(28, int(usable_vram / max(file_need, 0.1) * 32)))
                else:
                    rc.n_gpu_layers = 0
                # context: shrink if memory is tight (spec §24: ctx ↓ when tight)
                ctx = 4096 if mode in ("balanced", "fast", "research") else 2048
                if usable_ram + usable_vram < need * 1.3:
                    ctx = 2048
                if usable_ram + usable_vram < need * 1.1:
                    ctx = 1024
                cfg_ctx = models_cfg.get("max_context_tokens", "auto")
                if str(cfg_ctx).lower() != "auto":
                    ctx = min(ctx, parse_int(cfg_ctx, ctx))
                rc.n_ctx = ctx
                rc.n_batch = 256 if ctx >= 4096 else 128
                return rc

    # Nothing fit the preference list → smallest candidate, max downgrades
    rc.llm_backend = "llama_cpp"
    rc.llm_params_b = min(prefer_b) if prefer_b else 0.5
    rc.llm_quant = "Q4_K_M"
    rc.n_gpu_layers = 0
    rc.n_ctx = 1024
    rc.n_batch = 64
    rc.notes.append(
        f"hardware below minimum for preferences; using {rc.llm_params_b}B Q4_K_M, CPU-only, 1k ctx"
    )
    return rc


def validate_runtime(rc: RuntimeConfig, hw: HardwareProfile) -> list:
    """Post-hoc validation with human-readable warnings (never raises)."""
    warnings = []
    if rc.llm_backend == "llama_cpp" and rc.llm_params_b == 0.0:
        warnings.append("no LLM selected — extraction/answer will use deterministic fallbacks")
    if hw.free_disk_gb < 2.0:
        warnings.append(f"low disk: {hw.free_disk_gb:.1f} GB free — model downloads may fail")
    if rc.usable_ram_gb < 2.0:
        warnings.append("usable RAM < 2 GB — expect heavy swapping; close other applications")
    return warnings
