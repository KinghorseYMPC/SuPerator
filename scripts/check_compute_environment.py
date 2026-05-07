"""Print local or remote compute environment diagnostics.

This script is diagnostic only. It does not require torch or a GPU, does not
execute remote commands, and does not write artifacts.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


def _check_import(module_name: str, display_name: str | None = None) -> None:
    label = display_name or module_name
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"{label}: unavailable ({exc.__class__.__name__}: {exc})")
        return
    version = getattr(module, "__version__", "<unknown>")
    print(f"{label}: available (version={version})")


def _check_torch() -> None:
    try:
        import torch
    except ImportError as exc:
        print(f"torch: unavailable ({exc.__class__.__name__}: {exc})")
        return

    print(f"torch: available (version={torch.__version__})")
    cuda_available = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available: {cuda_available}")
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    print(f"torch.cuda.device_count: {device_count}")
    if cuda_available:
        for index in range(device_count):
            try:
                device_name = torch.cuda.get_device_name(index)
            except Exception as exc:  # pragma: no cover - hardware dependent
                device_name = f"<unavailable: {exc}>"
            print(f"torch.cuda.device[{index}].name: {device_name}")


def _print_environment_markers() -> None:
    print("SLURM markers:")
    for name in ["SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES"]:
        print(f"- {name}: {os.environ.get(name, '<unset>')}")

    kaggle_input = Path("/kaggle/input")
    print("Kaggle markers:")
    print(f"- KAGGLE_KERNEL_RUN_TYPE: {os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '<unset>')}")
    print(f"- /kaggle/input exists: {kaggle_input.exists()}")


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    for module_name, display_name in [
        ("numpy", "numpy"),
        ("h5py", "h5py"),
        ("pandas", "pandas"),
        ("yaml", "yaml"),
        ("pytest", "pytest"),
    ]:
        _check_import(module_name, display_name)

    _check_torch()
    _print_environment_markers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
