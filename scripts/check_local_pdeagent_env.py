"""Check whether the current Python environment matches the pdeagent conda env.

Usage:
    python scripts/check_local_pdeagent_env.py          # non-strict: always exit 0
    python scripts/check_local_pdeagent_env.py --strict # fail if not pdeagent or missing torch

Reads configs/local_pdeagent_env.yaml for expected environment settings.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "local_pdeagent_env.yaml"


def _try_conda_env() -> str | None:
    """Return conda env name from environment variables."""
    return os.environ.get("CONDA_DEFAULT_ENV")


def _try_conda_info() -> str | None:
    """Fallback: try `conda info --json` to detect active env."""
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            return info.get("active_prefix_name")
    except Exception:
        pass
    return None


def _import_check(pkg_name: str) -> tuple[bool, str]:
    try:
        __import__(pkg_name)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def run_checks(strict: bool = False) -> dict:
    """Run all environment checks and return a structured result dict.

    Returns:
        Dict with keys: python_exe, python_version, conda_env, conda_env_match,
        packages, cuda_available, cuda_device_count, gpu_name, all_pass,
        warnings, errors.
    """
    import yaml

    config = {}
    if DEFAULT_CONFIG.is_file():
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    checks_cfg = config.get("checks", {})
    expected_env = checks_cfg.get("expected_env_name", "pdeagent")
    required_pkgs = checks_cfg.get("required_packages", ["torch", "numpy", "h5py", "yaml"])
    require_cuda = checks_cfg.get("require_cuda_for_gpu", True)

    warnings: list[str] = []
    errors: list[str] = []

    python_exe = sys.executable
    python_version = sys.version.split()[0]

    conda_env = _try_conda_env()
    if conda_env is None:
        conda_env = _try_conda_info()
    env_match = conda_env == expected_env if conda_env else False

    if not env_match:
        msg = (f"Current conda env: '{conda_env or '<none>'}', "
               f"expected: '{expected_env}'")
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
        if conda_env is None:
            warnings.append("Hint: conda activate pdeagent")

    # Package checks
    packages: dict[str, dict] = {}
    for pkg in required_pkgs:
        ok, detail = _import_check(pkg)
        packages[pkg] = {"available": ok, "detail": detail}
        if not ok:
            err_msg = f"Package '{pkg}' not available: {detail}"
            if strict and pkg == "torch":
                errors.append(err_msg)
            else:
                warnings.append(err_msg)

    # CUDA check
    cuda_available = False
    cuda_device_count = 0
    gpu_name = None
    if packages.get("torch", {}).get("available"):
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            if cuda_available and cuda_device_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
            if require_cuda and not cuda_available:
                warnings.append("CUDA not available — GPU training will fall back to CPU")
        except Exception as exc:
            warnings.append(f"Could not check CUDA: {exc}")

    all_pass = len(errors) == 0

    return {
        "python_exe": python_exe,
        "python_version": python_version,
        "conda_env": conda_env,
        "expected_env": expected_env,
        "env_match": env_match,
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "gpu_name": gpu_name,
        "warnings": warnings,
        "errors": errors,
        "all_pass": all_pass,
        "strict": strict,
    }


def print_summary(result: dict) -> None:
    print("Local pdeagent environment check:")
    print(f"  Python executable : {result['python_exe']}")
    print(f"  Python version    : {result['python_version']}")
    print(f"  CONDA_DEFAULT_ENV : {result['conda_env'] or '<not in conda>'}")
    print(f"  Expected env      : {result['expected_env']}")
    print(f"  Env match         : {result['env_match']}")
    print()

    print("  Packages:")
    for pkg, info in result["packages"].items():
        status = "OK" if info["available"] else f"MISSING ({info['detail'][:60]})"
        print(f"    {pkg:12s}: {status}")

    print(f"\n  CUDA available    : {result['cuda_available']}")
    print(f"  CUDA device count : {result['cuda_device_count']}")
    if result["gpu_name"]:
        print(f"  GPU name          : {result['gpu_name']}")
    print()

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  [WARN] {w}")
        if not result["env_match"]:
            print("\n  Hint: Run `conda activate pdeagent` before local GPU tasks.")

    if result["errors"]:
        for e in result["errors"]:
            print(f"  [ERROR] {e}")

    status = "PASS" if result["all_pass"] else "FAIL"
    print(f"\nEnvironment check: {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if env does not match pdeagent")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to env config YAML")
    args = parser.parse_args(argv)

    result = run_checks(strict=args.strict)
    print_summary(result)

    if args.strict and not result["all_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
