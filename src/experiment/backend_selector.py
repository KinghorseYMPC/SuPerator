"""Local-only backend detection and selection for Task 1 experiment suites."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def detect_local_backend() -> dict[str, Any]:
    python_available = bool(sys.executable)
    torch_available = False
    cuda_available = False
    device_count = 0
    try:
        import torch  # type: ignore

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except ImportError:
        torch_available = False
    selected_backend = "local_gpu" if cuda_available and device_count > 0 else "local_cpu"
    return {
        "backend": "local",
        "python_available": python_available,
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "selected_backend": selected_backend,
        "status": "available" if python_available else "unavailable",
        "reasons": [],
    }


def detect_kaggle_backend(quick_check: bool = False) -> dict[str, Any]:
    executable = shutil.which("kaggle")
    result: dict[str, Any] = {
        "backend": "kaggle",
        "executable": executable,
        "cli_available": executable is not None,
        "auth_checked": False,
        "auth_available": None,
        "status": "available_or_recoverable" if executable else "unavailable",
        "reasons": [],
    }
    if executable is None:
        result["reasons"].append("kaggle CLI executable was not found")
        return result
    if not quick_check:
        result["reasons"].append("quick_check disabled; Kaggle auth was not queried")
        return result

    completed = subprocess.run(
        ["kaggle", "datasets", "list", "-s", "test"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    result["auth_checked"] = True
    result["auth_available"] = completed.returncode == 0
    if completed.returncode != 0:
        result["status"] = "recoverable_auth_required"
        result["reasons"].append("kaggle CLI auth check failed")
    return result


def _load_yaml_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Backend config must be a mapping: {path}")
    return payload


def detect_slurm_backend(
    config_path: str | Path = "configs/compute_backend.local.yaml",
) -> dict[str, Any]:
    resolved = resolve_path(config_path)
    result: dict[str, Any] = {
        "backend": "slurm",
        "config_path": str(resolved),
        "config_exists": resolved.is_file(),
        "enabled": False,
        "required_fields_present": False,
        "missing_fields": [],
        "status": "unavailable",
        "reasons": [],
    }
    payload = _load_yaml_if_present(resolved)
    if payload is None:
        result["reasons"].append("private SLURM backend config is not present")
        return result

    slurm = payload.get("slurm")
    if not isinstance(slurm, dict):
        result["reasons"].append("backend config does not contain a slurm mapping")
        return result

    result["enabled"] = bool(slurm.get("enabled"))
    required = ["host", "remote_project_dir", "env_type"]
    missing = [field for field in required if not slurm.get(field)]
    result["missing_fields"] = missing
    result["required_fields_present"] = not missing
    result["env_type"] = slurm.get("env_type") if slurm.get("env_type") else None
    if not result["enabled"]:
        result["reasons"].append("slurm.enabled is not true")
        return result
    if missing:
        result["reasons"].append(f"SLURM config is missing required fields: {', '.join(missing)}")
        return result

    result["status"] = "requires_manual_submit"
    result["reasons"].append("SLURM candidate is configured; remote submit remains manual")
    return result


def _policy_allows(policy: dict[str, Any], backend: str) -> bool:
    return bool(policy.get(f"allow_{backend}", True))


def _candidate_for_backend(backend: str) -> dict[str, Any]:
    if backend == "slurm":
        return detect_slurm_backend()
    if backend == "kaggle":
        return detect_kaggle_backend(quick_check=False)
    if backend == "local":
        return detect_local_backend()
    raise ValueError(f"Unsupported backend: {backend}")


def _candidate_available(candidate: dict[str, Any]) -> bool:
    status = candidate.get("status")
    if candidate.get("backend") == "local":
        return bool(candidate.get("python_available"))
    return status in {"requires_manual_submit", "available_or_recoverable", "available"}


def select_backend(policy: dict[str, Any], requested_backend: str = "auto") -> dict[str, Any]:
    if requested_backend != "auto":
        candidate = _candidate_for_backend(requested_backend)
        reasons = [f"requested backend: {requested_backend}", *candidate.get("reasons", [])]
        if not _policy_allows(policy, requested_backend):
            reasons.append(f"policy disallows backend: {requested_backend}")
        return {
            "requested_backend": requested_backend,
            "selected_backend": candidate.get("selected_backend", candidate.get("backend")),
            "backend": candidate.get("backend"),
            "status": candidate.get("status"),
            "candidate": candidate,
            "reasons": reasons,
        }

    preferred_order = policy.get("preferred_order", ["slurm", "kaggle", "local"])
    if not isinstance(preferred_order, list) or not preferred_order:
        raise ValueError("backend_policy.preferred_order must be a non-empty list")

    candidates: list[dict[str, Any]] = []
    reasons: list[str] = []
    for backend in preferred_order:
        if backend not in {"slurm", "kaggle", "local"}:
            reasons.append(f"unsupported backend in preferred_order: {backend}")
            continue
        if not _policy_allows(policy, backend):
            reasons.append(f"policy disallows backend: {backend}")
            continue
        candidate = _candidate_for_backend(backend)
        candidates.append(candidate)
        reasons.extend(f"{backend}: {reason}" for reason in candidate.get("reasons", []))
        if _candidate_available(candidate):
            return {
                "requested_backend": "auto",
                "selected_backend": candidate.get("selected_backend", candidate.get("backend")),
                "backend": candidate.get("backend"),
                "status": candidate.get("status"),
                "candidate": candidate,
                "candidates": candidates,
                "reasons": reasons,
            }

    return {
        "requested_backend": "auto",
        "selected_backend": None,
        "backend": None,
        "status": "unavailable",
        "candidate": None,
        "candidates": candidates,
        "reasons": reasons or ["no allowed backend candidates were available"],
    }
