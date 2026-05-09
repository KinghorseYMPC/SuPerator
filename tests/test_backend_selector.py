import sys
import types
from pathlib import Path

import yaml

from src.experiment import backend_selector


def test_detect_local_backend_with_mocked_torch(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = backend_selector.detect_local_backend()

    assert result["torch_available"] is True
    assert result["cuda_available"] is True
    assert result["device_count"] == 2
    assert result["selected_backend"] == "local_gpu"


def test_detect_kaggle_backend_without_auth_check(monkeypatch) -> None:
    monkeypatch.setattr(backend_selector.shutil, "which", lambda name: "kaggle" if name == "kaggle" else None)

    result = backend_selector.detect_kaggle_backend(quick_check=False)

    assert result["cli_available"] is True
    assert result["auth_checked"] is False
    assert result["status"] == "available_or_recoverable"


def test_detect_slurm_backend_from_local_config(tmp_path: Path) -> None:
    config = tmp_path / "compute_backend.local.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "slurm": {
                    "enabled": True,
                    "host": "private-host",
                    "remote_project_dir": "/private/project",
                    "env_type": "direct_python",
                }
            }
        ),
        encoding="utf-8",
    )

    result = backend_selector.detect_slurm_backend(config)

    assert result["status"] == "requires_manual_submit"
    assert result["required_fields_present"] is True


def test_select_backend_prefers_slurm_then_kaggle_then_local(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_selector,
        "detect_slurm_backend",
        lambda: {"backend": "slurm", "status": "requires_manual_submit", "reasons": []},
    )
    monkeypatch.setattr(
        backend_selector,
        "detect_kaggle_backend",
        lambda quick_check=False: {"backend": "kaggle", "status": "available_or_recoverable", "reasons": []},
    )
    monkeypatch.setattr(
        backend_selector,
        "detect_local_backend",
        lambda: {"backend": "local", "selected_backend": "local_cpu", "python_available": True, "status": "available", "reasons": []},
    )

    selected = backend_selector.select_backend(
        {"preferred_order": ["slurm", "kaggle", "local"], "allow_slurm": True, "allow_kaggle": True, "allow_local": True}
    )

    assert selected["backend"] == "slurm"
    assert selected["status"] == "requires_manual_submit"


def test_select_backend_falls_back_to_local(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_selector,
        "detect_slurm_backend",
        lambda: {"backend": "slurm", "status": "unavailable", "reasons": ["missing"]},
    )
    monkeypatch.setattr(
        backend_selector,
        "detect_kaggle_backend",
        lambda quick_check=False: {"backend": "kaggle", "status": "unavailable", "reasons": ["missing"]},
    )
    monkeypatch.setattr(
        backend_selector,
        "detect_local_backend",
        lambda: {"backend": "local", "selected_backend": "local_cpu", "python_available": True, "status": "available", "reasons": []},
    )

    selected = backend_selector.select_backend({"preferred_order": ["slurm", "kaggle", "local"]})

    assert selected["backend"] == "local"
    assert selected["selected_backend"] == "local_cpu"
