from pathlib import Path

import yaml

from scripts.render_slurm_jobs import main
from src.experiment.backend_config import PLACEHOLDER_PATTERN


def test_render_debug_slurm_job_from_venv_config(tmp_path: Path) -> None:
    backend_config = tmp_path / "compute_backend.local.yaml"
    output_dir = tmp_path / "slurm_job_files"
    backend_config.write_text(
        yaml.safe_dump(
            {
                "local": {"project_root": ".", "source_of_truth": True},
                "slurm": {
                    "enabled": True,
                    "host": "private-host",
                    "user": "private-user",
                    "remote_project_dir": "/remote/project",
                    "env_type": "venv",
                    "conda_env": "",
                    "env_path": "/remote/env",
                    "activate_script": "/remote/env/bin/activate",
                    "python_bin": "/remote/env/bin/python",
                    "pip_bin": "/remote/env/bin/pip",
                    "partition": "GPU",
                    "account": "",
                    "gpus": 2,
                    "gres": "gpu:4090:2",
                    "cpus_per_task": 4,
                    "memory": "32G",
                    "time_limit": "00:30:00",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--backend-config",
            str(backend_config),
            "--job",
            "debug_environment",
            "--output-dir",
            str(output_dir),
        ]
    )

    rendered = output_dir / "debug_environment.sbatch"
    content = rendered.read_text(encoding="utf-8")
    assert result == 0
    assert rendered.is_file()
    assert "source /remote/env/bin/activate" in content
    assert "#SBATCH --gres=gpu:4090:2" in content
    assert not PLACEHOLDER_PATTERN.search(content)
    assert "sbatch " not in content


def test_render_train_slurm_job_uses_train_config_without_placeholders(tmp_path: Path) -> None:
    backend_config = tmp_path / "compute_backend.local.yaml"
    output_dir = tmp_path / "slurm_job_files"
    backend_config.write_text(
        yaml.safe_dump(
            {
                "local": {"project_root": ".", "source_of_truth": True},
                "slurm": {
                    "enabled": True,
                    "host": "private-host",
                    "user": "private-user",
                    "remote_project_dir": "/remote/project",
                    "env_type": "venv",
                    "conda_env": "",
                    "env_path": "/remote/env",
                    "activate_script": "/remote/env/bin/activate",
                    "python_bin": "/remote/env/bin/python",
                    "pip_bin": "/remote/env/bin/pip",
                    "partition": "GPU",
                    "account": "",
                    "gpus": 1,
                    "gres": "gpu:4090:1",
                    "cpus_per_task": 4,
                    "memory": "32G",
                    "time_limit": "00:30:00",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--backend-config",
            str(backend_config),
            "--job",
            "train_task1_minimal",
            "--train-config",
            "configs/task1_a4_remote_min_train.yaml",
            "--output-dir",
            str(output_dir),
        ]
    )

    rendered = output_dir / "train_task1_minimal.sbatch"
    content = rendered.read_text(encoding="utf-8")
    assert result == 0
    assert rendered.is_file()
    assert 'cd "/remote/project"' in content
    assert "source /remote/env/bin/activate" in content
    assert 'python scripts/train_task1_minimal.py --config "configs/task1_a4_remote_min_train.yaml"' in content
    assert "#SBATCH --output=slurm_logs/%x-%j.out" in content
    assert "#SBATCH --error=slurm_logs/%x-%j.err" in content
    assert not PLACEHOLDER_PATTERN.search(content)
    assert "sbatch " not in content
