from pathlib import Path

import pytest

from src.experiment.backend_config import (
    render_environment_setup,
    render_sbatch_from_template,
    sanitize_slurm_config_for_report,
)


def test_render_environment_setup_supports_conda() -> None:
    setup, python_cmd = render_environment_setup({"env_type": "conda", "conda_env": "research"})

    assert "conda activate research" in setup
    assert python_cmd == "python"


def test_render_environment_setup_supports_venv() -> None:
    setup, python_cmd = render_environment_setup(
        {"env_type": "venv", "activate_script": "/remote/env/bin/activate"}
    )

    assert setup == "source /remote/env/bin/activate"
    assert python_cmd == "python"


def test_render_environment_setup_supports_direct_python() -> None:
    setup, python_cmd = render_environment_setup(
        {"env_type": "direct_python", "python_bin": "/remote/env/bin/python"}
    )

    assert "no environment activation" in setup
    assert python_cmd == "/remote/env/bin/python"


def test_render_environment_setup_rejects_missing_venv_activate_script() -> None:
    with pytest.raises(ValueError, match="activate_script"):
        render_environment_setup({"env_type": "venv"})


def test_render_environment_setup_rejects_missing_direct_python_bin() -> None:
    with pytest.raises(ValueError, match="python_bin"):
        render_environment_setup({"env_type": "direct_python"})


def test_render_environment_setup_rejects_unknown_env_type() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        render_environment_setup({"env_type": "module"})


def test_render_sbatch_rejects_unreplaced_placeholders(tmp_path: Path) -> None:
    template = tmp_path / "job.sbatch.template"
    output = tmp_path / "slurm_job_files" / "job.sbatch"
    template.write_text("#SBATCH --job-name=<JOB_NAME>\ncd <PROJECT_DIR>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unreplaced placeholders"):
        render_sbatch_from_template(template, output, {"JOB_NAME": "debug"})


def test_sanitize_slurm_config_for_report_omits_identity_fields() -> None:
    summary = sanitize_slurm_config_for_report(
        {
            "slurm": {
                "enabled": True,
                "host": "private-host",
                "user": "private-user",
                "remote_project_dir": "/private/project",
                "env_type": "venv",
                "activate_script": "/private/env/bin/activate",
                "python_bin": "/private/env/bin/python",
                "partition": "GPU",
                "gpus": 2,
                "gres": "gpu:4090:2",
            }
        }
    )

    rendered = str(summary)
    assert "host" not in summary
    assert "user" not in summary
    assert "private-host" not in rendered
    assert "private-user" not in rendered
    assert "/private/" not in rendered
    assert summary["remote_project_dir_configured"] is True
    assert summary["activate_script_configured"] is True
