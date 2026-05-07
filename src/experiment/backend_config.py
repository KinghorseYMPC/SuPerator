"""Backend configuration helpers for local-only SLURM job rendering."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from src.experiment.remote_manifest import ROOT, resolve_path


PLACEHOLDER_PATTERN = re.compile(r"<[^>\n]+>")
SENSITIVE_KEYS = {"host", "user", "token", "secret", "key", "credential"}


def load_backend_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML backend config file."""

    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"backend config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"backend config must be a YAML mapping: {config_path}")
    return data


def _required_text(config: dict[str, Any], key: str, env_type: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"slurm env_type {env_type!r} requires non-empty {key!r}")
    return value.strip()


def render_environment_setup(slurm_config: dict[str, Any]) -> tuple[str, str]:
    """Return a shell setup snippet and Python command for the SLURM config."""

    env_type = str(slurm_config.get("env_type", "")).strip()
    if env_type == "conda":
        conda_env = _required_text(slurm_config, "conda_env", env_type)
        snippet = "\n".join(
            [
                'source "$(conda info --base)/etc/profile.d/conda.sh"',
                f"conda activate {shlex.quote(conda_env)}",
            ]
        )
        return snippet, "python"
    if env_type == "venv":
        activate_script = _required_text(slurm_config, "activate_script", env_type)
        return f"source {shlex.quote(activate_script)}", "python"
    if env_type == "direct_python":
        python_bin = _required_text(slurm_config, "python_bin", env_type)
        return "# direct_python: no environment activation", shlex.quote(python_bin)

    supported = "conda, venv, direct_python"
    raise ValueError(f"unsupported slurm env_type {env_type!r}; expected one of: {supported}")


def _format_directive(name: str, value: str | None) -> str:
    if value is None or not str(value).strip():
        return f"# No SLURM {name} directive configured"
    return f"#SBATCH {name}={value}"


def build_sbatch_context(slurm_config: dict[str, Any], job_name: str, config_path: str | Path | None = None) -> dict[str, str]:
    """Build common placeholder values for SLURM sbatch templates."""

    env_setup, python_cmd = render_environment_setup(slurm_config)
    project_dir = _required_text(slurm_config, "remote_project_dir", "slurm")
    gres = str(slurm_config.get("gres") or "").strip()
    gpus = str(slurm_config.get("gpus") or "").strip()
    if not gres and gpus:
        gres = f"gpu:{gpus}"
    account = str(slurm_config.get("account") or "").strip()

    return {
        "JOB_NAME": job_name,
        "PARTITION": str(slurm_config.get("partition") or ""),
        "GRES_DIRECTIVE": _format_directive("--gres", gres),
        "ACCOUNT_DIRECTIVE": _format_directive("--account", account),
        "CPUS_PER_TASK": str(slurm_config.get("cpus_per_task") or ""),
        "MEMORY": str(slurm_config.get("memory") or ""),
        "TIME_LIMIT": str(slurm_config.get("time_limit") or ""),
        "PROJECT_DIR": project_dir,
        "CONFIG_PATH": str(config_path or "configs/task1_a3_min_train.yaml"),
        "ENV_SETUP": env_setup,
        "PYTHON_CMD": python_cmd,
    }


def render_sbatch_from_template(
    template_path: str | Path,
    output_path: str | Path,
    context: dict[str, Any],
) -> Path:
    """Render an sbatch template to slurm_job_files without running sbatch."""

    template_file = resolve_path(template_path)
    if not template_file.is_file():
        raise FileNotFoundError(f"sbatch template does not exist: {template_file}")

    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rendered = template_file.read_text(encoding="utf-8")
    for key, value in context.items():
        rendered = rendered.replace(f"<{key}>", str(value))

    leftovers = sorted(set(PLACEHOLDER_PATTERN.findall(rendered)))
    if leftovers:
        joined = ", ".join(leftovers)
        raise ValueError(f"rendered sbatch contains unreplaced placeholders: {joined}")

    output_file.write_text(rendered, encoding="utf-8", newline="\n")
    return output_file


def sanitize_slurm_config_for_report(config: dict[str, Any]) -> dict[str, Any]:
    """Return a SLURM config summary without identity, secret, or private path values."""

    slurm_config = config.get("slurm", config)
    if not isinstance(slurm_config, dict):
        raise ValueError("slurm config must be a mapping")

    summary: dict[str, Any] = {}
    for key in [
        "enabled",
        "env_type",
        "partition",
        "gpus",
        "gres",
        "cpus_per_task",
        "memory",
        "time_limit",
    ]:
        if key in slurm_config:
            summary[key] = slurm_config[key]

    for key in [
        "conda_env",
        "env_path",
        "activate_script",
        "python_bin",
        "pip_bin",
        "remote_project_dir",
        "account",
    ]:
        value = slurm_config.get(key)
        summary[f"{key}_configured"] = bool(isinstance(value, str) and value.strip())

    for key in list(summary):
        lower_key = key.lower()
        if any(sensitive == lower_key or sensitive in lower_key for sensitive in SENSITIVE_KEYS):
            summary.pop(key)
    return summary
