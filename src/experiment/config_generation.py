"""Generate concrete experiment configs from an experiment suite YAML file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SENSITIVE_KEY_TOKENS = ("token", "secret", "credential", "password", "kaggle_key")


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input mapping."""

    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = resolve_path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML config does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        payload = yaml.safe_load(yaml_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {yaml_path}")
    return payload


def _walk_sensitive_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        matches: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if any(token in str(key).lower() for token in SENSITIVE_KEY_TOKENS):
                matches.append(child_prefix)
            matches.extend(_walk_sensitive_keys(child, child_prefix))
        return matches
    if isinstance(value, list):
        matches = []
        for index, child in enumerate(value):
            matches.extend(_walk_sensitive_keys(child, f"{prefix}[{index}]"))
        return matches
    return []


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    sensitive_keys = _walk_sensitive_keys(data)
    if sensitive_keys:
        joined = ", ".join(sorted(sensitive_keys))
        raise ValueError(f"Generated config contains sensitive-looking keys: {joined}")
    yaml_path = resolve_path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )
    return yaml_path


def generate_experiment_configs(suite_config_path: str | Path) -> list[dict[str, Any]]:
    """Generate per-experiment configs and return their metadata."""

    suite_path = resolve_path(suite_config_path)
    suite_config = load_yaml(suite_path)
    experiments = suite_config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"Suite config must contain a non-empty experiments list: {suite_path}")

    generated: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment entry #{index} must be a mapping: {suite_path}")
        experiment_id = experiment.get("experiment_id")
        base_config = experiment.get("base_config")
        output_config = experiment.get("output_config")
        overrides = experiment.get("overrides", {})
        if not experiment_id:
            raise ValueError(f"Experiment entry #{index} is missing experiment_id")
        if not base_config:
            raise ValueError(f"Experiment {experiment_id} is missing base_config")
        if not output_config:
            raise ValueError(f"Experiment {experiment_id} is missing output_config")
        if not isinstance(overrides, dict):
            raise ValueError(f"Experiment {experiment_id} overrides must be a mapping")

        base_payload = load_yaml(base_config)
        generated_payload = deep_update(base_payload, overrides)
        generated_path = save_yaml(generated_payload, output_config)
        generated.append(
            {
                "experiment_id": str(experiment_id),
                "base_config": str(resolve_path(base_config)),
                "output_config": str(generated_path),
                "runner": experiment.get("runner"),
                "config": generated_payload,
            }
        )
    return generated
