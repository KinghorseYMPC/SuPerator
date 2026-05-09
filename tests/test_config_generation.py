from pathlib import Path

import yaml

from src.experiment import config_generation


def test_deep_update_does_not_mutate_base() -> None:
    base = {"train": {"epochs": 1, "batch_size": 2}, "model": {"width": 8}}
    overrides = {"train": {"epochs": 3}, "stage": "A6"}

    merged = config_generation.deep_update(base, overrides)

    assert merged == {"train": {"epochs": 3, "batch_size": 2}, "model": {"width": 8}, "stage": "A6"}
    assert base == {"train": {"epochs": 1, "batch_size": 2}, "model": {"width": 8}}


def test_generate_experiment_configs(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "project_name": "SuPerator",
                "stage": "base",
                "experiment_id": "base_exp",
                "train": {"epochs": 1, "batch_size": 2},
                "model": {"width": 8, "depth": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_config = tmp_path / "generated" / "exp.yaml"
    suite_config = tmp_path / "suite.yaml"
    suite_config.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "experiment_id": "exp_a6",
                        "base_config": str(base_config),
                        "output_config": str(output_config),
                        "overrides": {
                            "stage": "A6_task1_experiment_suite",
                            "experiment_id": "exp_a6",
                            "train": {"epochs": 3},
                        },
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    generated = config_generation.generate_experiment_configs(suite_config)

    assert generated[0]["experiment_id"] == "exp_a6"
    loaded = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert loaded["stage"] == "A6_task1_experiment_suite"
    assert loaded["experiment_id"] == "exp_a6"
    assert loaded["train"] == {"epochs": 3, "batch_size": 2}
    assert yaml.safe_load(base_config.read_text(encoding="utf-8"))["experiment_id"] == "base_exp"


def test_save_yaml_rejects_sensitive_keys(tmp_path: Path) -> None:
    try:
        config_generation.save_yaml({"kaggle_key": "value"}, tmp_path / "bad.yaml")
    except ValueError as exc:
        assert "sensitive-looking keys" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sensitive key rejection")
