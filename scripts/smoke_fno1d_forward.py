"""Run a CPU-sized FNO1D forward smoke test."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONFIG_PATH = ROOT / "configs" / "task1_baseline.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def main() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        print("torch unavailable; skipping FNO1D forward smoke test.")
        return {"status": "skipped", "reason": "torch unavailable"}

    from src.models.fno1d import FNO1D

    config = load_config()
    data_config = config["data"]
    model_config = config["model"]
    torch.manual_seed(int(config["train"]["seed"]))
    x = torch.randn(2, int(model_config["in_steps"]), int(data_config["spatial_points"]))
    model = FNO1D(
        in_steps=int(model_config["in_steps"]),
        out_steps=int(model_config["out_steps"]),
        width=int(model_config["width"]),
        modes=int(model_config["modes"]),
        depth=int(model_config["depth"]),
        padding=int(model_config["padding"]),
    )
    y = model(x)
    expected_shape = (2, int(model_config["out_steps"]), int(data_config["spatial_points"]))
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(f"Expected output shape {expected_shape}, got {tuple(y.shape)}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {"status": "ok", "output_shape": tuple(y.shape), "parameter_count": parameter_count}
    print(f"FNO1D forward ok: output_shape={tuple(y.shape)} parameter_count={parameter_count}")
    return result


if __name__ == "__main__":
    main()
