"""Run one Task 1 FNO1D training step and save a tiny checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hdf5_utils import find_main_array_key


CONFIG_PATH = ROOT / "configs" / "task1_baseline.yaml"


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def choose_device(device_config: str, torch: Any) -> str:
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def main() -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch unavailable; skipping one-batch Task 1 training smoke test.")
        return {"status": "skipped", "reason": "torch unavailable"}

    from src.data.task1_dataset import Task1TrajectoryDataset
    from src.models.fno1d import FNO1D
    from src.train.checkpointing import load_checkpoint, save_checkpoint

    config = load_config()
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    output_config = config["outputs"]

    torch.manual_seed(int(train_config["seed"]))
    val_path = resolve_path(data_config["val_path"])
    key = data_config.get("hdf5_key") or find_main_array_key(val_path)
    dataset = Task1TrajectoryDataset(
        val_path,
        input_steps=int(data_config["input_steps"]),
        total_steps=int(data_config["total_steps"]),
        key=key,
    )
    if dataset.shape[1] < int(data_config["input_steps"]) + 1:
        dataset.close()
        message = (
            f"Validation data has {dataset.shape[1]} time steps; one-batch training "
            f"needs at least {int(data_config['input_steps']) + 1}."
        )
        print(message)
        return {"status": "skipped", "reason": message}

    device = choose_device(str(train_config["device"]), torch)
    model = FNO1D(
        in_steps=int(model_config["in_steps"]),
        out_steps=int(model_config["out_steps"]),
        width=int(model_config["width"]),
        modes=int(model_config["modes"]),
        depth=int(model_config["depth"]),
        padding=int(model_config["padding"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    loader = DataLoader(dataset, batch_size=int(train_config["batch_size"]), shuffle=False)

    try:
        batch = next(iter(loader))
        inputs = batch["input"].to(device)
        target_next = batch["full"][:, int(data_config["input_steps"]) : int(data_config["input_steps"]) + 1, :].to(device)
        model.train()
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, target_next)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())

        checkpoint_path = resolve_path(output_config["checkpoint_dir"]) / "task1_fno1d_smoke.pt"
        metadata = {"config": config, "step": 1, "loss": loss_value}
        save_checkpoint(checkpoint_path, model, optimizer=optimizer, metadata=metadata)
        reloaded_model = FNO1D(
            in_steps=int(model_config["in_steps"]),
            out_steps=int(model_config["out_steps"]),
            width=int(model_config["width"]),
            modes=int(model_config["modes"]),
            depth=int(model_config["depth"]),
            padding=int(model_config["padding"]),
        ).to(device)
        reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters())
        loaded_metadata = load_checkpoint(
            checkpoint_path,
            reloaded_model,
            optimizer=reloaded_optimizer,
            map_location=device,
        )
    finally:
        dataset.close()

    result = {
        "status": "ok",
        "loss": loss_value,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_loaded": loaded_metadata.get("step") == 1,
    }
    print(
        "One-batch train ok: "
        f"loss={loss_value:.8f} checkpoint={result['checkpoint_path']} "
        f"checkpoint_loaded={result['checkpoint_loaded']}"
    )
    return result


if __name__ == "__main__":
    main()
