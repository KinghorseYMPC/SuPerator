"""Smoke-test the pdeagent Task 1 adapter with synthetic or real data.

Reads configs/pdeagent_task1_adapter_smoke.yaml, runs a tiny training
(max 1 epoch, 2 batches), saves a checkpoint, and performs rollout smoke.

If torch is unavailable, exits 0 with a skip message.
If real HDF5 data is missing, falls back to synthetic data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    config_path = ROOT / "configs" / "pdeagent_task1_adapter_smoke.yaml"
    if not config_path.exists():
        print(f"[ERROR] config not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Check torch
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as exc:
        print(f"[SKIP] torch not available: {exc}")
        print("Smoke test skipped — install torch for your local CUDA/CPU environment.")
        return 0

    from src.adapters.pdeagent.model_adapter import (
        PdeAgentBaselineConfig,
        build_pdeagent_task1_model,
    )
    from src.adapters.pdeagent.dataset_adapter import PdeAgentTask1WindowDataset
    from src.adapters.pdeagent.task1_training import (
        train_one_epoch,
        evaluate_one_step,
        resolve_device,
        set_seed,
        _save_checkpoint,
    )
    from src.adapters.pdeagent.inference_adapter import autoregressive_predict

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    output_cfg = cfg["outputs"]

    device = resolve_device(str(train_cfg.get("device", "cpu")))

    mconfig = PdeAgentBaselineConfig(
        input_steps=int(model_cfg["input_steps"]),
        output_steps=int(model_cfg["output_steps"]),
        width=int(model_cfg["width"]),
        modes=int(model_cfg["modes"]),
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        chunk_size=int(model_cfg.get("output_steps", 10)),
        use_film=False,
    )
    model = build_pdeagent_task1_model(mconfig).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model built: {param_count:,} parameters")

    # Try real data first, fall back to synthetic
    val_path = ROOT / data_cfg["val_path"]
    use_synthetic = not val_path.exists()

    if use_synthetic:
        print("[INFO] Real HDF5 not found, using synthetic data for smoke test")
        # Create synthetic HDF5
        import tempfile
        import h5py

        tmpdir = tempfile.mkdtemp()
        syn_path = os.path.join(tmpdir, "fake_task1_val.hdf5")
        rng = np.random.RandomState(42)
        syn_data = rng.randn(100, 200, 256).astype(np.float32) * 0.1
        with h5py.File(syn_path, "w") as f:
            f.create_dataset("tensor", data=syn_data)
        val_path = Path(syn_path)
        train_samples = 80
        dev_samples = 20
    else:
        train_samples = int(data_cfg.get("train_samples", 80))
        dev_samples = int(data_cfg.get("dev_samples", 20))

    # Dataset
    input_steps = mconfig.input_steps
    output_steps = mconfig.output_steps
    full_dataset = PdeAgentTask1WindowDataset(
        hdf5_path=str(val_path),
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=True,
    )

    total_traj = full_dataset.total_trajectories
    win_per_traj = len(full_dataset) // total_traj
    print(f"[OK] Dataset: {total_traj} trajectories, {len(full_dataset)} windows")

    from torch.utils.data import DataLoader, Subset

    train_indices = list(range(min(train_samples, total_traj) * win_per_traj))
    dev_start = min(train_samples, total_traj) * win_per_traj
    dev_end = dev_start + min(dev_samples, total_traj - min(train_samples, total_traj)) * win_per_traj
    dev_indices = list(range(dev_start, dev_end))

    batch_size = int(train_cfg.get("batch_size", 4))
    train_loader = DataLoader(Subset(full_dataset, train_indices),
                              batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(Subset(full_dataset, dev_indices),
                            batch_size=batch_size, shuffle=False)

    # Optimizer
    lr = float(train_cfg.get("learning_rate", 0.001))
    wd = float(train_cfg.get("weight_decay", 1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Tiny training
    set_seed(42)
    max_batches = int(train_cfg.get("max_train_batches_per_epoch", 2))
    epochs = int(train_cfg.get("epochs", 1))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     max_batches=max_batches)
        dev_loss = evaluate_one_step(model, dev_loader, device, max_batches=max_batches)
        print(f"[OK] Epoch {epoch}: train_loss={train_loss:.6f}, dev_loss={dev_loss:.6f}")

    # Save checkpoint
    checkpoint_dir = ROOT / output_cfg.get("checkpoint_dir", "outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{cfg['experiment_id']}_best.pt"
    _save_checkpoint(ckpt_path, model, optimizer)
    print(f"[OK] Checkpoint saved: {ckpt_path}")

    # Rollout smoke
    rng = np.random.RandomState(99)
    initial = rng.randn(2, input_steps, 256).astype(np.float32) * 0.1
    pred = autoregressive_predict(model, initial, total_steps=200,
                                  input_steps=input_steps, device=str(device))
    print(f"[OK] Rollout shape: {pred.shape}")

    error_init = np.abs(pred[:, :input_steps, :] - initial).max()
    print(f"[OK] First {input_steps} step max error: {error_init:.6f}")

    # Score
    from src.adapters.pdeagent.scoring import segment_scores
    scores = segment_scores(pred[:, input_steps:, :], np.zeros_like(pred[:, input_steps:, :]))
    print("[OK] Segment scores (vs zero baseline):")
    for k in ("score1", "score2", "score3", "score_total"):
        print(f"  {k}: {scores.get(k)}")

    full_dataset.close()
    if use_synthetic:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\nSmoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
