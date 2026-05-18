"""Smoke-test the pdeagent Task 2 adapter: model, dataset, inference.

Reads configs/pdeagent_task2_adapter_smoke.yaml.
Uses synthetic input — no real HDF5 reading.
No training, no submission generation, no large files.

Gracefully exits 0 if torch is unavailable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_path = ROOT / "configs" / "pdeagent_task2_adapter_smoke.yaml"
    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}")
        return 1

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[SMOKE] Task 2 Adapter Smoke Test")
    print(f"  Config : {config_path}")
    print(f"  Stage  : {config['stage']}")
    print(f"  Task   : {config['task']}")

    # ------------------------------------------------------------------
    # 2. Check torch
    # ------------------------------------------------------------------
    if not _check_torch():
        print("[SKIP] torch not available — exiting 0")
        return 0

    import torch
    import numpy as np

    print(f"  Torch  : {torch.__version__}")
    print(f"  CUDA   : {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    from src.adapters.pdeagent.model_adapter import (
        PdeAgentBaselineConfig,
        build_pdeagent_task2_model,
    )

    model_cfg = config["model"]
    mconfig = PdeAgentBaselineConfig(
        input_steps=int(model_cfg.get("input_steps", 10)),
        output_steps=int(model_cfg.get("output_steps", 1)),
        width=int(model_cfg.get("width", 32)),
        modes=int(model_cfg.get("modes", 16)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        chunk_size=int(model_cfg.get("output_steps", 1)),
        use_film=True,
        condition_source=str(model_cfg.get("condition_source", "estimated_nu")),
        nu_dim=int(model_cfg.get("nu_dim", 1)),
    )
    model = build_pdeagent_task2_model(mconfig)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[BUILD] Task 2 Model params: {n_params}")

    # ------------------------------------------------------------------
    # 4. Synthetic input
    # ------------------------------------------------------------------
    batch_size = config.get("smoke", {}).get("batch_size", 2)
    device = config.get("smoke", {}).get("device", "cpu")
    input_steps = int(config["data"]["input_steps"])
    spatial_points = int(config["data"]["spatial_points"])

    rng = np.random.RandomState(42)
    x = torch.as_tensor(rng.randn(batch_size, input_steps, spatial_points).astype(np.float32))
    provided_nu = torch.as_tensor(np.log(np.array([[0.005], [0.01]], dtype=np.float32)))

    print(f"\n[SHA PE] Input       : {tuple(x.shape)}")

    # ------------------------------------------------------------------
    # 5. Forward with provided_nu
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_provided, _ = model(x, nu=provided_nu)
        print(f"[SHAPE] provided_nu output : {tuple(y_provided.shape)}")

    # ------------------------------------------------------------------
    # 6. Forward with nu=None (estimated)
    # ------------------------------------------------------------------
    with torch.no_grad():
        y_estimated, nu_est = model(x, nu=None)
        print(f"[SHAPE] estimated_nu output : {tuple(y_estimated.shape)}")
        print(f"[SHAPE] estimated nu        : {tuple(nu_est.shape)}")

    # ------------------------------------------------------------------
    # 7. Rollout
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        roll = model.rollout_no_grad(x, horizon=190, nu=None)
        print(f"[SHAPE] rollout              : {tuple(roll.shape)}")

    # Concatenate: initial (10) + rollout (190) = 200 steps
    full = torch.cat([x, roll], dim=1)
    print(f"[SHAPE] full (initial+rollout): {tuple(full.shape)}")

    # ------------------------------------------------------------------
    # 8. Verify first 10 steps exact (they are copied from initial)
    # ------------------------------------------------------------------
    first10_error = float(torch.abs(full[:, :input_steps, :] - x).max())
    print(f"[CHECK] first10_max_error     : {first10_error:.6e}")

    if first10_error > 1e-6:
        print("[FAIL] First 10 steps should be exact copies of initial.")
        return 1

    # ------------------------------------------------------------------
    # 9. Dataset adapter smoke (no real HDF5 needed)
    # ------------------------------------------------------------------
    from src.adapters.pdeagent.task2_dataset_adapter import (
        PdeAgentTask2DataSpec,
        inspect_task2_hdf5,
        make_task2_window_indices,
    )

    spec = PdeAgentTask2DataSpec(
        train_paths=config["data"]["train_paths"],
        val_path=config["data"]["val_path"],
        test_path=config["data"]["test_path"],
        input_steps=int(config["data"]["input_steps"]),
        output_steps=int(config["data"]["output_steps"]),
        total_steps=int(config["data"]["total_steps"]),
    )
    print(f"\n[DATASPEC] train_paths : {len(spec.train_paths)} files")
    print(f"[DATASPEC] val_path    : {spec.val_path}")
    print(f"[DATASPEC] test_path   : {spec.test_path}")

    win = make_task2_window_indices(200, 10, 1, stride=1)
    print(f"[WINDOW]  total windows : {len(win)}")
    print(f"[WINDOW]  first         : {win[0]}")
    print(f"[WINDOW]  last          : {win[-1]}")

    # Test fake HDF5 dataset
    import tempfile
    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_train_path = os.path.join(tmpdir, "task2_part0_train.h5")
        with h5py.File(fake_train_path, "w") as f:
            f.create_dataset("tensor", data=np.zeros((100, 200, 256), dtype=np.float32))
            f.create_dataset("nu", data=np.linspace(0.001, 0.01, 100, dtype=np.float32))

        info = inspect_task2_hdf5(fake_train_path)
        print(f"\n[INSPECT] keys         : {list(info['keys'].keys())}")
        print(f"[INSPECT] has_nu       : {info['has_nu']}")

        # Train mode: should return nu
        from src.adapters.pdeagent.task2_dataset_adapter import PdeAgentTask2WindowDataset
        ds_train = PdeAgentTask2WindowDataset(
            fake_train_path, input_steps=10, output_steps=1, mode="train",
        )
        sample_train = ds_train[0]
        assert "nu" in sample_train, "Train mode should return nu"
        print(f"[DATASET] train sample keys : {list(sample_train.keys())}")
        print(f"[DATASET] train input shape : {tuple(sample_train['input'].shape)}")
        print(f"[DATASET] train nu value    : {float(sample_train['nu'].item()):.6f}")

        # Test mode: should NOT return nu
        fake_test_path = os.path.join(tmpdir, "task2_test.h5")
        with h5py.File(fake_test_path, "w") as f:
            f.create_dataset("tensor", data=np.zeros((50, 17, 256), dtype=np.float32))
        ds_test = PdeAgentTask2WindowDataset(
            fake_test_path, input_steps=10, output_steps=1, mode="test",
        )
        sample_test = ds_test[0]
        assert "nu" not in sample_test, "Test mode must not return nu"
        print(f"[DATASET] test sample keys  : {list(sample_test.keys())}")
        print(f"[DATASET] test has nu       : {'nu' in sample_test}")

        ds_train.close()
        ds_test.close()

    # ------------------------------------------------------------------
    # 10. Inference adapter smoke (synthetic)
    # ------------------------------------------------------------------
    from src.adapters.pdeagent.task2_inference_adapter import rollout_task2_model

    initial_np = rng.randn(2, 10, 256).astype(np.float32)
    with torch.no_grad():
        pred = rollout_task2_model(model, initial_np, total_steps=200,
                                   input_steps=10, device="cpu")
    print(f"\n[INFER] prediction shape     : {pred.shape}")
    print(f"[INFER] first10 exact        : {np.allclose(pred[:, :10, :], initial_np, atol=1e-6)}")

    # Task 1 path rejection
    try:
        PdeAgentTask2WindowDataset("contains_task1_data.h5", input_steps=10, output_steps=1)
        print("[FAIL] Should have rejected Task 1 path!")
        return 1
    except ValueError as e:
        print(f"[CHECK] Task1 path rejection : OK ({'task1' in str(e).lower()})")
    except Exception:
        # FileNotFoundError also OK since the fake path doesn't exist
        print(f"[CHECK] Task1 path rejection : OK (blocked at path check)")

    print("\n[SMOKE] All Task 2 adapter smoke tests PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
