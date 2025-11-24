from __future__ import annotations

RUN_DIR = "runs/2025-11-22_10-33-12"

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lasernet.dataset import SliceSequenceDataset
from lasernet.model.CNN_LSTM import CNN_LSTM
#from train import evaluate_test     # reuse your exact evaluation function
from lasernet.utils import plot_losses       # optional
from tqdm import tqdm
from lasernet.utils import visualize_prediction


def evaluate_test(model, test_loader, device, run_dir):
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)

    model.eval()
    test_mse = 0.0
    test_mae = 0.0
    num_test_samples = 0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in test_pbar:
            context = batch["context"].float().to(device)
            target = batch["target"].float().to(device)
            target_mask = batch["target_mask"].to(device)

            pred = model(context)

            mask = target_mask.unsqueeze(1)

            mse = nn.functional.mse_loss(pred[mask], target[mask])
            mae = nn.functional.l1_loss(pred[mask], target[mask])

            batch_size = context.size(0)
            test_mse += mse.item() * batch_size
            test_mae += mae.item() * batch_size
            num_test_samples += batch_size

            test_pbar.set_postfix({"mse": f"{mse.item():.4f}", "mae": f"{mae:.4f}"})

    avg_test_mse = test_mse / max(1, num_test_samples)
    avg_test_mae = test_mae / max(1, num_test_samples)

    print(f"Test MSE: {avg_test_mse:.4f}")
    print(f"Test MAE: {avg_test_mae:.4f}")

    # ----------- SAVE VISUALIZATION WITHOUT OVERWRITING -----------
    test_batch = next(iter(test_loader))
    test_context = test_batch["context"].float()
    test_target = test_batch["target"].float()
    test_pred = model(test_context.to(device)).detach().cpu()



    vis_path = run_dir / "visualizations" / "test_prediction_best_model.png"
    visualize_prediction(
        context=test_context,
        target=test_target,
        prediction=test_pred,
        save_path=str(vis_path),
        sample_idx=0
    )

    print(f"\nSaved visualization to: {vis_path}")

    # ----------- SAVE METRICS -----------
    test_results = {
        "mse": avg_test_mse,
        "mae": avg_test_mae,
        "num_samples": num_test_samples,
    }

    with open(run_dir / "test_results_best_model.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"Saved test results to: {run_dir / 'test_results_best_model.json'}")

    return test_results


# ---------------------------------------------------------------------
# 1. SELECT RUN DIRECTORY (only thing you need to change)
# ---------------------------------------------------------------------
RUN_DIR = Path("runs/lr3_4")   # <--- CHANGE FOR EACH RUN

CKPT_PATH = RUN_DIR / "checkpoints" / "best_model.pt"
CONFIG_PATH = RUN_DIR / "config.json"

assert CKPT_PATH.exists(), f"Best model not found: {CKPT_PATH}"
assert CONFIG_PATH.exists(), f"Config file not found: {CONFIG_PATH}"

print(f"Loading run from: {RUN_DIR}")
print()


# ---------------------------------------------------------------------
# 2. LOAD CONFIG (same as in train.py)
# ---------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

seq_len = config["dataset"]["sequence_length"]
train_ratio, val_ratio, test_ratio = config["dataset"]["split_ratio"]

print("Loaded config:")
print(f"  seq_len = {seq_len}")
print(f"  split   = train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}")
print()


# ---------------------------------------------------------------------
# 3. SELECT DEVICE
# ---------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
print()


# ---------------------------------------------------------------------
# 4. CREATE TEST DATASET + LOADER (identical to train.py)
# ---------------------------------------------------------------------
test_dataset = SliceSequenceDataset(
    field="temperature",
    plane="xz",
    split="test",
    sequence_length=seq_len,
    target_offset=1,
    preload=True,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

print(f"Loaded test dataset with {len(test_dataset)} samples.")
print()


# ---------------------------------------------------------------------
# 5. LOAD MODEL + BEST CHECKPOINT
# ---------------------------------------------------------------------
model = CNN_LSTM().to(device)

print(f"Loading best model: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"  Best epoch: {ckpt['epoch']}")
print(f"  Val loss:   {ckpt['val_loss']:.4f}")
print()


#criterion = nn.MSELoss()


# ---------------------------------------------------------------------
# 6. RUN EVALUATION (reuses your evaluate_test)
# ---------------------------------------------------------------------
test_results = evaluate_test(
    model=model,
    test_loader=test_loader,
    device=device,
    run_dir=RUN_DIR,
)


# Rename test image so it doesn't overwrite existing one
old = RUN_DIR / "visualizations" / "test_prediction.png"
new = RUN_DIR / "visualizations" / "test_prediction_best_model.png"

if old.exists():
    old.rename(new)

# Save results like train.py
with open(RUN_DIR / "test_results_best_model.json", "w") as f:
    json.dump(test_results, f, indent=2)

print("\nSaved test results to:", RUN_DIR / "test_results_best_model.json")
print("Saved visualization to:", RUN_DIR / "visualizations" / "test_prediction_best_model.png")
print("\nDone.")
