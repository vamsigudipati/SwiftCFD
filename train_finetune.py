"""
SwiftCFD — Pass 2: Fine-Tuning
Fine-tunes a pre-trained DeepCFD UNetEx checkpoint using:
  - AdamW optimizer (lr=1e-4, weight_decay=0.02)
  - ReduceLROnPlateau scheduler (factor=0.5, patience=50)
  - Channel-weighted custom loss (MSE for Ux/Uy, MAE for p)
  - Early stopping (patience=200)

Usage:
    python train_finetune.py --checkpoint checkpoint.pt \
                             --dataX dataX.pkl \
                             --dataY dataY.pkl \
                             --output mymodel_v2.pt
"""

import os
import argparse
import random
import pickle

import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepcfd.train_functions import train_model
from deepcfd.functions import split_tensors
from deepcfd.models.UNetEx import UNetEx

# ══════════════════════════════════════════════════════════════════
# ARGUMENTS — makes script portable (no hardcoded Colab paths)
# ══════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="SwiftCFD Fine-Tuning — Pass 2")
parser.add_argument("--dataX",      default="dataX.pkl",      help="Path to input data pkl")
parser.add_argument("--dataY",      default="dataY.pkl",      help="Path to output data pkl")
parser.add_argument("--checkpoint", default="checkpoint.pt",  help="Pass 1 checkpoint to fine-tune from")
parser.add_argument("--output",     default="mymodel_v2.pt",  help="Where to save the best fine-tuned model")
parser.add_argument("--epochs",     default=2000, type=int)
parser.add_argument("--batch_size", default=64,   type=int)
parser.add_argument("--lr",         default=1e-4, type=float)
parser.add_argument("--patience",   default=200,  type=int)
parser.add_argument("--save_plot",  default="training_curves_v2.png", help="Path to save training curves")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILTERS     = [8, 16, 32, 32]
KERNEL_SIZE = 5

print(f"Using device     : {DEVICE}")
print(f"Checkpoint       : {args.checkpoint}")
print(f"Output           : {args.output}")
print(f"Epochs           : {args.epochs}")
print(f"Batch size       : {args.batch_size}")
print(f"Learning rate    : {args.lr}")
print(f"Early stop pat.  : {args.patience}")

# ══════════════════════════════════════════════════════════════════
# 2. LOAD DATA
# ══════════════════════════════════════════════════════════════════

print("\nLoading data...")
x = torch.FloatTensor(pickle.load(open(args.dataX, "rb")))
y = torch.FloatTensor(pickle.load(open(args.dataY, "rb")))
print(f"dataX: {x.shape}  dataY: {y.shape}")

# ── Shuffle (reproducible) ──────────────────────────────────────
indices = list(range(len(x)))
random.seed(42)
random.shuffle(indices)
x = x[indices]
y = y[indices]

batch, nx, ny = x.shape[0], x.shape[2], x.shape[3]

# ── Channel weights (normalise loss per field magnitude) ────────
channels_weights = torch.sqrt(
    torch.mean(y.permute(0, 2, 3, 1).reshape((batch * nx * ny, 3)) ** 2, dim=0)
).view(1, -1, 1, 1).to(DEVICE)

# ── Train / Test split (70 / 30) ────────────────────────────────
train_data, test_data = split_tensors(x, y, ratio=0.7)
train_dataset = TensorDataset(*train_data)
test_dataset  = TensorDataset(*test_data)
test_x, test_y = test_dataset[:]

print(f"Train: {len(train_dataset)} samples  |  Test: {len(test_dataset)} samples")

# ══════════════════════════════════════════════════════════════════
# 3. LOAD MODEL FROM CHECKPOINT (fine-tuning)
# ══════════════════════════════════════════════════════════════════

print(f"\nLoading checkpoint: {args.checkpoint}")
state_dict = torch.load(args.checkpoint, map_location=DEVICE)
for key in ["filters", "kernel_size", "input_shape", "architecture"]:
    state_dict.pop(key, None)

model = UNetEx(
    3, 3,
    filters     = FILTERS,
    kernel_size = KERNEL_SIZE,
    batch_norm  = False,
    weight_norm = False
)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
print("✅ Checkpoint loaded successfully")

# ══════════════════════════════════════════════════════════════════
# 4. OPTIMIZER + SCHEDULER
# ══════════════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = args.lr,
    weight_decay = 0.02
)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode     = "min",
    factor   = 0.5,
    patience = 50,
)

print(f"Optimizer : AdamW | LR: {args.lr} | Weight Decay: 0.02")
print(f"Scheduler : ReduceLROnPlateau | Factor: 0.5 | Patience: 50")

# ══════════════════════════════════════════════════════════════════
# 5. TRACKING CURVES
# ══════════════════════════════════════════════════════════════════

train_loss_curve, test_loss_curve = [], []
train_mse_curve,  test_mse_curve  = [], []
lr_curve = []

def after_epoch(scope):
    val_loss = scope["val_loss"]
    scheduler.step(val_loss)
    train_loss_curve.append(scope["train_loss"])
    test_loss_curve.append(val_loss)
    train_mse_curve.append(scope["train_metrics"]["mse"])
    test_mse_curve.append(scope["val_metrics"]["mse"])
    lr_curve.append(optimizer.param_groups[0]["lr"])

# ══════════════════════════════════════════════════════════════════
# 6. CUSTOM LOSS FUNCTION
# MSE for Ux and Uy, MAE for pressure p — channel-weighted
# ══════════════════════════════════════════════════════════════════

def loss_func(model, batch):
    x, y   = batch
    output = model(x)
    lossu  = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
               output.shape[0], 1, output.shape[2], output.shape[3])
    lossv  = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
               output.shape[0], 1, output.shape[2], output.shape[3])
    lossp  = torch.abs((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
               output.shape[0], 1, output.shape[2], output.shape[3])
    loss   = (lossu + lossv + lossp) / channels_weights
    return torch.sum(loss), output

# ══════════════════════════════════════════════════════════════════
# 7. TRAIN
# ══════════════════════════════════════════════════════════════════

print("\n🚀 Starting fine-tuned training...\n")

DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(
    model,
    loss_func,
    train_dataset,
    test_dataset,
    optimizer,
    epochs      = args.epochs,
    batch_size  = args.batch_size,
    device      = DEVICE,
    m_mse_name  = "Total MSE",
    m_mse_on_batch  = lambda scope: float(torch.sum(
        (scope["output"] - scope["batch"][1]) ** 2)),
    m_mse_on_epoch  = lambda scope: sum(scope["list"]) / len(scope["dataset"]),
    m_ux_name       = "Ux MSE",
    m_ux_on_batch   = lambda scope: float(torch.sum(
        (scope["output"][:, 0, :, :] - scope["batch"][1][:, 0, :, :]) ** 2)),
    m_ux_on_epoch   = lambda scope: sum(scope["list"]) / len(scope["dataset"]),
    m_uy_name       = "Uy MSE",
    m_uy_on_batch   = lambda scope: float(torch.sum(
        (scope["output"][:, 1, :, :] - scope["batch"][1][:, 1, :, :]) ** 2)),
    m_uy_on_epoch   = lambda scope: sum(scope["list"]) / len(scope["dataset"]),
    m_p_name        = "p MSE",
    m_p_on_batch    = lambda scope: float(torch.sum(
        (scope["output"][:, 2, :, :] - scope["batch"][1][:, 2, :, :]) ** 2)),
    m_p_on_epoch    = lambda scope: sum(scope["list"]) / len(scope["dataset"]),
    patience        = args.patience,
    after_epoch     = after_epoch
)

# ══════════════════════════════════════════════════════════════════
# 8. SAVE MODEL
# ══════════════════════════════════════════════════════════════════

state_dict = DeepCFD.state_dict()
state_dict["input_shape"]  = (1, 3, nx, ny)
state_dict["filters"]      = FILTERS
state_dict["kernel_size"]  = KERNEL_SIZE
state_dict["architecture"] = "UNetEx"
torch.save(state_dict, args.output)
print(f"\n✅ Best fine-tuned model saved → {args.output}")

# ══════════════════════════════════════════════════════════════════
# 9. PLOT TRAINING CURVES
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_loss_curve, label="Train Loss", alpha=0.7)
axes[0].plot(test_loss_curve,  label="Val Loss",   alpha=0.7)
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(train_mse_curve, label="Train MSE", alpha=0.7)
axes[1].plot(test_mse_curve,  label="Val MSE",   alpha=0.7)
axes[1].set_title("MSE Curves")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
axes[1].legend(); axes[1].grid(True)

axes[2].plot(lr_curve, label="Learning Rate", color="orange")
axes[2].set_title("Learning Rate Schedule")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
axes[2].set_yscale("log"); axes[2].legend(); axes[2].grid(True)

plt.suptitle("SwiftCFD — Fine-Tuning Training Curves", fontsize=14)
plt.tight_layout()
plt.savefig(args.save_plot, dpi=150)
plt.show()
print(f"✅ Training curves saved → {args.save_plot}")