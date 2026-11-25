from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.config import (
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    POS_WEIGHT_MAX,
    PRED_THRESHOLD,
)
from src.metrics import compute_batch_iou


def compute_pos_weight(
    train_dataset: Dataset,
    device: torch.device,
    max_weight: float = POS_WEIGHT_MAX,
) -> torch.Tensor:
    pixel_counts = torch.zeros(2)
    for _, mask in train_dataset:
        pixel_counts += torch.bincount(mask.reshape(-1), minlength=2).float()
    pos_weight = (pixel_counts[0] / pixel_counts[1].clamp_min(1)).clamp(max=max_weight)
    return pos_weight.to(device)


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_dataset: Dataset,
    device: torch.device,
    num_epochs: int,
    model_name: str = "Model",
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
    threshold: float = PRED_THRESHOLD,
    save_path: str | None = None,
) -> tuple[nn.Module, dict]:
    pos_weight = compute_pos_weight(train_dataset, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: dict | None = None

    history = {"train_losses": [], "val_losses": [], "train_ious": [], "val_ious": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss = run_iou = 0.0
        n_train = 0

        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(outputs).squeeze(1) >= threshold
                run_iou += compute_batch_iou(preds, masks.bool()).sum().item()
                n_train += masks.size(0)

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch [{epoch}/{num_epochs}] | "
                    f"Batch [{batch_idx:02d}/{len(train_loader)}] | "
                    f"Batch Loss: {loss.item():.4f}"
                )

        avg_train_loss = run_loss / len(train_loader)
        avg_train_iou = run_iou / n_train
        history["train_losses"].append(avg_train_loss)
        history["train_ious"].append(avg_train_iou)

        model.eval()
        val_loss = val_iou = 0.0
        n_val = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)["out"]
                loss = criterion(outputs, masks.unsqueeze(1).float())
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).squeeze(1) >= threshold
                val_iou += compute_batch_iou(preds, masks.bool()).sum().item()
                n_val += masks.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / n_val
        history["val_losses"].append(avg_val_loss)
        history["val_ious"].append(avg_val_iou)

        print(
            f"---> Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}\n"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Best {model_name} weights restored.")

    if save_path:
        torch.save(model, save_path)
        print(f"{model_name} saved to {save_path}")

    return model, history


def plot_training_curves(history: dict, model_name: str = "Model") -> None:
    has_iou = "train_ious" in history and len(history["train_ious"]) > 0
    n_plots = 2 if has_iou else 1
    epochs_r = range(1, len(history["train_losses"]) + 1)

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(
        epochs_r, history["train_losses"], marker="o", label="Train Loss", color="royalblue", lw=2
    )
    axes[0].plot(
        epochs_r, history["val_losses"], marker="s", label="Val Loss", color="crimson", lw=2
    )
    axes[0].set(title=f"{model_name}: Training vs. Validation Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_xticks(list(epochs_r))

    if has_iou:
        axes[1].plot(
            epochs_r, history["train_ious"], marker="o", label="Train IoU", color="seagreen", lw=2
        )
        axes[1].plot(
            epochs_r, history["val_ious"], marker="s", label="Val IoU", color="darkorange", lw=2
        )
        axes[1].set(
            title=f"{model_name}: Training vs. Validation IoU", xlabel="Epoch", ylabel="Mean IoU"
        )
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].set_xticks(list(epochs_r))

    plt.tight_layout()
    plt.show()
