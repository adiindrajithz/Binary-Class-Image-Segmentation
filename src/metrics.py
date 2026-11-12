from __future__ import annotations

import torch
import torch.nn as nn

from src.config import PRED_THRESHOLD


def compute_batch_iou(
    preds_bin: torch.Tensor,
    masks_bool: torch.Tensor,
) -> torch.Tensor:
    """Per-sample IoU within a batch."""
    intersections = (preds_bin & masks_bool).flatten(1).sum(1).float()
    unions = (preds_bin | masks_bool).flatten(1).sum(1).float()
    return torch.where(
        unions > 0,
        intersections / unions.clamp_min(1),
        torch.ones_like(unions),
    )


def compute_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    target_class: int = 1,
) -> float:
    """Intersection-over-Union for target_class."""
    intersection = ((pred_mask == target_class) & (true_mask == target_class)).sum().item()
    union = ((pred_mask == target_class) | (true_mask == target_class)).sum().item()
    return 1.0 if union == 0 else intersection / union


def compute_metrics(
    tp: int,
    fp: int,
    fn: int,
    iou_sum: float,
    n_samples: int,
    correct_px: int,
    total_px: int,
    total_loss: float,
    n_batches: int,
) -> dict:
    avg_loss = total_loss / max(n_batches, 1)
    pixel_accuracy = (correct_px / max(total_px, 1)) * 100
    mean_iou = iou_sum / max(n_samples, 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(
        avg_loss=avg_loss,
        pixel_accuracy=pixel_accuracy,
        mean_iou=mean_iou,
        precision=precision,
        recall=recall,
        dice=dice,
    )


def evaluate_model(
    model,
    test_loader,
    test_dataset,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = PRED_THRESHOLD,
) -> dict:
    model.eval()

    total_loss = 0.0
    iou_sum = 0.0
    tp = fp = fn = 0
    correct_px = 0
    total_px = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks.unsqueeze(1).float())
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= threshold).squeeze(1).long()

            for p, t in zip(preds, masks):
                iou_sum += compute_iou(p, t, target_class=1)
                correct_px += (p == t).sum().item()
                total_px += t.numel()
                tp += ((p == 1) & (t == 1)).sum().item()
                fp += ((p == 1) & (t == 0)).sum().item()
                fn += ((p == 0) & (t == 1)).sum().item()

    return compute_metrics(
        tp,
        fp,
        fn,
        iou_sum,
        n_samples=len(test_dataset),
        correct_px=correct_px,
        total_px=total_px,
        total_loss=total_loss,
        n_batches=len(test_loader),
    )


def print_eval_report(metrics: dict, model_name: str = "Model") -> None:
    bar = "=" * 45
    print(bar)
    print(f"  {model_name.upper()} EVALUATION REPORT (Test Set)")
    print(bar)
    if metrics["avg_loss"] > 0:
        print(f"  Average Test Loss  : {metrics['avg_loss']:.4f}")
    print(f"  Pixel Accuracy     : {metrics['pixel_accuracy']:.2f}%")
    print(f"  Mean IoU (Person)  : {metrics['mean_iou']:.4f}")
    print(f"  Precision (Person) : {metrics['precision']:.4f}")
    print(f"  Recall    (Person) : {metrics['recall']:.4f}")
    print(f"  Dice / F1 (Person) : {metrics['dice']:.4f}")
    print(bar)
