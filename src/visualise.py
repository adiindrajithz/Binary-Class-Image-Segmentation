from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PRED_THRESHOLD,
    VIZ_NUM_SAMPLES,
)


def denormalize_image(
    img_tensor: torch.Tensor,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> np.ndarray:
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img_vis = img_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img_vis * std_arr + mean_arr, 0.0, 1.0)


def visualise_predictions(
    model,
    test_dataset,
    device: torch.device,
    model_name: str = "Model",
    num_samples: int = VIZ_NUM_SAMPLES,
    threshold: float = PRED_THRESHOLD,
) -> None:
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 2.5))

    with torch.no_grad():
        for i in range(num_samples):
            img_tensor, mask_tensor = test_dataset[i]
            output = model(img_tensor.unsqueeze(0).to(device))["out"]
            pred_mask = (torch.sigmoid(output) >= threshold).squeeze(0).squeeze(0).cpu().numpy()

            img_vis = denormalize_image(img_tensor)

            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title(f"Test Image #{i + 1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_tensor.cpu().numpy(), cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_mask, cmap="gray")
            axes[i, 2].set_title(f"Predicted Mask ({model_name})")
            axes[i, 2].axis("off")

    plt.suptitle(
        f"Inference: Input Image vs. Ground Truth vs. {model_name} Prediction",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_benchmark_comparison(metrics_dict: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in metrics_dict.items():
        rows.append(
            {
                "Model": name,
                "Pixel Accuracy (%)": round(m.get("pixel_accuracy", 0.0), 2),
                "Mean IoU": round(m.get("mean_iou", 0.0), 4),
                "Dice / F1": round(m.get("dice", 0.0), 4),
                "Precision": round(m.get("precision", 0.0), 4),
                "Recall": round(m.get("recall", 0.0), 4),
            }
        )

    comparison_df = pd.DataFrame(rows)

    bar = "=" * 80
    print(bar)
    print("                    MODEL BENCHMARK COMPARISON TABLE")
    print(bar)
    print(comparison_df.to_string(index=False))
    print(bar)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(comparison_df))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2, comparison_df["Mean IoU"], width, label="Mean IoU", color="tab:blue"
    )
    rects2 = ax.bar(
        x + width / 2, comparison_df["Dice / F1"], width, label="Dice / F1", color="tab:orange"
    )

    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
    ax.set_title("Model Comparison: Mean IoU vs. Dice/F1 Score Across All Models", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for rect in (*rects1, *rects2):
        h = rect.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()

    return comparison_df
