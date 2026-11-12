# Binary Image Segmentation – Modular Python Project

Binary segmentation of people from background using four deep-learning architectures:
**MobileNetV3-UNet · FPN · SAM 2 · FCN (ResNet50)**

---

## Project Structure

```
Binary Image Segmentation/
│
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                 # Centralised hyper-parameters & paths
│   ├── dataset.py                # Dataset classes & DataLoader factories
│   ├── models.py                 # All model architectures
│   ├── trainer.py                # Training loops, early stopping, plots
│   ├── metrics.py                # IoU, precision/recall/dice, eval loops
│   └── visualise.py              # Inference plots, benchmark bar chart
│
├── data/                         # (placeholder – datasets stay in grayscaleImages/)
│
├── grayscaleImages/              # IMAGE DATASET  (unchanged)
│   ├── images/                   # Input images  (*.jpg)
│   ├── masks/                    # Binary masks  (*.png, 0 = background, 255 = person)
│   └── human_pixel_report.csv
│
├── JPEGImages/                   # Original colour images  (unchanged)
├── SegmentationClass/            # Original colour masks   (unchanged)
├── Visualizations/               # Visualisation outputs   (unchanged)
│
├── checkpoints/                  # Saved model weights (auto-created)
│   ├── MobileNetV3UNet.pth
│   ├── fpn_model.pth
│   ├── sam2_model.pth
│   └── fcn_model.pth
│
├── outputs/                      # Any extra outputs (auto-created)
│
├── imageSegmentation_refactored.ipynb  # ← USE THIS NOTEBOOK
├── imageSegmentation.ipynb             # Original notebook (kept for reference)
├── requirements.txt
└── README.md
```

---

## Modules

### `src/config.py`
Single source of truth for every configurable value.

| Constant | Value | Description |
|---|---|---|
| `IMAGE_DIR` | `grayscaleImages/images` | Input image directory |
| `MASK_DIR` | `grayscaleImages/masks` | Binary mask directory |
| `IMG_SIZE` | `(256, 256)` | Resize target for all models |
| `BATCH_SIZE` | 16 | Training batch size |
| `VAL_BATCH` | 32 | Validation / testing batch size |
| `UNET_EPOCHS` | 150 | Max epochs for MobileNetV3-UNet |
| `FPN_EPOCHS` | 50 | Max epochs for FPN |
| `SAM2_EPOCHS` | 150 | Max epochs for SAM 2 |
| `FCN_EPOCHS` | 150 | Max epochs for FCN |
| `EARLY_STOP_PATIENCE` | 5 | Early stopping patience |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `POS_WEIGHT_MAX` | 5.0 | Upper bound for BCE pos_weight |
| `PRED_THRESHOLD` | 0.5 | Sigmoid threshold |

### `src/dataset.py`
- `build_dataframes()` – scans image/mask directories, returns train/val/test DataFrames (70/15/15 split)
- `BinarySegmentationDataset` – PIL→Tensor, ImageNet normalisation, threshold-based mask binarisation
- `build_dataloaders()` – wraps standard datasets in DataLoaders

### `src/models.py`
| Factory | Architecture | Backbone |
|---|---|---|
| `build_mobileunet()` | Custom UNet | MobileNetV3-Large |
| `build_fpn()` | FPN (SMP or native fallback) | ResNet34 |
| `build_sam2()` | Two-way attention + decoder | Custom hierarchical encoder |
| `build_fcn()` | FCN | ResNet50 |

All models return `{"out": logits}` for a uniform interface.

### `src/trainer.py`
- `compute_pos_weight()` – class-balanced BCE weight from training data
- `train_model()` – generic training loop with early stopping (UNet / FPN / SAM 2 / FCN)
- `plot_training_curves()` – loss & IoU plots from history dict

### `src/metrics.py`
- `compute_batch_iou()` – vectorised per-batch IoU (used in training)
- `compute_iou()` – per-sample scalar IoU (used in evaluation)
- `evaluate_model()` – full test-set loop for standard models
- `print_eval_report()` – formatted console output

### `src/visualise.py`
- `denormalize_image()` – reverse ImageNet normalisation
- `visualise_predictions()` – 3-column plot (input / GT / prediction)
- `plot_benchmark_comparison()` – bar chart + comparison table

---

## Quick Start (Colab)

1. **Upload** the entire project folder to your Google Drive.
2. Open `imageSegmentation_refactored.ipynb` in Google Colab.
3. Update `PROJECT_ROOT` in **Cell 1** to match your Drive path.
4. Run all cells from top to bottom.

> [!IMPORTANT]
> The notebook mounts Google Drive and adds the project root to `sys.path` so that `from src.xxx import ...` works correctly in Colab.

---

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook locally with Jupyter
jupyter notebook imageSegmentation_refactored.ipynb
```

> [!NOTE]
> When running locally, comment out the Google Drive mount cells (Cells 0–1). The `config.py` uses `Path(__file__).resolve().parent.parent` to auto-detect the project root.

---

## Data

| Folder | Contents |
|---|---|
| `grayscaleImages/images/` | Grayscale JPEG images |
| `grayscaleImages/masks/` | Binary PNG masks (0 = background, 255 = person) |

Images and masks are **not** moved; they stay in their original locations. `config.py` resolves absolute paths relative to the project root.

---

## Model Checkpoints

Trained weights are saved to `checkpoints/` automatically during training.

| File | Model |
|---|---|
| `MobileNetV3UNet.pth` | Full model object (`torch.save(model, ...)`) |
| `fpn_model.pth` | Full model object |
| `sam2_model.pth` | Full model object |
| `fcn_model.pth` | State dict (`torch.save(model.state_dict(), ...)`) |

---

## Preserved from Original Notebook

| Aspect | Details |
|---|---|
| Model architectures | Identical (same layers, channels, skip connections) |
| Hyper-parameters | lr=1e-4, patience=5, pos_weight_max=5.0, threshold=0.5 |
| Preprocessing | ImageNet normalisation, 256×256 resize, NEAREST mask interpolation |
| Data split | 70/15/15, random_state=42 |
| Loss function | BCEWithLogitsLoss with class-balanced pos_weight |
| Optimiser | Adam |

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
Pillow>=10.0.0
segmentation-models-pytorch>=0.3.3  # optional (FPN fallback available)
```
