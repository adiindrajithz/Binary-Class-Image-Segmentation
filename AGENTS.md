## Project Overview

Binary segmentation of people from background using four pretrained deep-learning models.

- **Input** : RGB image (3 × H × W)
- **Output**: Binary mask (1 × H × W) — 1 = person, 0 = background
- **Loss** : BCEWithLogitsLoss with class-balanced `pos_weight`
- **Data split**: 70 / 15 / 15 (train / val / test), `random_state=42`

---

## Project Structure

```
Binary Image Segmentation/
├── src/
│   ├── config.py       # All hyperparameters and paths — edit here first
│   ├── dataset.py      # BinarySegmentationDataset, build_dataframes, build_dataloaders
│   ├── models.py       # 4 model classes + builder functions
│   ├── trainer.py      # train_model(), plot_training_curves()
│   ├── metrics.py      # evaluate_model(), compute_metrics(), print_eval_report()
│   └── visualise.py    # visualise_predictions(), plot_benchmark_comparison()
├── cli.py              # argparse CLI — entry point for all tasks
├── pyproject.toml      # Ruff linter/formatter configuration
├── requirements.txt    # Runtime dependencies
└── grayscaleImages/
    ├── images/         # Input JPEGs
    └── masks/          # Binary PNGs (0 = background, 255 = person)
```

---

## Models (`src/models.py`)

All models follow a **uniform interface** — `forward()` returns `{"out": logits}`.
Checkpoints are saved as full model objects via `torch.save(model, path)`.

| Class | Architecture | Backbone | Checkpoint |
|---|---|---|---|
| `MobileNetV3UNet` | LR-ASPP | MobileNetV3-Large | `checkpoints/MobileNetV3UNet.pth` |
| `FPN` | FPN (via smp) | ResNet34 | `checkpoints/fpn_model.pth` |
| `SAM2SegmentationModel` | DeepLabV3 | MobileNetV3-Large | `checkpoints/sam2_model.pth` |
| `FCNBinary` | FCN | ResNet50 | `checkpoints/fcn_model.pth` |

Each model has a corresponding builder function: `build_mobileunet()`, `build_fpn()`, `build_sam2()`, `build_fcn()`.

---

## Configuration (`src/config.py`)

**Always edit `config.py` to change hyperparameters** — never hardcode values in other files.

Key constants: `IMG_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `EARLY_STOP_PATIENCE`, `PRED_THRESHOLD`, `*_EPOCHS`, `*_CKPT`.

---

## CLI (`cli.py`)

The primary entry point. All training, evaluation, and inference goes through here.

```bash
python cli.py train    --model <unet|fpn|sam2|fcn|all> [--epochs N] [--lr F] [--plot-curves]
python cli.py evaluate --model <unet|fpn|sam2|fcn> [--checkpoint PATH] [--threshold F]
python cli.py predict  --model <unet|fpn|sam2|fcn> [--num-samples N] [--checkpoint PATH]
python cli.py benchmark
python cli.py data-info
python cli.py list-models
```

**Adding a new model to the CLI**: register it in `MODEL_REGISTRY` at the top of `cli.py` and add its class to `safe_classes` inside `_load_checkpoint()`.

---

## Code Style & Linting

This project uses **Ruff** for linting and formatting (configured in `pyproject.toml`).

**Run before every commit:**
```bash
ruff check src cli.py --fix
ruff format src cli.py
```

**Rules enforced**: `E`, `W`, `F`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `C4`, `SIM`, `TID`.

**Key conventions:**
- Line length: **100 characters**
- Quotes: **double** (`"`)
- Python target: **3.12** — use modern syntax (`X | None` not `Optional[X]`, `dict[str, int]` not `Dict`)
- Import order: stdlib → third-party (torch/numpy/...) → `src.*`
- No `dict()` calls — use `{}` literals
- Use `contextlib.suppress()` instead of bare `try/except/pass`

---

## Checkpoint Loading

Checkpoints are **full pickled model objects** (not state-dicts). PyTorch ≥ 2.6 requires explicit allowlisting:

```python
# Always load via the _load_checkpoint() helper in cli.py
# Never call torch.load() directly on these checkpoints elsewhere
model = _load_checkpoint(ckpt_path, device)
```

If a new model class is added, add it to `safe_classes` in `_load_checkpoint()`.

---

## Key Conventions

- **All models return `{"out": logits}`** — never change this interface.
- **Never import from `src.config` inside `src/` modules at function scope** — keep imports at the top of each file.
- **`src/config.py` is the single source of truth** for all paths and hyperparameters.
- **Do not add `scipy`** — it is not used in this project (old leftover install cell in the notebook only).
- Dataset paths are resolved relative to the project root via `Path(__file__).resolve().parent.parent` — do not use hardcoded absolute paths.

---

## Adding a New Model

1. Add the model class and builder function to `src/models.py` — keep the `{"out": logits}` return format.
2. Add checkpoint path constant to `src/config.py`.
3. Register the model in `MODEL_REGISTRY` in `cli.py`.
4. Add the class to `safe_classes` in `_load_checkpoint()` in `cli.py`.
5. Run `ruff check src cli.py --fix && ruff format src cli.py`.
