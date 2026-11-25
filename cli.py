"""
Binary Image Segmentation – CLI
================================
Usage examples:

  # Train a single model
  python cli.py train --model unet --epochs 100 --lr 1e-4 --batch-size 16

  # Train all models sequentially
  python cli.py train --model all

  # Evaluate a saved checkpoint
  python cli.py evaluate --model fpn --checkpoint checkpoints/fpn_model.pth

  # Run inference / visualise predictions
  python cli.py predict --model unet --checkpoint checkpoints/MobileNetV3UNet.pth --num-samples 6

  # Benchmark all models and print comparison table
  python cli.py benchmark

  # Show dataset split statistics
  python cli.py data-info

  # List available models
  python cli.py list-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "unet": {
        "display": "MobileNetV3-UNet",
        "builder": "build_mobileunet",
        "ckpt": "UNET_CKPT",
        "epochs_key": "UNET_EPOCHS",
    },
    "fpn": {
        "display": "FPN (ResNet34)",
        "builder": "build_fpn",
        "ckpt": "FPN_CKPT",
        "epochs_key": "FPN_EPOCHS",
    },
    "sam2": {
        "display": "SAM2 Segmentation",
        "builder": "build_sam2",
        "ckpt": "SAM2_CKPT",
        "epochs_key": "SAM2_EPOCHS",
    },
    "fcn": {
        "display": "FCN (ResNet50)",
        "builder": "build_fcn",
        "ckpt": "FCN_CKPT",
        "epochs_key": "FCN_EPOCHS",
    },
}


def _resolve_device(device_str: str):
    import torch

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_checkpoint(ckpt_path, device):

    import torch

    from src import models as _models

    safe_classes = [
        _models.MobileNetV3UNet,
        _models.FPN,
        _models.SAM2SegmentationModel,
        _models.FCNBinary,
    ]
    import contextlib

    with contextlib.suppress(AttributeError):
        torch.serialization.add_safe_globals(safe_classes)

    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _load_data(image_dir: str | None, mask_dir: str | None, seed: int, img_size: tuple[int, int]):
    """Build dataframes and dataloaders, optionally overriding config paths."""
    from pathlib import Path as _Path

    from src import config
    from src.dataset import build_dataframes, build_dataloaders

    image_dir_p = _Path(image_dir) if image_dir else config.IMAGE_DIR
    mask_dir_p = _Path(mask_dir) if mask_dir else config.MASK_DIR

    df, train_df, val_df, test_df = build_dataframes(
        image_dir=image_dir_p,
        mask_dir=mask_dir_p,
        random_seed=seed,
    )
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        img_size=img_size,
    )
    return df, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def _build_model(model_key: str, device):
    from src import models

    builder_name = MODEL_REGISTRY[model_key]["builder"]
    builder = getattr(models, builder_name)
    return builder(device)


def _get_ckpt_path(model_key: str, checkpoint: str | None) -> Path:
    from src import config

    if checkpoint:
        return Path(checkpoint)
    attr = MODEL_REGISTRY[model_key]["ckpt"]
    return getattr(config, attr)


def _get_default_epochs(model_key: str, epochs_override: int | None) -> int:
    from src import config

    if epochs_override is not None:
        return epochs_override
    attr = MODEL_REGISTRY[model_key]["epochs_key"]
    return getattr(config, attr)


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def cmd_list_models(args):
    """List all available model keys."""
    print("\nAvailable models:")
    print(f"  {'key':<8}  {'display name'}")
    print("  " + "-" * 36)
    for key, info in MODEL_REGISTRY.items():
        print(f"  {key:<8}  {info['display']}")
    print()


def cmd_data_info(args):
    """Print dataset split statistics."""
    img_size = tuple(args.img_size)
    df, train_ds, val_ds, test_ds, *_ = _load_data(
        args.image_dir, args.mask_dir, args.seed, img_size
    )

    print("\n" + "=" * 45)
    print("  DATASET STATISTICS")
    print("=" * 45)
    print(f"  Total image-mask pairs : {len(df)}")
    print(f"  Train set size         : {len(train_ds)}")
    print(f"  Validation set size    : {len(val_ds)}")
    print(f"  Test set size          : {len(test_ds)}")
    print(f"  Image size (H x W)     : {img_size[0]} x {img_size[1]}")
    print(f"  Random seed            : {args.seed}")
    print("=" * 45 + "\n")


def cmd_train(args):
    """Train one or all models."""
    from src import config
    from src.trainer import plot_training_curves, train_model

    device = _resolve_device(args.device)
    img_size = tuple(args.img_size)
    model_key = args.model.lower()

    keys_to_train = list(MODEL_REGISTRY.keys()) if model_key == "all" else [model_key]

    if model_key != "all" and model_key not in MODEL_REGISTRY:
        print(
            f"[ERROR] Unknown model '{model_key}'. Run 'python cli.py list-models' to see options."
        )
        sys.exit(1)

    # Load data once
    print("Loading dataset …")
    _, train_ds, val_ds, _, train_loader, val_loader, _ = _load_data(
        args.image_dir, args.mask_dir, args.seed, img_size
    )

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for key in keys_to_train:
        info = MODEL_REGISTRY[key]
        n_epochs = _get_default_epochs(key, args.epochs)
        ckpt = _get_ckpt_path(key, args.checkpoint if model_key != "all" else None)
        lr = args.lr
        patience = args.patience

        print(f"\n{'=' * 50}")
        print(f"  Training: {info['display']}")
        print(f"  Epochs  : {n_epochs} | LR: {lr} | Device: {device}")
        print(f"  Save to : {ckpt}")
        print(f"{'=' * 50}\n")

        model = _build_model(key, device)

        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_ds,
            device=device,
            num_epochs=n_epochs,
            model_name=info["display"],
            lr=lr,
            patience=patience,
            save_path=str(ckpt),
        )

        if args.plot_curves:
            plot_training_curves(history, model_name=info["display"])


def cmd_evaluate(args):
    """Evaluate a saved model checkpoint on the test set."""
    import torch.nn as nn

    from src.metrics import evaluate_model, print_eval_report
    from src.trainer import compute_pos_weight

    model_key = args.model.lower()
    if model_key not in MODEL_REGISTRY:
        print(f"[ERROR] Unknown model '{model_key}'.")
        sys.exit(1)

    device = _resolve_device(args.device)
    img_size = tuple(args.img_size)

    print("Loading dataset …")
    _, train_ds, _, test_ds, _, _, test_loader = _load_data(
        args.image_dir, args.mask_dir, args.seed, img_size
    )

    ckpt = _get_ckpt_path(model_key, args.checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt}")
    model = _load_checkpoint(ckpt, device)
    model.to(device)

    pos_weight = compute_pos_weight(train_ds, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Evaluating {MODEL_REGISTRY[model_key]['display']} …\n")
    metrics = evaluate_model(
        model, test_loader, test_ds, criterion, device, threshold=args.threshold
    )
    print_eval_report(metrics, model_name=MODEL_REGISTRY[model_key]["display"])


def cmd_predict(args):
    """Run visual inference on test samples."""
    from src.visualise import visualise_predictions

    model_key = args.model.lower()
    if model_key not in MODEL_REGISTRY:
        print(f"[ERROR] Unknown model '{model_key}'.")
        sys.exit(1)

    device = _resolve_device(args.device)
    img_size = tuple(args.img_size)

    print("Loading dataset …")
    _, _, _, test_ds, _, _, _ = _load_data(args.image_dir, args.mask_dir, args.seed, img_size)

    ckpt = _get_ckpt_path(model_key, args.checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt}")
    model = _load_checkpoint(ckpt, device)
    model.to(device)

    visualise_predictions(
        model=model,
        test_dataset=test_ds,
        device=device,
        model_name=MODEL_REGISTRY[model_key]["display"],
        num_samples=args.num_samples,
        threshold=args.threshold,
    )


def cmd_benchmark(args):
    """Evaluate all models with saved checkpoints and print a comparison table."""
    import torch.nn as nn

    from src.metrics import evaluate_model
    from src.trainer import compute_pos_weight
    from src.visualise import plot_benchmark_comparison

    device = _resolve_device(args.device)
    img_size = tuple(args.img_size)

    print("Loading dataset …")
    _, train_ds, _, test_ds, _, _, test_loader = _load_data(
        args.image_dir, args.mask_dir, args.seed, img_size
    )

    pos_weight = compute_pos_weight(train_ds, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics_dict: dict[str, dict] = {}

    for key, info in MODEL_REGISTRY.items():
        ckpt = _get_ckpt_path(key, None)
        if not ckpt.exists():
            print(f"  [SKIP] {info['display']} – checkpoint not found ({ckpt})")
            continue

        print(f"  Evaluating {info['display']} …")
        model = _load_checkpoint(ckpt, device)
        model.to(device)
        m = evaluate_model(model, test_loader, test_ds, criterion, device, threshold=args.threshold)
        metrics_dict[info["display"]] = m

    if not metrics_dict:
        print("[ERROR] No checkpoints found. Train models first with 'python cli.py train'.")
        sys.exit(1)

    plot_benchmark_comparison(metrics_dict)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_data_args(parser: argparse.ArgumentParser):
    """Shared dataset-related arguments."""
    parser.add_argument(
        "--image-dir",
        default=None,
        metavar="PATH",
        help="Override path to the images directory (default: config.IMAGE_DIR).",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        metavar="PATH",
        help="Override path to the masks directory (default: config.MASK_DIR).",
    )
    parser.add_argument(
        "--img-size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("H", "W"),
        help="Image resize dimensions (default: 256 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test split (default: 42).",
    )


def _add_device_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use: 'auto' picks CUDA if available, else CPU (default: auto).",
    )


def _add_threshold_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary prediction threshold (default: 0.5).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Binary Image Segmentation – command-line interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # ------------------------------------------------------------------
    # list-models
    # ------------------------------------------------------------------
    sub.add_parser(
        "list-models",
        help="List all available model architectures.",
    )

    # ------------------------------------------------------------------
    # data-info
    # ------------------------------------------------------------------
    p_data = sub.add_parser(
        "data-info",
        help="Print dataset split statistics.",
    )
    _add_common_data_args(p_data)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    p_train = sub.add_parser(
        "train",
        help="Train a segmentation model (or all models).",
    )
    p_train.add_argument(
        "--model",
        "-m",
        required=True,
        choices=[*MODEL_REGISTRY.keys(), "all"],
        help="Model architecture to train, or 'all' to train every model sequentially.",
    )
    p_train.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Number of training epochs. Defaults to the value in config.py for each model.",
    )
    p_train.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: config.BATCH_SIZE).",
    )
    p_train.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience in epochs (default: 5).",
    )
    p_train.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to save the trained checkpoint. Defaults to config path for the chosen model.",
    )
    p_train.add_argument(
        "--plot-curves",
        action="store_true",
        help="Show loss/IoU curves after training.",
    )
    _add_common_data_args(p_train)
    _add_device_arg(p_train)

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate a saved checkpoint on the test set.",
    )
    p_eval.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture.",
    )
    p_eval.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to the checkpoint file. Defaults to config path for the chosen model.",
    )
    _add_common_data_args(p_eval)
    _add_device_arg(p_eval)
    _add_threshold_arg(p_eval)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    p_pred = sub.add_parser(
        "predict",
        help="Run inference and visualise predictions on test samples.",
    )
    p_pred.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture.",
    )
    p_pred.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to the checkpoint file. Defaults to config path for the chosen model.",
    )
    p_pred.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of test samples to visualise (default: 4).",
    )
    _add_common_data_args(p_pred)
    _add_device_arg(p_pred)
    _add_threshold_arg(p_pred)

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------
    p_bench = sub.add_parser(
        "benchmark",
        help="Evaluate all saved checkpoints and print a comparison table.",
    )
    _add_common_data_args(p_bench)
    _add_device_arg(p_bench)
    _add_threshold_arg(p_bench)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "list-models": cmd_list_models,
    "data-info": cmd_data_info,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "predict": cmd_predict,
    "benchmark": cmd_benchmark,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    fn = COMMAND_MAP.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
