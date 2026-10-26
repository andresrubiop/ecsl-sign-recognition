"""Etapa 2: reproduce la validación anidada LOSO y el refit final en CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from pfrutas.model import (
    ASL_TRANSFER_MODEL_PRESET,
    DEFAULT_MODEL_PRESET,
    MODEL_PRESETS,
)
from pfrutas.training import train_cross_validation
from pfrutas.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_v2"),
        help="Dataset v2 preparado a 30 FPS.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/lstm_compact_v2_gpu"),
        help="Directorio aislado de la corrida reproducible CUDA v2.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-video", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-preset",
        choices=tuple(
            sorted(
                preset
                for preset in MODEL_PRESETS
                if preset != ASL_TRANSFER_MODEL_PRESET
            )
        ),
        default=DEFAULT_MODEL_PRESET,
        help="Arquitectura versionada del clasificador; compact_v2 es la recomendada.",
    )
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    resolve_device("cuda")
    train_cross_validation(
        processed_dir=args.processed_dir,
        run_dir=args.run_dir,
        folds=args.folds,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_video=args.samples_per_video,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_request="cuda",
        num_workers=args.num_workers,
        model_preset=args.model_preset,
        only_fold=args.only_fold,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
