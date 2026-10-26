from __future__ import annotations

import argparse
from pathlib import Path

from .model import DEFAULT_MODEL_PRESET, MODEL_PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfrutas",
        description="MediaPipe + BiLSTM para reconocimiento de señas de frutas.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspeccionar la estructura de raw.")
    inspect_parser.add_argument("--raw-dir", type=Path, default=Path("raw"))

    prepare = subparsers.add_parser(
        "prepare",
        help="Extraer y normalizar landmarks desde los videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument("--raw-dir", type=Path, default=Path("raw"))
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_v2"),
        help="Directorio v2 para las secuencias y el manifiesto reproducible.",
    )
    prepare.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frecuencia temporal usada al extraer landmarks.",
    )
    prepare.add_argument("--max-side", type=int, default=960)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument(
        "--include-excluded",
        action="store_true",
        help=(
            "Procesar también videos marcados con _x; sin esta opción se "
            "registran en el manifiesto, pero no entran al experimento."
        ),
    )
    prepare.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Solo para pruebas rápidas del extractor.",
    )

    prepare_transfer = subparsers.add_parser(
        "prepare-transfer",
        help="Extraer v2 y el grafo pose/manos exacto para transferencia ASL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare_transfer.add_argument("--raw-dir", type=Path, default=Path("raw"))
    prepare_transfer.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_asl_transfer_v1"),
    )
    prepare_transfer.add_argument("--fps", type=float, default=30.0)
    prepare_transfer.add_argument("--max-side", type=int, default=960)
    prepare_transfer.add_argument("--overwrite", action="store_true")
    prepare_transfer.add_argument("--include-excluded", action="store_true")
    prepare_transfer.add_argument("--max-videos", type=int, default=None)

    train = subparsers.add_parser(
        "train",
        help="Entrenar con validación anidada agrupada por participante.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_v2"),
        help="Dataset v2 preparado a 30 FPS.",
    )
    train.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/lstm_compact_v2_gpu"),
        help="Directorio aislado para los artefactos reproducibles de la corrida CUDA v2.",
    )
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--sequence-length", type=int, default=48)
    train.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Épocas fijas para todas las validaciones internas.",
    )
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--samples-per-video", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=8e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cuda",
        help="Backend de entrenamiento; CUDA es obligatorio en el flujo recomendado.",
    )
    train.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Procesos que preparan batches para la GPU.",
    )
    train.add_argument(
        "--model-preset",
        choices=tuple(sorted(MODEL_PRESETS)),
        default=DEFAULT_MODEL_PRESET,
        help="Arquitectura versionada; compact_v2 es la recomendada para este dataset pequeño.",
    )
    train.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Pesos ST-GCN oficiales; obligatorio solo para el preset ASL transfer.",
    )
    train.add_argument("--only-fold", type=int, default=None)
    train.add_argument("--resume", action="store_true")

    live = subparsers.add_parser(
        "live",
        help="Probar el ensemble en vivo con la cámara.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    live.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/lstm_compact_v2_gpu"),
        help="Directorio v2 que contiene el ensemble ejecutado por CUDA.",
    )
    live.add_argument(
        "--fruit",
        type=str,
        default=None,
        help="Fruta objetivo. Sin este argumento se usa el modo general.",
    )
    live.add_argument("--camera", type=int, default=0)
    live.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cuda",
        help="Backend del clasificador en vivo.",
    )
    live.add_argument("--threshold", type=float, default=None)
    live.add_argument("--smoothing", type=float, default=0.30)
    live.add_argument("--no-mirror", action="store_true")
    live.add_argument(
        "--list-fruits",
        action="store_true",
        help="Mostrar las frutas disponibles y salir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        from .data import dataset_summary

        summary = dataset_summary(args.raw_dir)
        import json

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "prepare":
        from .data import prepare_dataset

        result = prepare_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            target_fps=args.fps,
            max_side=args.max_side,
            overwrite=args.overwrite,
            max_videos=args.max_videos,
            include_excluded=args.include_excluded,
        )
        print(
            f"Preparación terminada: {len(result['videos'])} videos, "
            f"{len(result['failures'])} fallos, {len(result['labels'])} clases."
        )
        if result["failures"]:
            for failure in result["failures"]:
                print(f"  ERROR {failure['source']}: {failure['error']}")
        return
    if args.command == "prepare-transfer":
        from .data import prepare_transfer_dataset

        result = prepare_transfer_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            target_fps=args.fps,
            max_side=args.max_side,
            overwrite=args.overwrite,
            max_videos=args.max_videos,
            include_excluded=args.include_excluded,
        )
        print(
            f"Preparación transfer terminada: {len(result['videos'])} videos, "
            f"{len(result['failures'])} fallos, {len(result['labels'])} clases."
        )
        return
    if args.command == "train":
        from .training import train_cross_validation

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
            device_request=args.device,
            num_workers=args.num_workers,
            model_preset=args.model_preset,
            pretrained_checkpoint=args.pretrained_checkpoint,
            only_fold=args.only_fold,
            resume=args.resume,
        )
        return
    if args.command == "live":
        from .live import run_live
        from .utils import read_json

        if args.list_fruits:
            ensemble = read_json(args.run_dir / "ensemble.json")
            print("\n".join(ensemble["labels"]))
            return
        run_live(
            run_dir=args.run_dir,
            fruit=args.fruit,
            camera=args.camera,
            device_request=args.device,
            confidence_threshold=args.threshold,
            smoothing=args.smoothing,
            mirror_display=not args.no_mirror,
        )
        return


if __name__ == "__main__":
    main()
