"""Inferencia en vivo optimizada para Jetson Orin NX de 16 GB."""

from __future__ import annotations

import argparse
from pathlib import Path

from pfrutas.live import build_jetson_csi_pipeline, run_live
from pfrutas.utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/orin_nx16_fusion"),
        help="Bundle portable creado para Orin NX.",
    )
    parser.add_argument("--fruit", default=None)
    parser.add_argument(
        "--profile",
        choices=("full", "final"),
        default="full",
        help=(
            "full conserva los seis miembros; final reduce latencia pero no "
            "tiene métricas OOF propias."
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de cámara USB; se ignora con --csi.",
    )
    parser.add_argument(
        "--csi",
        action="store_true",
        help="Usar una cámara CSI mediante nvarguscamerasrc/GStreamer.",
    )
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--flip-method",
        type=int,
        choices=tuple(range(8)),
        default=0,
        help="Transformación de nvvidconv para una cámara CSI.",
    )
    parser.add_argument(
        "--prediction-fps",
        type=float,
        default=5.0,
        help="Frecuencia máxima del clasificador; los landmarks siguen a 30 FPS.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smoothing", type=float, default=0.30)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--allow-unsafe-checkpoint-loading",
        action="store_true",
        help=(
            "Permitir checkpoints históricos con pickle completo. El bundle "
            "Orin portable debe funcionar sin esta opción."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Este launcher nunca degrada silenciosamente a CPU.
    resolve_device("cuda")
    if not args.allow_unsafe_checkpoint_loading:
        from pfrutas.jetson import validate_bundle

        # Verifica esquema, límites de ruta, tamaño y SHA-256 de cada activo.
        # La deserialización segura posterior vuelve innecesario reconstruir
        # aquí los seis modelos en CPU.
        validate_bundle(args.run_dir, verify_checkpoints=False)
    pipeline = (
        build_jetson_csi_pipeline(
            sensor_id=args.sensor_id,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            flip_method=args.flip_method,
        )
        if args.csi
        else None
    )
    run_live(
        run_dir=args.run_dir,
        fruit=args.fruit,
        camera=args.camera,
        device_request="cuda",
        confidence_threshold=args.threshold,
        smoothing=args.smoothing,
        mirror_display=not args.no_mirror,
        prediction_fps=args.prediction_fps,
        camera_pipeline=pipeline,
        width=args.width,
        height=args.height,
        checkpoint_profile=args.profile,
        weights_only=not args.allow_unsafe_checkpoint_loading,
    )


if __name__ == "__main__":
    main()
