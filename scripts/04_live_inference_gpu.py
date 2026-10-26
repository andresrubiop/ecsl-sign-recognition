"""Etapa 4: ejecuta el ensemble en vivo usando obligatoriamente CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from pfrutas.live import run_live
from pfrutas.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/lstm_compact_v2_gpu"),
        help=(
            "Ensemble v2 (BiLSTM o fusión ASL) que se cargará "
            "obligatoriamente en CUDA."
        ),
    )
    parser.add_argument("--fruit", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smoothing", type=float, default=0.30)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    resolve_device("cuda")
    run_live(
        run_dir=args.run_dir,
        fruit=args.fruit,
        camera=args.camera,
        device_request="cuda",
        confidence_threshold=args.threshold,
        smoothing=args.smoothing,
        mirror_display=not args.no_mirror,
    )


if __name__ == "__main__":
    main()
