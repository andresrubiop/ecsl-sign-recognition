"""Prepara el cache multimodal v2 + pose/manos para transferencia ASL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pfrutas.data import dataset_summary, prepare_transfer_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_asl_transfer_v1"),
        help="Directorio para manifest.json y secuencias multimodales.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frecuencia temporal reproducible para extraer landmarks.",
    )
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Incluye videos _x; por defecto quedan fuera y se auditan en el manifiesto.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            dataset_summary(
                args.raw_dir,
                include_excluded=args.include_excluded,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
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
    if result["failures"]:
        raise RuntimeError(
            "La preparación terminó con videos fallidos; revisa manifest.json."
        )


if __name__ == "__main__":
    main()
