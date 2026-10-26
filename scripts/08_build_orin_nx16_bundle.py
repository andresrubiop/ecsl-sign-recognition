"""Construye el bundle seguro de despliegue para Jetson Orin NX 16 GB."""

from __future__ import annotations

import argparse
from pathlib import Path

from pfrutas.jetson import DEFAULT_ORIN_BUNDLE_DIR, build_orin_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/fusion_asl_citizen_stgcn_v1_gpu"),
        help="Corrida completa que contiene ensemble.json y sus seis modelos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ORIN_BUNDLE_DIR,
        help="Directorio nuevo donde se escribira el bundle.",
    )
    parser.add_argument(
        "--parity-sample",
        type=Path,
        default=None,
        help=(
            "NPZ de data/processed_asl_transfer_v1 usado para incluir una "
            "prediccion golden reproducible."
        ),
    )
    parser.add_argument(
        "--parity-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Backend usado una sola vez para calcular la prediccion golden.",
    )
    parser.add_argument(
        "--skip-mediapipe-fingerprint",
        action="store_true",
        help=(
            "No registrar hashes de los activos MediaPipe del host constructor; "
            "el contrato de versiones se conserva."
        ),
    )
    args = parser.parse_args()

    manifest = build_orin_bundle(
        run_dir=args.run_dir,
        bundle_dir=args.output_dir,
        parity_sample=args.parity_sample,
        parity_device=args.parity_device,
        include_mediapipe_fingerprint=not args.skip_mediapipe_fingerprint,
    )
    total_bytes = sum(int(asset["size_bytes"]) for asset in manifest["assets"])
    checkpoints = [
        asset for asset in manifest["assets"] if asset.get("kind") == "checkpoint"
    ]
    print(f"Bundle Orin creado: {args.output_dir.resolve()}")
    print(f"Checkpoints seguros: {len(checkpoints)}")
    print(f"Tamano total registrado: {total_bytes / 1024**2:.2f} MiB")
    print("Carga de checkpoints: torch.load(weights_only=True) verificada")
    if manifest.get("parity"):
        print("Muestra de paridad y probabilidades golden: incluidas")


if __name__ == "__main__":
    main()
