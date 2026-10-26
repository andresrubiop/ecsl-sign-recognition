"""Mide latencia, FPS y memoria del bundle Orin NX 16 GB."""

from __future__ import annotations

import argparse
from pathlib import Path

from pfrutas.jetson import DEFAULT_ORIN_BUNDLE_DIR, benchmark_bundle
from pfrutas.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_ORIN_BUNDLE_DIR,
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="CUDA es el backend de despliegue; CPU solo permite smoke tests.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "folds", "final"),
        default="full",
        help=(
            "full usa los seis miembros; final reduce latencia pero no hereda "
            "la accuracy OOF del ensemble."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/orin_nx16_benchmark.json"),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Numero de hilos CPU de PyTorch; no se modifica si se omite.",
    )
    args = parser.parse_args()
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            parser.error("--torch-threads debe ser mayor que cero.")
        import torch

        torch.set_num_threads(args.torch_threads)

    result = benchmark_bundle(
        bundle_dir=args.bundle_dir,
        device_request=args.device,
        mode=args.mode,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    write_json(args.output, result)
    latency = result["latency"]
    memory = result["memory"]
    print(
        f"{result['mode']} ({result['member_count']} modelos): "
        f"p50={latency['p50_ms']:.2f} ms | "
        f"p95={latency['p95_ms']:.2f} ms | "
        f"FPS={latency['throughput_fps']:.2f}"
    )
    if result["device"] == "cuda":
        print(
            "Memoria CUDA pico: "
            f"{memory['peak_allocated_bytes'] / 1024**2:.2f} MiB asignados | "
            f"{memory['peak_reserved_bytes'] / 1024**2:.2f} MiB reservados"
        )
    if result.get("parity") is not None:
        parity = result["parity"]
        print(
            "Paridad golden: "
            f"argmax={'OK' if parity['argmax_matches'] else 'ERROR'} | "
            f"error max={parity['maximum_absolute_error']:.6f}"
        )
    print(f"Reporte JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
