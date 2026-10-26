"""Comprueba que PyTorch puede ejecutar operaciones en la GPU NVIDIA."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA de PyTorch: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("ERROR: PyTorch no puede acceder a CUDA.", file=sys.stderr)
        return 1
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    tensor = torch.randn((1024, 1024), device="cuda")
    result = tensor @ tensor.T
    torch.cuda.synchronize()
    print(f"GPU: {properties.name}")
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"Prueba CUDA: OK ({result.device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
