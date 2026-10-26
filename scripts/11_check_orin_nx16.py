"""Preflight reproducible para el despliegue PFrutas en Orin NX 16 GB.

No cambia modos de potencia ni instala paquetes. Comprueba la placa, el runtime
CUDA, OpenCV/GStreamer, el contrato MediaPipe y la integridad del bundle.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


RUNTIME_PROFILES: dict[str, dict[str, Any]] = {
    "jetpack6": {
        "label": "JetPack 6",
        "python": (3, 10),
        "l4t_marker": "R36",
        "jetpack_prefixes": ("6.1", "6.2"),
        "mediapipe_version": "0.10.18",
        "torch_version": None,
    },
    "jetpack5": {
        "label": "JetPack 5.1.5",
        "python": (3, 8),
        "l4t_marker": "R35 (release), REVISION: 6.1",
        "jetpack_prefixes": ("5.1.5",),
        "mediapipe_version": "0.10.5",
        "torch_version": "2.1.0a0+41361538.nv23.06",
    },
}


def _text_file(path: Path) -> str | None:
    try:
        return path.read_bytes().replace(b"\x00", b"").decode(
            "utf-8", errors="replace"
        ).strip()
    except OSError:
        return None


def _memory_gib() -> float | None:
    content = _text_file(Path("/proc/meminfo"))
    if content is None:
        return None
    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            try:
                return float(line.split()[1]) * 1024.0 / 1024**3
            except (IndexError, ValueError):
                return None
    return None


def _gstreamer_enabled(build_information: str) -> bool:
    for line in build_information.splitlines():
        if line.strip().startswith("GStreamer:"):
            return line.split(":", 1)[1].strip().upper().startswith("YES")
    return False


def _nvpmodel_query() -> str | None:
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output or None


def _package_version(package: str) -> str | None:
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _matches_jetpack_version(
    version: str | None,
    prefixes: tuple[str, ...],
) -> bool:
    return version is not None and any(version.startswith(prefix) for prefix in prefixes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/orin_nx16_fusion"),
    )
    parser.add_argument(
        "--camera-mode",
        choices=("usb", "csi"),
        default="usb",
        help="Con CSI, GStreamer pasa a ser un requisito estricto.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=tuple(RUNTIME_PROFILES),
        default="jetpack6",
        help=(
            "Contrato de sistema que se valida. JetPack 5 es el perfil "
            "experimental para Ubuntu 20.04/Python 3.8."
        ),
    )
    parser.add_argument(
        "--allow-non-jetson",
        action="store_true",
        help="Convierte las comprobaciones de hardware en avisos para probar el script en el host.",
    )
    parser.add_argument("--skip-bundle", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    runtime_profile = RUNTIME_PROFILES[args.runtime_profile]

    checks: list[Check] = []
    environment: dict[str, Any] = {}

    def add(name: str, ok: bool, detail: str, *, hardware: bool = False) -> None:
        status = "ok" if ok else ("warning" if hardware and args.allow_non_jetson else "error")
        checks.append(Check(name=name, status=status, detail=detail))

    machine = platform.machine().lower()
    model = _text_file(Path("/proc/device-tree/model"))
    l4t = _text_file(Path("/etc/nv_tegra_release"))
    memory_gib = _memory_gib()
    jetpack_version = _package_version("nvidia-jetpack")
    environment.update(
        {
            "runtime_profile": args.runtime_profile,
            "runtime_profile_label": runtime_profile["label"],
            "platform": platform.platform(),
            "machine": machine,
            "python": platform.python_version(),
            "device_model": model,
            "l4t": l4t,
            "jetpack": jetpack_version,
            "memory_gib": memory_gib,
            "nvpmodel": _nvpmodel_query(),
        }
    )
    add("Arquitectura ARM64", machine in {"aarch64", "arm64"}, machine, hardware=True)
    add("Jetson Linux", l4t is not None, l4t or "No se encontró /etc/nv_tegra_release", hardware=True)
    l4t_marker = runtime_profile["l4t_marker"]
    add(
        f"L4T del perfil {runtime_profile['label']}",
        l4t is not None and l4t_marker in l4t,
        f"{l4t or 'No se encontró L4T'}; se requiere {l4t_marker}.",
        hardware=True,
    )
    jetpack_prefixes = runtime_profile["jetpack_prefixes"]
    add(
        f"Metapaquete {runtime_profile['label']}",
        _matches_jetpack_version(jetpack_version, jetpack_prefixes),
        (
            f"{jetpack_version or 'No se encontró nvidia-jetpack'}; se requiere "
            f"una versión que empiece por {', '.join(jetpack_prefixes)}."
        ),
        hardware=True,
    )
    add(
        "Módulo Orin NX",
        model is not None and "orin nx" in model.lower(),
        model or "No se pudo leer /proc/device-tree/model",
        hardware=True,
    )
    add(
        "RAM de 16 GB",
        memory_gib is not None and memory_gib >= 14.0,
        f"{memory_gib:.2f} GiB detectados" if memory_gib is not None else "Memoria desconocida",
        hardware=True,
    )
    expected_python = runtime_profile["python"]
    python_ok = sys.version_info[:2] == expected_python
    add(
        f"Python del perfil {runtime_profile['label']}",
        python_ok,
        platform.python_version()
        + (
            ""
            if python_ok
            else (
                "; se requiere "
                f"{expected_python[0]}.{expected_python[1]} en este perfil"
            )
        ),
        hardware=True,
    )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        installed_torch = str(torch.__version__)
        torch_info: dict[str, Any] = {
            "version": installed_torch,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
        }
        expected_torch = runtime_profile["torch_version"]
        torch_ok = cuda_available and (
            expected_torch is None or installed_torch == expected_torch
        )
        torch_detail = (
            f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
            f"cuDNN={torch.backends.cudnn.version()}"
        )
        if expected_torch is not None and installed_torch != expected_torch:
            torch_detail += f"; este perfil requiere torch={expected_torch}."
        add("PyTorch CUDA", torch_ok, torch_detail)
        if cuda_available:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            capability = (properties.major, properties.minor)
            torch_info.update(
                {
                    "gpu": properties.name,
                    "compute_capability": list(capability),
                    "gpu_memory_gib": properties.total_memory / 1024**3,
                }
            )
            add(
                "GPU Orin (SM 8.7)",
                capability == (8, 7),
                f"{properties.name}, SM {properties.major}.{properties.minor}, "
                f"{properties.total_memory / 1024**3:.2f} GiB",
                hardware=True,
            )
            try:
                tensor = torch.randn((256, 256), device="cuda")
                result = tensor @ tensor.T
                torch.cuda.synchronize()
                add("Operación CUDA real", bool(torch.isfinite(result).all()), str(result.device))
            except Exception as exc:  # pragma: no cover - depende del driver
                add("Operación CUDA real", False, f"{type(exc).__name__}: {exc}")
        environment["torch"] = torch_info
    except Exception as exc:
        add("PyTorch CUDA", False, f"{type(exc).__name__}: {exc}")

    try:
        import cv2

        build_information = cv2.getBuildInformation()
        gstreamer = _gstreamer_enabled(build_information)
        environment["opencv"] = {
            "version": cv2.__version__,
            "gstreamer": gstreamer,
        }
        add("OpenCV", True, f"cv2={cv2.__version__}")
        add(
            "OpenCV con GStreamer",
            gstreamer or args.camera_mode == "usb",
            "YES" if gstreamer else "NO; USB funciona, CSI requiere el OpenCV de JetPack",
        )
    except Exception as exc:
        add("OpenCV", False, f"{type(exc).__name__}: {exc}")

    try:
        from pfrutas.jetson import validate_mediapipe_runtime

        media_report = validate_mediapipe_runtime()
        environment["mediapipe"] = media_report
        expected_mediapipe = runtime_profile["mediapipe_version"]
        media_ok = (
            bool(media_report.get("valid"))
            and media_report.get("version") == expected_mediapipe
        )
        add(
            f"MediaPipe Holistic {expected_mediapipe}",
            media_ok,
            str(media_report.get("summary", media_report))
            + (
                ""
                if media_ok
                else f"; este perfil requiere MediaPipe {expected_mediapipe}."
            ),
        )
    except Exception as exc:
        add(
            f"MediaPipe Holistic {runtime_profile['mediapipe_version']}",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    if not args.skip_bundle:
        try:
            from pfrutas.jetson import validate_bundle

            bundle_report = validate_bundle(args.bundle_dir, verify_checkpoints=True)
            environment["bundle"] = bundle_report
            add(
                "Bundle Orin",
                True,
                f"{bundle_report.get('checkpoint_count', '?')} checkpoints; "
                f"{bundle_report.get('total_checkpoint_bytes', 0) / 1024**2:.1f} MiB",
            )
        except Exception as exc:
            add("Bundle Orin", False, f"{type(exc).__name__}: {exc}")

    payload = {
        "schema": "pfrutas_orin_nx16_preflight_v1",
        "environment": environment,
        "checks": [asdict(check) for check in checks],
        "ok": not any(check.status == "error" for check in checks),
    }
    for check in checks:
        marker = {"ok": "OK", "warning": "AVISO", "error": "ERROR"}[check.status]
        print(f"[{marker:5}] {check.name}: {check.detail}")
    if environment.get("nvpmodel"):
        print("\nModo de potencia actual (solo consulta):")
        print(environment["nvpmodel"])
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nReporte: {args.json_out}")
    print("\nPreflight aprobado." if payload["ok"] else "\nPreflight falló; corrige los ERROR antes de ejecutar la cámara.")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
