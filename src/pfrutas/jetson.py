"""Empaquetado y medicion reproducible para NVIDIA Jetson Orin NX 16 GB.

El bundle conserva la arquitectura PyTorch validada y elimina exclusivamente
el estado de entrenamiento. Los checkpoints resultantes se cargan con
``weights_only=True`` y mantienen cada tensor del ``state_dict`` bit a bit.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from .features import FeatureStandardizer, PRESENCE_SLICE, static_feature_dim
from .model import (
    ASL_TRANSFER_MODEL_FAMILY,
    ASLTransferModelConfig,
    ModelConfig,
    SignModel,
    build_model,
)
from .utils import read_json, resolve_device, write_json


JETSON_BUNDLE_SCHEMA = "pfrutas_orin_nx16_bundle_v1"
JETSON_BUNDLE_SCHEMA_VERSION = 1
JETSON_BUNDLE_MANIFEST = "jetson_bundle.json"
JETSON_EXPECTED_MEMBER_COUNT = 6
JETSON_TARGET = "NVIDIA Jetson Orin NX 16GB"
DEFAULT_ORIN_BUNDLE_DIR = Path("artifacts/orin_nx16_fusion")

# El perfil principal (JetPack 6) usa 0.10.18. El perfil experimental
# JetPack 5/Python 3.8 necesita 0.10.5 porque es el último wheel oficial
# Linux ARM64 para cp38. Cada versión conserva su propia huella de activos.
MEDIAPIPE_SUPPORTED_VERSIONS = ("0.10.5", "0.10.18", "0.10.21")
MEDIAPIPE_CONTRACT: dict[str, Any] = {
    "api": "mediapipe.solutions.holistic.Holistic",
    "supported_versions": list(MEDIAPIPE_SUPPORTED_VERSIONS),
    "pose_landmarks": 33,
    "hand_landmarks_per_hand": 21,
    "minimum_face_landmarks": 468,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "refine_face_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "coordinate_contract": "pose33 + right_hand21 + left_hand21; normalized xy",
}

# Comprobados directamente en los wheels oficiales ARM64 0.10.5/0.10.18 y
# x86_64 0.10.21. Los diez activos de Holistic de este diccionario son idénticos
# entre esas versiones; ``solution_base.py`` se valida por versión más abajo.
# Exigir los hashes evita desplegar otro extractor bajo el mismo nombre y perder
# paridad con el entrenamiento.
MEDIAPIPE_COMMON_ASSET_SHA256: dict[str, str] = {
    "python/solutions/holistic.py": (
        "a9c5cff8c4b61bb3546bd001d4753666cbde4f686f1ddc5320deb378ea6ef9ea"
    ),
    "modules/holistic_landmark/holistic_landmark_cpu.binarypb": (
        "c1da3ea92cd0dfaeb35a1b3863a3a34202c6f4beffe410eff4a337569c808fbd"
    ),
    "modules/holistic_landmark/hand_recrop.tflite": (
        "67d996ce96f9d36fe17d2693022c6da93168026ab2f028f9e2365398d8ac7d5d"
    ),
    "modules/pose_detection/pose_detection.tflite": (
        "9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c"
    ),
    "modules/pose_landmark/pose_landmark_full.tflite": (
        "e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f"
    ),
    "modules/hand_landmark/hand_landmark_full.tflite": (
        "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3"
    ),
    "modules/palm_detection/palm_detection_full.tflite": (
        "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54"
    ),
    "modules/face_landmark/face_landmark_with_attention.tflite": (
        "e06a804e0144f9929eda782122916b35d60c697c3c9344013ca2bbe76a6ce2b4"
    ),
    "modules/face_detection/face_detection_short_range.tflite": (
        "bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8"
    ),
    "modules/iris_landmark/iris_landmark.tflite": (
        "d1744d2a09c25f501d39eba4faff47e53ecca8852c5ce19bce8eeac39357521f"
    ),
}
MEDIAPIPE_ASSET_SHA256_BY_VERSION: dict[str, dict[str, str]] = {
    "0.10.5": {
        **MEDIAPIPE_COMMON_ASSET_SHA256,
        "python/solution_base.py": (
            "0025e315bec76c2b76e1f3daf47eb7cfde07685a2a9c43499d51306e857dfa31"
        ),
    },
    "0.10.18": {
        **MEDIAPIPE_COMMON_ASSET_SHA256,
        "python/solution_base.py": (
            "9c422ab21a3d0e5bad7ef596cdd4b108ea49d90ccdee7dbddfc58b0e61739f7f"
        ),
    },
    "0.10.21": {
        **MEDIAPIPE_COMMON_ASSET_SHA256,
        "python/solution_base.py": (
            "9c422ab21a3d0e5bad7ef596cdd4b108ea49d90ccdee7dbddfc58b0e61739f7f"
        ),
    },
}
# Alias conservado para consumidores que importaban el contrato principal JP6.
MEDIAPIPE_ASSET_SHA256 = MEDIAPIPE_ASSET_SHA256_BY_VERSION["0.10.18"]
_HEX_SHA256_LENGTH = 64


@dataclass
class BundleMember:
    """Miembro cargado con el mismo contrato que ``live.LoadedMember``."""

    model: SignModel
    standardizer: FeatureStandardizer
    payload: dict[str, Any]
    relative_path: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_part(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Huella estable de nombres, formas, tipos y bytes de un state_dict."""
    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("model_state debe mapear nombres de texto a tensores.")
        if tensor.layout != torch.strided:
            raise TypeError(f"El tensor {key!r} no usa un layout denso soportado.")
        value = tensor.detach().cpu().contiguous()
        _hash_part(digest, key.encode("utf-8"))
        _hash_part(digest, str(value.dtype).encode("ascii"))
        shape = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        _hash_part(digest, shape)
        # ``reshape(-1)`` conserva los bytes y evita que ``view(uint8)`` falle
        # con los contadores BatchNorm escalares (tensores int64 de dimension 0).
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        _hash_part(digest, raw)
    return digest.hexdigest()


def _safe_relative_path(value: Any, field: str = "path") -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} debe ser una ruta relativa no vacia.")
    if "\\" in value:
        raise ValueError(f"{field} debe usar separadores POSIX: {value!r}.")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field} no puede ser absoluta: {value!r}.")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{field} contiene segmentos inseguros: {value!r}.")
    return path


def _resolve_inside(root: Path, relative: Path, field: str = "path") -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{field} sale del bundle: {relative.as_posix()!r}.") from exc
    return candidate


def _portable_basename(value: str) -> str:
    posix_name = Path(value).name
    windows_name = PureWindowsPath(value).name
    return windows_name if windows_name != value else posix_name


def _sanitize_for_weights_only(value: Any, key: str | None = None) -> Any:
    """Convierte metadatos a primitivas admitidas por el cargador seguro."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, str) and key and key.casefold().endswith("path"):
            if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
                return _portable_basename(value)
        return value
    # TorchVersion y otras subclases de str activan globals de pickle. La
    # conversion explicita produce un ``str`` nativo compatible con
    # weights_only=True.
    if isinstance(value, str):
        text = str(value)
        if key and key.casefold().endswith("path"):
            if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
                return _portable_basename(text)
        return text
    if isinstance(value, Path):
        return value.name if value.is_absolute() else value.as_posix()
    if isinstance(value, np.generic):
        return _sanitize_for_weights_only(value.item(), key=key)
    if isinstance(value, np.ndarray):
        return _sanitize_for_weights_only(value.tolist(), key=key)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            clean_key = str(raw_key)
            if clean_key in sanitized:
                raise ValueError(f"Clave duplicada tras sanear: {clean_key!r}.")
            sanitized[clean_key] = _sanitize_for_weights_only(item, key=clean_key)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_weights_only(item, key=key) for item in value]
    if isinstance(value, (set, frozenset)):
        clean = [_sanitize_for_weights_only(item, key=key) for item in value]
        return sorted(clean, key=lambda item: repr(item))
    raise TypeError(
        "El checkpoint contiene metadatos no seguros de tipo "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _assert_states_bit_exact(
    source: Mapping[str, Any],
    exported: Mapping[str, Any],
) -> None:
    import torch

    if set(source) != set(exported):
        raise RuntimeError("El checkpoint slim cambio las claves de model_state.")
    for key in source:
        left = source[key]
        right = exported[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"model_state[{key!r}] no es un tensor.")
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left.detach().cpu(), right.detach().cpu())
        ):
            raise RuntimeError(f"El tensor {key!r} no se conservo bit a bit.")


def export_slim_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    """Exporta un checkpoint de inferencia y comprueba su carga segura."""
    import torch

    if not source.is_file():
        raise FileNotFoundError(f"No existe el checkpoint fuente: {source}")
    # El artefacto fuente es un resultado local confiable del entrenamiento.
    # La salida deja de necesitar pickle general y siempre se valida con el
    # cargador restringido antes de publicarse.
    original = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(original, Mapping):
        raise TypeError(f"El checkpoint {source} no contiene un diccionario.")
    source_state = original.get("model_state")
    if not isinstance(source_state, Mapping) or not source_state:
        raise ValueError(f"El checkpoint {source} no contiene model_state valido.")

    slim_source = {key: value for key, value in original.items() if key != "optimizer_state"}
    slim = _sanitize_for_weights_only(slim_source)
    if not isinstance(slim, dict) or "optimizer_state" in slim:
        raise RuntimeError("No se pudo eliminar optimizer_state.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(slim, temporary)
    try:
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(verified, dict):
            raise RuntimeError("El checkpoint slim no se cargo como diccionario.")
        exported_state = verified.get("model_state")
        if not isinstance(exported_state, Mapping):
            raise RuntimeError("El checkpoint slim perdio model_state.")
        _assert_states_bit_exact(source_state, exported_state)
        source_tensor_hash = tensor_state_sha256(source_state)
        exported_tensor_hash = tensor_state_sha256(exported_state)
        if source_tensor_hash != exported_tensor_hash:
            raise RuntimeError("La huella tensorial cambio durante la exportacion.")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "source_size_bytes": source.stat().st_size,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "tensor_state_sha256": exported_tensor_hash,
        "optimizer_state_removed": True,
        "weights_only_verified": True,
    }


def load_slim_checkpoint(
    path: Path,
    device: str,
    expected_tensor_sha256: str | None = None,
) -> tuple[SignModel, FeatureStandardizer, dict[str, Any]]:
    """Carga segura de un miembro del bundle sin retener otra copia del estado."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "optimizer_state" in payload:
        raise RuntimeError(f"{path} no es un checkpoint slim valido.")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"{path} no contiene model_state.")
    if expected_tensor_sha256 is not None:
        if tensor_state_sha256(state) != expected_tensor_sha256:
            raise RuntimeError(f"Huella tensorial incorrecta en {path}.")

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"{path} no contiene model_config.")
    if raw_config.get("model_family") == ASL_TRANSFER_MODEL_FAMILY:
        config = ASLTransferModelConfig(**raw_config)
    else:
        config = ModelConfig(**raw_config)
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    raw_scaler = payload.get("feature_scaler")
    if not isinstance(raw_scaler, dict):
        raise RuntimeError(f"{path} no contiene feature_scaler.")
    standardizer = FeatureStandardizer.from_dict(raw_scaler)
    # Evita duplicar todos los pesos en RAM despues de construir el modelo.
    payload.pop("model_state", None)
    return model, standardizer, payload


def validate_mediapipe_runtime() -> dict[str, Any]:
    """Valida la API Holistic y calcula hashes de sus activos portables."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        allowed = ", ".join(MEDIAPIPE_SUPPORTED_VERSIONS)
        raise RuntimeError(
            "MediaPipe no esta instalado. En Linux aarch64 debe usarse un wheel "
            f"compatible con el contrato ({allowed})."
        ) from exc

    version = str(getattr(mp, "__version__", ""))
    if version not in MEDIAPIPE_SUPPORTED_VERSIONS:
        allowed = ", ".join(MEDIAPIPE_SUPPORTED_VERSIONS)
        raise RuntimeError(
            f"MediaPipe {version or 'sin version'} no cumple el contrato; "
            f"versiones admitidas: {allowed}."
        )
    solutions = getattr(mp, "solutions", None)
    holistic = getattr(solutions, "holistic", None)
    constructor = getattr(holistic, "Holistic", None)
    if not callable(constructor):
        raise RuntimeError("MediaPipe no expone mp.solutions.holistic.Holistic.")

    module_file = Path(getattr(mp, "__file__", ""))
    if not module_file.is_file():
        raise RuntimeError("No se pudo localizar el paquete MediaPipe.")
    package_root = module_file.parent.resolve()
    graph_reference = getattr(holistic, "_BINARYPB_FILE_PATH", "")
    expected_graph = "mediapipe/modules/holistic_landmark/holistic_landmark_cpu.binarypb"
    if str(graph_reference) != expected_graph:
        raise RuntimeError(
            "MediaPipe Holistic referencia un grafo distinto del validado."
        )

    expected_assets = MEDIAPIPE_ASSET_SHA256_BY_VERSION[version]
    assets: list[dict[str, Any]] = []
    for relative, expected_sha256 in sorted(expected_assets.items()):
        path = package_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Falta el activo MediaPipe validado: {relative}.")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 incompatible para el activo MediaPipe {relative}."
            )
        assets.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    return {
        "valid": True,
        "summary": f"MediaPipe {version}: Holistic y activos SHA-256 verificados",
        "package": "mediapipe",
        "version": version,
        "contract": MEDIAPIPE_CONTRACT,
        "assets": assets,
        "asset_hash_algorithm": "sha256",
    }


def _asset_entry(path: Path, root: Path, kind: str, **extra: Any) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    result = {
        "path": relative,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    result.update(extra)
    return result


def _checkpoint_paths(ensemble: Mapping[str, Any]) -> list[str]:
    values = ensemble.get("checkpoints")
    if (
        not isinstance(values, list)
        or len(values) != JETSON_EXPECTED_MEMBER_COUNT
        or any(not isinstance(value, str) for value in values)
        or len(set(values)) != len(values)
    ):
        raise RuntimeError(
            "El ensemble Orin debe declarar exactamente seis checkpoints unicos."
        )
    for value in values:
        _safe_relative_path(value, "ensemble.checkpoints")
    return values


def _selected_checkpoint_paths(
    ensemble: Mapping[str, Any],
    mode: str,
) -> list[str]:
    all_paths = _checkpoint_paths(ensemble)
    if mode == "full":
        return all_paths
    if mode == "folds":
        values = ensemble.get("fold_checkpoints")
        if not isinstance(values, list) or not values:
            final_value = ensemble.get("final_checkpoint")
            values = [value for value in all_paths if value != final_value]
    elif mode == "final":
        final_value = ensemble.get("final_checkpoint")
        if not isinstance(final_value, str):
            raise RuntimeError("El ensemble no declara final_checkpoint.")
        values = [final_value]
    else:
        raise ValueError("mode debe ser full, folds o final.")
    if any(value not in all_paths for value in values):
        raise RuntimeError(f"La seleccion {mode!r} contiene checkpoints desconocidos.")
    return list(values)


def _load_members_from_bundle(
    bundle_dir: Path,
    ensemble: dict[str, Any],
    manifest: dict[str, Any],
    device: str,
    mode: str,
) -> list[BundleMember]:
    asset_by_path = {
        item["path"]: item
        for item in manifest["assets"]
        if item.get("kind") == "checkpoint"
    }
    members: list[BundleMember] = []
    for relative_text in _selected_checkpoint_paths(ensemble, mode):
        relative = _safe_relative_path(relative_text, "checkpoint")
        asset = asset_by_path.get(relative_text)
        if asset is None:
            raise RuntimeError(f"El manifiesto no registra {relative_text}.")
        model, standardizer, payload = load_slim_checkpoint(
            _resolve_inside(bundle_dir, relative, "checkpoint"),
            device,
            expected_tensor_sha256=asset.get("tensor_state_sha256"),
        )
        if payload.get("labels") != ensemble.get("labels"):
            raise RuntimeError(f"Orden de clases incompatible en {relative_text}.")
        if payload.get("model_config") != ensemble.get("model_config"):
            raise RuntimeError(f"model_config incompatible en {relative_text}.")
        members.append(BundleMember(model, standardizer, payload, relative_text))
    return members


def load_bundle_members(
    bundle_dir: Path,
    device: str = "cuda",
    mode: str = "full",
    validate: bool = True,
) -> tuple[list[BundleMember], dict[str, Any], dict[str, Any]]:
    """Carga miembros seleccionados del bundle usando solo deserializacion segura."""
    manifest = validate_bundle(bundle_dir, verify_checkpoints=False) if validate else read_json(
        bundle_dir / JETSON_BUNDLE_MANIFEST
    )
    ensemble = read_json(bundle_dir / "ensemble.json")
    members = _load_members_from_bundle(bundle_dir, ensemble, manifest, device, mode)
    return members, ensemble, manifest


def _load_processed_parity_sample(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la muestra de paridad: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "features" not in payload or "timestamps" not in payload:
            raise ValueError("La muestra procesada requiere features y timestamps.")
        sequence = np.asarray(payload["features"], dtype=np.float32)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        raw = (
            np.asarray(payload["pose_hands_xy"], dtype=np.float32)
            if "pose_hands_xy" in payload
            else None
        )
    if sequence.ndim != 2 or sequence.shape[1] != static_feature_dim() or len(sequence) == 0:
        raise ValueError("features no cumple el layout estatico v2.")
    if (
        timestamps.shape != (len(sequence),)
        or not np.all(np.isfinite(timestamps))
        or (len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0))
    ):
        raise ValueError("timestamps de la muestra no son estrictamente crecientes.")
    result = {"static_sequence": sequence, "timestamps": timestamps}
    if raw is not None:
        if raw.shape != (len(sequence), 75, 2) or not np.all(np.isfinite(raw)):
            raise ValueError("pose_hands_xy de la muestra es invalido.")
        result["pose_hands_xy"] = raw
    return result


def _write_parity_sample(
    destination: Path,
    sample: dict[str, np.ndarray],
    members: Sequence[BundleMember],
    ensemble: dict[str, Any],
    device: str,
    source_name: str,
) -> dict[str, Any]:
    from .live import ensemble_predict

    probabilities = ensemble_predict(
        list(members),  # BundleMember cumple el contrato estructural requerido.
        sample["static_sequence"],
        device,
        timestamps=sample["timestamps"],
        pose_hands_xy=sample.get("pose_hands_xy"),
    )
    if probabilities.shape != (len(ensemble["labels"]),):
        raise RuntimeError("La prediccion golden tiene una dimension inesperada.")
    arrays: dict[str, Any] = dict(sample)
    arrays.update(
        {
            "golden_probabilities": probabilities.astype(np.float32),
            "labels": np.asarray(ensemble["labels"], dtype=np.str_),
            "source_name": np.asarray(source_name, dtype=np.str_),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "path": destination.name,
        "golden_device": device,
        "absolute_tolerance": 0.02,
        "argmax_must_match": True,
        "source_name": source_name,
    }


def build_orin_bundle(
    run_dir: Path,
    bundle_dir: Path,
    parity_sample: Path | None = None,
    parity_device: str = "cpu",
    include_mediapipe_fingerprint: bool = True,
) -> dict[str, Any]:
    """Construye atomicamente el bundle Orin a partir de una corrida completa."""
    run_dir = run_dir.resolve()
    bundle_dir = bundle_dir.resolve()
    if bundle_dir.exists():
        raise FileExistsError(
            f"El destino {bundle_dir} ya existe; usa un directorio nuevo."
        )
    ensemble_source = run_dir / "ensemble.json"
    if not ensemble_source.is_file():
        raise FileNotFoundError(f"No existe {ensemble_source}.")
    ensemble = read_json(ensemble_source)
    if not isinstance(ensemble, dict):
        raise RuntimeError("ensemble.json debe contener un objeto JSON.")
    checkpoints = _checkpoint_paths(ensemble)
    cv_relative = _safe_relative_path(
        ensemble.get("cv_metrics", "cv_metrics.json"),
        "ensemble.cv_metrics",
    )
    cv_source = _resolve_inside(run_dir, cv_relative, "ensemble.cv_metrics")
    if not cv_source.is_file():
        raise FileNotFoundError(f"No existe {cv_source}.")

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_dir.name}.tmp-",
            dir=bundle_dir.parent,
        )
    )
    assets: list[dict[str, Any]] = []
    try:
        ensemble_destination = staging / "ensemble.json"
        shutil.copyfile(ensemble_source, ensemble_destination)
        assets.append(_asset_entry(ensemble_destination, staging, "metadata"))

        cv_destination = staging / cv_relative
        cv_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cv_source, cv_destination)
        assets.append(_asset_entry(cv_destination, staging, "metrics"))

        for relative_text in checkpoints:
            relative = _safe_relative_path(relative_text, "checkpoint")
            source = _resolve_inside(run_dir, relative, "checkpoint")
            destination = staging / relative
            info = export_slim_checkpoint(source, destination)
            assets.append(
                _asset_entry(
                    destination,
                    staging,
                    "checkpoint",
                    tensor_state_sha256=info["tensor_state_sha256"],
                    optimizer_state_removed=True,
                    weights_only_verified=True,
                    source_size_bytes=info["source_size_bytes"],
                )
            )

        mediapipe_runtime = (
            validate_mediapipe_runtime()
            if include_mediapipe_fingerprint
            else {
                "package": "mediapipe",
                "contract": MEDIAPIPE_CONTRACT,
                "build_runtime_fingerprint": "not_collected",
            }
        )
        manifest: dict[str, Any] = {
            "schema": JETSON_BUNDLE_SCHEMA,
            "schema_version": JETSON_BUNDLE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": {
                "device": JETSON_TARGET,
                "architecture": "aarch64",
                "runtime": "PyTorch CUDA autocast FP16",
                "default_member_mode": "full",
            },
            "source_run": {
                "name": run_dir.name,
                "run_fingerprint": ensemble.get("run_fingerprint"),
                "dataset_content_fingerprint": ensemble.get(
                    "dataset_content_fingerprint"
                ),
            },
            "model": {
                "member_count": len(checkpoints),
                "checkpoint_paths": checkpoints,
                "labels": ensemble.get("labels"),
                "model_config": ensemble.get("model_config"),
                "recommended_confidence_threshold": ensemble.get(
                    "recommended_confidence_threshold"
                ),
            },
            "checkpoint_contract": {
                "serialization": "torch.save state_dict + primitive metadata",
                "loader": "torch.load(weights_only=True)",
                "optimizer_state_present": False,
                "tensor_preservation": "bit_exact_from_training_checkpoint",
                "tensor_hash": "sha256(name,dtype,shape,raw_bytes)",
            },
            "mediapipe": mediapipe_runtime,
            "parity": None,
            "assets": assets,
        }
        checkpoint_assets = [
            item for item in assets if item.get("kind") == "checkpoint"
        ]
        manifest["checkpoint_count"] = len(checkpoint_assets)
        manifest["total_checkpoint_bytes"] = sum(
            int(item["size_bytes"]) for item in checkpoint_assets
        )

        if parity_sample is not None:
            parity_device = resolve_device(parity_device)
            members = _load_members_from_bundle(
                staging,
                ensemble,
                manifest,
                parity_device,
                "full",
            )
            sample = _load_processed_parity_sample(parity_sample)
            parity_path = staging / "parity_sample.npz"
            manifest["parity"] = _write_parity_sample(
                parity_path,
                sample,
                members,
                ensemble,
                parity_device,
                parity_sample.name,
            )
            assets.append(_asset_entry(parity_path, staging, "parity"))
            del members

        write_json(staging / JETSON_BUNDLE_MANIFEST, manifest)
        validate_bundle(staging, verify_checkpoints=True)
        staging.replace(bundle_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_bundle(bundle_dir, verify_checkpoints=False)


def _validate_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"{field} no contiene un SHA-256 hexadecimal valido.")
    return value


def _validate_parity_asset(path: Path, labels: list[str]) -> None:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "static_sequence",
            "timestamps",
            "golden_probabilities",
            "labels",
        }
        if not required.issubset(payload.files):
            raise RuntimeError("parity_sample.npz esta incompleto.")
        sequence = np.asarray(payload["static_sequence"])
        timestamps = np.asarray(payload["timestamps"])
        probabilities = np.asarray(payload["golden_probabilities"])
        parity_labels = [str(value) for value in payload["labels"].tolist()]
        if sequence.ndim != 2 or sequence.shape[1] != static_feature_dim():
            raise RuntimeError("La secuencia de paridad no usa el layout v2.")
        if timestamps.shape != (len(sequence),):
            raise RuntimeError("Los timestamps de paridad no coinciden.")
        if probabilities.shape != (len(labels),) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Las probabilidades golden son invalidas.")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
            raise RuntimeError("Las probabilidades golden no suman uno.")
        if parity_labels != labels:
            raise RuntimeError("Las etiquetas de paridad no coinciden con el ensemble.")


def validate_bundle(
    bundle_dir: Path,
    verify_checkpoints: bool = True,
) -> dict[str, Any]:
    """Valida esquema, rutas, hashes y checkpoints de un bundle Orin."""
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / JETSON_BUNDLE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe {manifest_path}.")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("jetson_bundle.json debe contener un objeto.")
    if (
        manifest.get("schema") != JETSON_BUNDLE_SCHEMA
        or manifest.get("schema_version") != JETSON_BUNDLE_SCHEMA_VERSION
    ):
        raise RuntimeError("Version de esquema de bundle incompatible.")
    model_info = manifest.get("model")
    if not isinstance(model_info, dict):
        raise RuntimeError("El manifiesto no contiene model.")
    if model_info.get("member_count") != JETSON_EXPECTED_MEMBER_COUNT:
        raise RuntimeError("El bundle debe contener exactamente seis miembros.")
    labels = model_info.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) for x in labels):
        raise RuntimeError("El bundle no contiene etiquetas validas.")

    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("El manifiesto no contiene activos.")
    seen: set[str] = set()
    asset_by_path: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(assets):
        if not isinstance(item, dict):
            raise RuntimeError(f"assets[{index}] no es un objeto.")
        relative = _safe_relative_path(item.get("path"), f"assets[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise RuntimeError(f"Activo duplicado: {relative_text}.")
        seen.add(relative_text)
        path = _resolve_inside(bundle_dir, relative, f"assets[{index}].path")
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Activo ausente o inseguro: {relative_text}.")
        expected_size = item.get("size_bytes")
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            raise RuntimeError(f"Tamano incorrecto para {relative_text}.")
        expected_sha = _validate_sha256(item.get("sha256"), f"{relative_text}.sha256")
        if _sha256_file(path) != expected_sha:
            raise RuntimeError(f"SHA-256 incorrecto para {relative_text}.")
        asset_by_path[relative_text] = item

    ensemble_asset = asset_by_path.get("ensemble.json")
    if ensemble_asset is None or ensemble_asset.get("kind") != "metadata":
        raise RuntimeError("ensemble.json no esta registrado como metadata.")
    ensemble = read_json(bundle_dir / "ensemble.json")
    if not isinstance(ensemble, dict):
        raise RuntimeError("ensemble.json es invalido.")
    checkpoint_paths = _checkpoint_paths(ensemble)
    if checkpoint_paths != model_info.get("checkpoint_paths"):
        raise RuntimeError("Las rutas de checkpoints no coinciden con el manifiesto.")
    if ensemble.get("labels") != labels or ensemble.get("model_config") != model_info.get(
        "model_config"
    ):
        raise RuntimeError("El contrato del ensemble difiere del manifiesto.")
    cv_path = _safe_relative_path(
        ensemble.get("cv_metrics", "cv_metrics.json"),
        "ensemble.cv_metrics",
    ).as_posix()
    if cv_path not in asset_by_path or asset_by_path[cv_path].get("kind") != "metrics":
        raise RuntimeError("Las metricas CV no estan registradas.")

    for relative_text in checkpoint_paths:
        item = asset_by_path.get(relative_text)
        if item is None or item.get("kind") != "checkpoint":
            raise RuntimeError(f"Checkpoint no registrado: {relative_text}.")
        tensor_hash = _validate_sha256(
            item.get("tensor_state_sha256"),
            f"{relative_text}.tensor_state_sha256",
        )
        if (
            item.get("optimizer_state_removed") is not True
            or item.get("weights_only_verified") is not True
        ):
            raise RuntimeError(f"Checkpoint sin contrato slim: {relative_text}.")
        if verify_checkpoints:
            model, _, payload = load_slim_checkpoint(
                bundle_dir / relative_text,
                "cpu",
                expected_tensor_sha256=tensor_hash,
            )
            if payload.get("labels") != labels:
                raise RuntimeError(f"Clases incompatibles en {relative_text}.")
            del model, payload

    checkpoint_assets = [
        item for item in assets if item.get("kind") == "checkpoint"
    ]
    checkpoint_bytes = sum(int(item["size_bytes"]) for item in checkpoint_assets)
    if (
        manifest.get("checkpoint_count") != len(checkpoint_assets)
        or manifest.get("total_checkpoint_bytes") != checkpoint_bytes
    ):
        raise RuntimeError("El resumen de checkpoints del manifiesto es incorrecto.")

    parity = manifest.get("parity")
    if parity is not None:
        if not isinstance(parity, dict):
            raise RuntimeError("El contrato parity es invalido.")
        parity_relative = _safe_relative_path(parity.get("path"), "parity.path")
        parity_text = parity_relative.as_posix()
        if parity_text not in asset_by_path or asset_by_path[parity_text].get("kind") != "parity":
            raise RuntimeError("La muestra de paridad no esta registrada.")
        _validate_parity_asset(bundle_dir / parity_relative, labels)
    return manifest


def _synthetic_benchmark_sample(ensemble: Mapping[str, Any]) -> dict[str, np.ndarray]:
    frames = max(int(ensemble.get("sequence_length", 48)), 60)
    target_fps = float(ensemble.get("target_fps", 30.0))
    sequence = np.zeros((frames, static_feature_dim()), dtype=np.float32)
    phase = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)
    sequence[:, 0] = 0.05 * np.sin(phase)
    sequence[:, 1] = 0.05 * np.cos(phase)
    sequence[:, PRESENCE_SLICE] = 1.0
    timestamps = np.arange(frames, dtype=np.float64) / target_fps

    raw = np.full((frames, 75, 2), 0.5, dtype=np.float32)
    raw[:, 11, 0] = 0.35
    raw[:, 12, 0] = 0.65
    raw[:, 11:13, 1] = 0.40
    raw[:, 33:, 0] += 0.04 * np.sin(phase)[:, None]
    return {
        "static_sequence": sequence,
        "timestamps": timestamps,
        "pose_hands_xy": raw,
    }


def _load_bundle_benchmark_sample(
    bundle_dir: Path,
    ensemble: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray | None, str]:
    path = bundle_dir / "parity_sample.npz"
    if not path.is_file():
        return _synthetic_benchmark_sample(ensemble), None, "synthetic"
    with np.load(path, allow_pickle=False) as payload:
        sample = {
            "static_sequence": np.asarray(payload["static_sequence"], dtype=np.float32),
            "timestamps": np.asarray(payload["timestamps"], dtype=np.float64),
        }
        if "pose_hands_xy" in payload:
            sample["pose_hands_xy"] = np.asarray(
                payload["pose_hands_xy"], dtype=np.float32
            )
        golden = np.asarray(payload["golden_probabilities"], dtype=np.float32)
    return sample, golden, "parity_sample.npz"


def summarize_latencies(latencies_seconds: Sequence[float]) -> dict[str, float]:
    values = np.asarray(latencies_seconds, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Se requiere al menos una latencia finita.")
    if np.any(values <= 0.0):
        raise ValueError("Las latencias deben ser positivas.")
    mean = float(values.mean())
    return {
        "mean_ms": mean * 1000.0,
        "p50_ms": float(np.percentile(values, 50)) * 1000.0,
        "p95_ms": float(np.percentile(values, 95)) * 1000.0,
        "min_ms": float(values.min()) * 1000.0,
        "max_ms": float(values.max()) * 1000.0,
        "throughput_fps": 1.0 / mean,
    }


def benchmark_bundle(
    bundle_dir: Path,
    device_request: str = "cuda",
    mode: str = "full",
    warmup: int = 5,
    iterations: int = 20,
) -> dict[str, Any]:
    """Mide preprocesamiento del buffer + ensemble; excluye MediaPipe/camara."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup debe ser >= 0 e iterations debe ser >= 1.")
    device = resolve_device(device_request)
    members, ensemble, manifest = load_bundle_members(
        bundle_dir,
        device=device,
        mode=mode,
        validate=True,
    )
    sample, golden, sample_name = _load_bundle_benchmark_sample(bundle_dir, ensemble)

    import torch

    from .live import ensemble_predict

    synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    predictions: list[np.ndarray] = []
    for _ in range(warmup):
        ensemble_predict(
            members,
            sample["static_sequence"],
            device,
            timestamps=sample["timestamps"],
            pose_hands_xy=sample.get("pose_hands_xy"),
        )
    synchronize()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    for _ in range(iterations):
        synchronize()
        started = time.perf_counter()
        prediction = ensemble_predict(
            members,
            sample["static_sequence"],
            device,
            timestamps=sample["timestamps"],
            pose_hands_xy=sample.get("pose_hands_xy"),
        )
        synchronize()
        latencies.append(time.perf_counter() - started)
        predictions.append(prediction)

    latency = summarize_latencies(latencies)
    reference = predictions[0]
    consistency = max(
        float(np.max(np.abs(reference - prediction))) for prediction in predictions
    )
    predicted_index = int(np.argmax(reference))
    memory: dict[str, Any] = {"backend": device}
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
    }
    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        memory.update(
            {
                "model_allocated_bytes": torch.cuda.memory_allocated(index),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
                "device_total_bytes": properties.total_memory,
            }
        )
        runtime.update(
            {
                "gpu_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )

    parity_result: dict[str, Any] | None = None
    if golden is not None:
        maximum_error = float(np.max(np.abs(reference - golden)))
        tolerance = float((manifest.get("parity") or {}).get("absolute_tolerance", 0.02))
        parity_result = {
            "maximum_absolute_error": maximum_error,
            "absolute_tolerance": tolerance,
            "within_tolerance": maximum_error <= tolerance,
            "argmax_matches": int(np.argmax(golden)) == predicted_index,
        }

    return {
        "schema": "pfrutas_orin_nx16_benchmark_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": bundle_dir.name,
        "bundle_schema": manifest["schema"],
        "mode": mode,
        "member_count": len(members),
        "device": device,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "scope": "buffer preprocessing + PyTorch ensemble; excludes MediaPipe and camera",
        "sample": sample_name,
        "latency": latency,
        "memory": memory,
        "runtime": runtime,
        "prediction": {
            "label": ensemble["labels"][predicted_index],
            "confidence": float(reference[predicted_index]),
            "consistency_max_abs": consistency,
        },
        "parity": parity_result,
    }
