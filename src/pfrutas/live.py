from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import transfer_feature_layout
from .features import (
    HolisticFeatureExtractor,
    POSE_HANDS_XY_LAYOUT_VERSION,
    POSE_HANDS_XY_POINT_COUNT,
    TEMPORAL_VIEW_EXPONENTS,
    add_temporal_deltas,
    deterministic_views,
    feature_layout,
    model_feature_dim,
    static_feature_dim,
)
from .model import (
    ASL_TRANSFER_MODEL_FAMILY,
    ASLTransferFusion,
    SignModel,
    load_checkpoint,
)
from .training import (
    EXPERIMENT_IMPLEMENTATION_VERSION,
    SUPPORTED_EXPERIMENT_IMPLEMENTATION_VERSIONS,
    TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION,
)
from .utils import ascii_text, read_json, resolve_device, resolve_label


@dataclass
class LoadedMember:
    model: SignModel
    standardizer: Any
    payload: dict[str, Any]


@dataclass(frozen=True)
class PredictionPanel:
    """Contenido independiente de OpenCV para un resultado en vivo."""

    primary_text: str
    confidence_text: str
    cv_text: str | None
    status_text: str | None
    session_text: str | None
    positive: bool
    show_probability_bars: bool
    show_global_metrics: bool


def build_general_oof_metrics_text(
    accuracy: float,
    macro_precision: float,
) -> str:
    """Devuelve la métrica global que sí es válida en modo general.

    ``accuracy`` y ``macro_precision`` proceden de predicciones *out-of-fold*
    (OOF), no de la confianza de la cámara actual. Mantener esa procedencia en
    el texto evita presentar una probabilidad instantánea como si fuera la
    exactitud validada del modelo.
    """

    values = (float(accuracy), float(macro_precision))
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Las métricas OOF deben estar en el intervalo [0, 1].")
    return (
        f"Exactitud OOF general: {values[0]:.1%} | "
        f"precision macro: {values[1]:.1%}"
    )


def _unit_fraction(value: Any) -> float | None:
    """Convierte una métrica válida a fracción o devuelve ``None``.

    Una métrica ausente o corrupta nunca se representa como ``0 %``: ese valor
    parecería una medición real y podría llevar a una conclusión incorrecta.
    """

    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _class_oof_metrics(
    per_class: dict[str, Any],
    label: str,
) -> tuple[float, float, float] | None:
    """Obtiene precisión, recall y F1 de una clase si el registro es válido."""

    candidate = per_class.get(label)
    if not isinstance(candidate, dict):
        return None
    precision = _unit_fraction(candidate.get("precision"))
    recall = _unit_fraction(candidate.get("recall"))
    f1 = _unit_fraction(candidate.get("f1"))
    if precision is None or recall is None or f1 is None:
        return None
    return precision, recall, f1


def _class_oof_precision(per_class: dict[str, Any], label: str) -> float | None:
    """Obtiene la precisión OOF de una clase sin inventar un valor faltante."""

    candidate = per_class.get(label)
    if not isinstance(candidate, dict):
        return None
    return _unit_fraction(candidate.get("precision"))


def build_prediction_panel(
    labels: list[str],
    probabilities: np.ndarray,
    threshold: float,
    per_class: dict[str, Any],
    target_index: int | None = None,
    attempts: int = 0,
    correct_attempts: int = 0,
    show_cv_metrics: bool = True,
    show_global_metrics: bool | None = None,
) -> PredictionPanel:
    """Construye el texto visible sin depender de cámara ni de OpenCV.

    En modo objetivo, el resultado se expresa exclusivamente respecto a la
    fruta solicitada. En particular, nunca incluye el nombre ni las métricas
    de la clase que ganó el ``argmax`` cuando esa clase es otra fruta.
    """

    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape != (len(labels),):
        raise ValueError("Las probabilidades deben coincidir con las etiquetas.")
    if not labels:
        raise ValueError("Se necesita al menos una etiqueta.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Las probabilidades deben ser finitas.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    if attempts < 0 or correct_attempts < 0 or correct_attempts > attempts:
        raise ValueError("Los contadores de sesión son inválidos.")

    predicted_index = int(np.argmax(values))
    predicted_label = labels[predicted_index]
    predicted_confidence = float(values[predicted_index])
    global_metrics_visible = (
        show_cv_metrics if show_global_metrics is None else show_global_metrics
    )

    if target_index is None:
        predicted_precision = _class_oof_precision(per_class, predicted_label)
        return PredictionPanel(
            primary_text=f"Prediccion: {predicted_label}",
            confidence_text=(
                f"Confianza actual: {predicted_confidence:.1%}  "
                f"(umbral heuristico {threshold:.0%})"
            ),
            cv_text=(
                (
                    f"Precision OOF de {predicted_label}: "
                    f"{predicted_precision:.1%}"
                )
                if show_cv_metrics and predicted_precision is not None
                else None
            ),
            status_text=None,
            session_text=None,
            positive=predicted_confidence >= threshold,
            show_probability_bars=True,
            show_global_metrics=global_metrics_visible,
        )

    if not 0 <= target_index < len(labels):
        raise ValueError("El índice de la fruta objetivo es inválido.")
    target_label = labels[target_index]
    target_probability = float(values[target_index])
    target_cv = _class_oof_metrics(per_class, target_label)
    is_correct = (
        predicted_index == target_index and target_probability >= threshold
    )
    session_accuracy = correct_attempts / attempts if attempts else 0.0
    return PredictionPanel(
        primary_text=(
            f"Confianza actual de {target_label}: {target_probability:.1%}"
        ),
        confidence_text=f"Umbral de deteccion: {threshold:.0%}",
        cv_text=(
            (
                f"OOF de {target_label}: P {target_cv[0]:.1%} | "
                f"R {target_cv[1]:.1%} | F1 {target_cv[2]:.1%}"
            )
            if show_cv_metrics and target_cv is not None
            else None
        ),
        status_text=(
            "Resultado: CORRECTO"
            if is_correct
            else "Resultado: NO DETECTADA / REPITE"
        ),
        session_text=(
            f"Exactitud de sesion {target_label}: "
            f"{correct_attempts}/{attempts} ({session_accuracy:.1%})"
        ),
        positive=is_correct,
        show_probability_bars=False,
        show_global_metrics=False,
    )


def load_ensemble(
    run_dir: Path,
    device: str,
    *,
    weights_only: bool = False,
    checkpoint_profile: str = "full",
) -> tuple[list[LoadedMember], dict[str, Any]]:
    ensemble_path = run_dir / "ensemble.json"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"No existe {ensemble_path}. Entrena todos los folds antes de ejecutar live."
        )
    ensemble = read_json(ensemble_path)
    expected_implementation = ensemble.get("implementation_version")
    if expected_implementation not in SUPPORTED_EXPERIMENT_IMPLEMENTATION_VERSIONS:
        raise RuntimeError(
            "El ensemble pertenece a otra versión de la implementación. "
            "Entrena en un --run-dir nuevo."
        )
    labels = ensemble.get("labels")
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise RuntimeError(f"{ensemble_path} no contiene una lista de clases válida.")
    expected_layout = feature_layout()
    if ensemble.get("feature_layout") != expected_layout:
        raise RuntimeError(
            "El ensemble no usa el feature layout v2 esperado. "
            "Regenera el dataset y entrena en un --run-dir nuevo."
        )
    expected_model_config = ensemble.get("model_config")
    if not isinstance(expected_model_config, dict):
        raise RuntimeError(f"{ensemble_path} no contiene model_config.")
    try:
        expected_input_dim = int(expected_model_config["input_dim"])
        expected_num_classes = int(expected_model_config["num_classes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{ensemble_path} contiene model_config inválido.") from exc
    if expected_input_dim != model_feature_dim():
        raise RuntimeError(
            "model_config.input_dim no coincide con el feature layout v2 "
            f"({expected_model_config.get('input_dim')} != {model_feature_dim()})."
        )
    if expected_num_classes != len(labels):
        raise RuntimeError(
            "model_config.num_classes no coincide con las etiquetas del ensemble."
        )
    expects_transfer = (
        expected_model_config.get("model_family") == ASL_TRANSFER_MODEL_FAMILY
    )
    family_implementation = (
        TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION
        if expects_transfer
        else EXPERIMENT_IMPLEMENTATION_VERSION
    )
    if expected_implementation != family_implementation:
        raise RuntimeError(
            "La familia del modelo y la versión de implementación del ensemble "
            "no coinciden. Entrena en un --run-dir nuevo."
        )
    expected_transfer_contract: str | None = None
    expected_transfer_layout: dict[str, Any] | None = None
    expected_pretraining_sha: str | None = None
    if expects_transfer:
        from .asl_transfer import (
            ASL_CITIZEN_CHECKPOINT_SHA256,
            ASL_GRAPH_CONTRACT,
        )

        expected_transfer_contract = POSE_HANDS_XY_LAYOUT_VERSION
        expected_transfer_layout = transfer_feature_layout()
        ensemble_pretraining = ensemble.get("pretraining")
        if (
            ensemble.get("transfer_contract") != expected_transfer_contract
            or ensemble.get("transfer_layout") != expected_transfer_layout
            or not isinstance(ensemble_pretraining, dict)
            or ensemble_pretraining.get("checkpoint_sha256")
            != ASL_CITIZEN_CHECKPOINT_SHA256
            or ensemble_pretraining.get("cache_layout_version")
            != expected_transfer_contract
            or ensemble_pretraining.get("input_contract") != ASL_GRAPH_CONTRACT
        ):
            raise RuntimeError(
                "El ensemble transfer no declara el contrato pose/ASL Citizen "
                "exacto. Entrena en un --run-dir nuevo."
            )
        expected_pretraining_sha = ASL_CITIZEN_CHECKPOINT_SHA256
    try:
        expected_sequence_length = int(ensemble["sequence_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{ensemble_path} no contiene sequence_length válido.") from exc
    if expected_sequence_length < 1:
        raise RuntimeError("sequence_length debe ser mayor que cero.")
    expected_protocol = ensemble.get("validation_protocol")
    expected_fingerprint = ensemble.get("run_fingerprint")
    expected_dataset_fingerprint = ensemble.get("dataset_content_fingerprint")
    checkpoint_entries = ensemble.get("checkpoints")
    if (
        not isinstance(checkpoint_entries, list)
        or not checkpoint_entries
        or any(not isinstance(item, str) or not item for item in checkpoint_entries)
        or len(checkpoint_entries) != len(set(checkpoint_entries))
    ):
        raise RuntimeError("El ensemble no contiene una lista de checkpoints válida.")
    if checkpoint_profile == "full":
        selected_entries = checkpoint_entries
    elif checkpoint_profile == "final":
        final_checkpoint = ensemble.get("final_checkpoint")
        if (
            not isinstance(final_checkpoint, str)
            or not final_checkpoint
            or final_checkpoint not in checkpoint_entries
        ):
            raise RuntimeError(
                "El ensemble no declara un final_checkpoint válido para el "
                "perfil reducido."
            )
        selected_entries = [final_checkpoint]
    else:
        raise ValueError(
            "checkpoint_profile debe ser 'full' o 'final'; "
            f"se recibió {checkpoint_profile!r}."
        )
    members: list[LoadedMember] = []
    for relative in selected_entries:
        checkpoint_path = run_dir / relative
        model, standardizer, payload = load_checkpoint(
            checkpoint_path,
            device,
            weights_only=weights_only,
            retain_training_state=False,
        )
        incompatibilities: list[str] = []
        raw_preprocessing = payload.get("preprocessing")
        if isinstance(raw_preprocessing, dict):
            preprocessing = raw_preprocessing
        else:
            preprocessing = {}
            incompatibilities.append("metadata de preprocesamiento")
        if payload.get("labels") != labels:
            incompatibilities.append("orden de clases")
        try:
            checkpoint_sequence_length = int(payload["sequence_length"])
        except (KeyError, TypeError, ValueError):
            checkpoint_sequence_length = -1
        if checkpoint_sequence_length != expected_sequence_length:
            incompatibilities.append("longitud de secuencia")
        if payload.get("model_config") != expected_model_config:
            incompatibilities.append("model_config")
        if asdict(model.config) != expected_model_config:
            incompatibilities.append("arquitectura cargada")
        if preprocessing.get("model_config") != expected_model_config:
            incompatibilities.append("model_config de preprocesamiento")
        if preprocessing.get("feature_layout") != expected_layout:
            incompatibilities.append("feature layout de preprocesamiento")
        if preprocessing.get("implementation_version") != expected_implementation:
            incompatibilities.append("versión de implementación")
        if expects_transfer:
            member_pretraining = preprocessing.get("pretraining")
            if not isinstance(model, ASLTransferFusion):
                incompatibilities.append("familia transfer")
            if preprocessing.get("transfer_contract") != expected_transfer_contract:
                incompatibilities.append("contrato pose transfer")
            if preprocessing.get("transfer_layout") != expected_transfer_layout:
                incompatibilities.append("layout pose transfer")
            if (
                not isinstance(member_pretraining, dict)
                or member_pretraining.get("sha256") != expected_pretraining_sha
                or member_pretraining.get("expected_sha256")
                != expected_pretraining_sha
                or member_pretraining.get("sha256_verified") is not True
                or member_pretraining.get("strict") is not True
                or member_pretraining.get("safe_weights_only") is not True
            ):
                incompatibilities.append("preentrenamiento ASL Citizen")
        elif isinstance(model, ASLTransferFusion):
            incompatibilities.append("familia del modelo")
        if standardizer.layout_version != expected_layout["version"]:
            incompatibilities.append("layout del scaler")
        if (
            standardizer.mean.shape != (model_feature_dim(),)
            or standardizer.std.shape != (model_feature_dim(),)
        ):
            incompatibilities.append("dimensión del scaler")
        if (
            expected_protocol is not None
            and preprocessing.get("validation_protocol") != expected_protocol
        ):
            incompatibilities.append("protocolo de validación")
        if (
            expected_fingerprint is not None
            and preprocessing.get("run_fingerprint") != expected_fingerprint
        ):
            incompatibilities.append("huella de ejecución")
        if (
            expected_dataset_fingerprint is not None
            and preprocessing.get("dataset_content_fingerprint")
            != expected_dataset_fingerprint
        ):
            incompatibilities.append("huella del dataset procesado")
        if incompatibilities:
            details = ", ".join(incompatibilities)
            raise RuntimeError(
                f"Checkpoint incompatible {checkpoint_path}: {details}."
            )
        members.append(LoadedMember(model, standardizer, payload))
    if not members:
        raise RuntimeError("El ensemble no contiene checkpoints.")
    # Metadatos de ejecución; no se escriben sobre ensemble.json. Las métricas
    # OOF describen el protocolo completo, no el refit final aislado.
    ensemble["_checkpoint_profile"] = checkpoint_profile
    ensemble["_cv_metrics_applicable"] = checkpoint_profile == "full"
    return members, ensemble


def ensemble_predict(
    members: list[LoadedMember],
    static_sequence: np.ndarray,
    device: str,
    timestamps: np.ndarray | None = None,
    pose_hands_xy: np.ndarray | None = None,
) -> np.ndarray:
    import torch

    sequence = np.asarray(static_sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != static_feature_dim():
        raise ValueError(
            "La secuencia en vivo debe tener forma "
            f"[tiempo, {static_feature_dim()}]."
        )
    if len(sequence) == 0:
        raise ValueError("La secuencia en vivo está vacía.")
    time_axis: np.ndarray | None = None
    if timestamps is not None:
        time_axis = np.asarray(timestamps, dtype=np.float64)
        if (
            time_axis.shape != (len(sequence),)
            or not np.all(np.isfinite(time_axis))
            or (len(time_axis) > 1 and np.any(np.diff(time_axis) <= 0.0))
        ):
            raise ValueError(
                "Los timestamps en vivo deben ser finitos, estrictamente crecientes "
                "y coincidir con la secuencia."
            )
        # Evita perder resolución al convertir un monotonic() grande a float32.
        time_axis = time_axis - time_axis[0]
    if not members:
        raise RuntimeError("No hay modelos para ejecutar la inferencia.")
    transfer_required = any(
        isinstance(member.model, ASLTransferFusion) for member in members
    )
    raw_pose_hands: np.ndarray | None = None
    if transfer_required:
        if pose_hands_xy is None:
            raise ValueError(
                "El ensemble transfer requiere pose_hands_xy de la misma secuencia."
            )
        raw_pose_hands = np.asarray(pose_hands_xy, dtype=np.float32)
        if (
            raw_pose_hands.shape
            != (len(sequence), POSE_HANDS_XY_POINT_COUNT, 2)
            or not np.all(np.isfinite(raw_pose_hands))
        ):
            raise ValueError(
                "pose_hands_xy debe tener forma "
                f"[tiempo, {POSE_HANDS_XY_POINT_COUNT}, 2] y ser finito."
            )

    views_by_length: dict[int, list[np.ndarray]] = {}
    graph_tensors_by_length: dict[int, torch.Tensor] = {}
    total_probability: Any = None
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device == "cuda",
    ):
        for member in members:
            sequence_length = int(member.payload["sequence_length"])
            if sequence_length not in views_by_length:
                views_by_length[sequence_length] = deterministic_views(
                    sequence,
                    sequence_length,
                    timestamps=time_axis,
                )
            views = views_by_length[sequence_length]
            model_inputs = np.stack(
                [
                    member.standardizer.transform(add_temporal_deltas(view))
                    for view in views
                ]
            )
            tensor = torch.from_numpy(model_inputs).to(
                device,
                non_blocking=True,
            )
            if isinstance(member.model, ASLTransferFusion):
                from .asl_transfer import prepare_asl_graph

                graph_length = member.model.config.graph_sequence_length
                if graph_length not in graph_tensors_by_length:
                    assert raw_pose_hands is not None
                    graph_tensors_by_length[graph_length] = torch.from_numpy(
                        np.stack(
                            [
                                prepare_asl_graph(
                                    raw_pose_hands,
                                    static_sequence=sequence,
                                    timestamps=time_axis,
                                    phase_exponent=exponent,
                                )
                                for exponent in TEMPORAL_VIEW_EXPONENTS
                            ]
                        )
                    ).to(device, non_blocking=True)
                graph_tensor = graph_tensors_by_length[graph_length]
                logits = member.model(tensor, graph_tensor)
            else:
                logits = member.model(tensor)
            member_probability = torch.softmax(
                logits.float(),
                dim=-1,
            ).mean(dim=0)
            total_probability = (
                member_probability
                if total_probability is None
                else total_probability + member_probability
            )
    return (total_probability / len(members)).cpu().numpy()


def _put(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        frame,
        ascii_text(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


@dataclass(frozen=True)
class OverlayRect:
    """Rectángulo de pantalla para el HUD de inferencia en vivo."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class LiveOverlayLayout:
    """Geometría adaptativa del HUD, independiente de OpenCV."""

    padding: int
    header: OverlayRect
    card: OverlayRect
    metrics: OverlayRect | None
    toast: OverlayRect | None
    line_height: int
    bar_height: int
    probability_bar_count: int
    global_metrics_in_card: bool


def build_live_overlay_layout(
    frame_width: int,
    frame_height: int,
    *,
    prediction_ready: bool,
    target_mode: bool,
    show_probability_bars: bool,
    show_global_metrics: bool,
) -> LiveOverlayLayout:
    """Devuelve un HUD compacto que deja libre la zona central de cámara.

    La cabecera ocupa sólo una franja superior. El resultado se ancla abajo a
    la izquierda y, cuando cabe, la exactitud OOF aparece en una tarjeta aparte
    arriba a la derecha. En pantallas bajas se integra dentro de la tarjeta en
    vez de superponer elementos sobre la imagen de la persona.
    """

    if (
        isinstance(frame_width, bool)
        or isinstance(frame_height, bool)
        or not isinstance(frame_width, (int, np.integer))
        or not isinstance(frame_height, (int, np.integer))
        or frame_width < 160
        or frame_height < 160
    ):
        raise ValueError(
            "El HUD requiere dimensiones enteras de al menos 160x160 píxeles."
        )
    if target_mode and (show_probability_bars or show_global_metrics):
        raise ValueError(
            "El modo objetivo no puede mostrar barras ni métricas globales."
        )
    if show_probability_bars and not prediction_ready:
        raise ValueError("Las barras de probabilidad requieren una predicción.")

    width = int(frame_width)
    height = int(frame_height)
    padding = int(np.clip(round(min(width, height) * 0.018), 8, 18))
    # En resoluciones muy compactas las barras ocuparían el torso de la persona.
    # Se conservan predicción y exactitud, y el top-k vuelve a aparecer al haber
    # espacio suficiente para no invadir el centro de la cámara.
    probability_bars_visible = (
        show_probability_bars and width >= 480 and height >= 360
    )
    header_height = min(
        max(42, int(round(height * 0.075))),
        min(62, height - 3 * padding),
    )
    header = OverlayRect(0, 0, width, header_height)
    inner_width = width - 2 * padding
    card_width = min(
        inner_width,
        int(
            np.clip(
                round(width * (0.64 if target_mode else 0.40)),
                280,
                560 if target_mode else 470,
            )
        ),
    )
    line_height = int(np.clip(round(height * 0.040), 22, 30))
    bar_height = int(np.clip(round(height * 0.037), 21, 28))
    available_card_height = height - header_height - 2 * padding

    # Una predicción general necesita tres líneas antes de las barras. El modo
    # objetivo reserva sus cinco líneas (confianza, umbral, OOF, estado y
    # exactitud de sesión), incluso si alguna métrica OOF no está disponible.
    base_lines = 1 if not prediction_ready else (5 if target_mode else 3)

    def card_dimensions(metrics_inside: bool) -> tuple[int, int]:
        text_lines = base_lines + int(metrics_inside)
        text_height = 2 * padding + text_lines * line_height
        if not probability_bars_visible:
            return min(available_card_height, max(72, text_height)), 0
        available_for_bars = max(0, available_card_height - text_height)
        bar_count = min(5, available_for_bars // bar_height)
        card_height = text_height + bar_count * bar_height
        return min(available_card_height, max(84, card_height)), int(bar_count)

    initial_height, _ = card_dimensions(metrics_inside=False)
    initial_top = height - padding - initial_height
    metrics_height = max(40, line_height + 18)
    metrics_top = header.bottom + padding
    # Si la tarjeta del resultado empieza demasiado arriba, la métrica se
    # vuelve una línea interna. Así no cubrimos el torso/rostro con dos paneles.
    global_metrics_in_card = bool(
        show_global_metrics
        and initial_top <= metrics_top + metrics_height + padding
    )
    card_height, bar_count = card_dimensions(global_metrics_in_card)
    card = OverlayRect(
        padding,
        height - padding - card_height,
        card_width,
        card_height,
    )

    metrics: OverlayRect | None = None
    if show_global_metrics and not global_metrics_in_card:
        metrics_width = min(
            inner_width,
            int(np.clip(round(width * 0.42), 300, 560)),
        )
        metrics = OverlayRect(
            width - padding - metrics_width,
            metrics_top,
            metrics_width,
            metrics_height,
        )

    toast_width = min(
        inner_width,
        int(np.clip(round(width * 0.34), 250, 390)),
    )
    toast_height = max(36, line_height + 14)
    toast_top = (metrics.bottom + padding) if metrics is not None else metrics_top
    toast_candidate = OverlayRect(
        width - padding - toast_width,
        toast_top,
        toast_width,
        toast_height,
    )
    toast = (
        toast_candidate
        if toast_candidate.bottom + padding <= card.y
        else None
    )

    return LiveOverlayLayout(
        padding=padding,
        header=header,
        card=card,
        metrics=metrics,
        toast=toast,
        line_height=line_height,
        bar_height=bar_height,
        probability_bar_count=bar_count,
        global_metrics_in_card=global_metrics_in_card,
    )


def _draw_translucent_rect(
    frame: np.ndarray,
    rect: OverlayRect,
    color: tuple[int, int, int] = (12, 18, 28),
    alpha: float = 0.76,
) -> None:
    """Pinta sólo el rectángulo necesario, no una copia opaca de toda la vista."""

    import cv2

    roi = frame[rect.y : rect.bottom, rect.x : rect.right]
    overlay = np.empty_like(roi)
    overlay[...] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)


def _put_fit(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    max_width: int,
    scale: float = 0.65,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
    minimum_scale: float = 0.36,
) -> None:
    """Dibuja texto legible dentro de una tarjeta, reduciendo o truncando."""

    import cv2

    if max_width <= 0:
        return
    rendered = ascii_text(text)
    fitted_scale = scale
    while fitted_scale > minimum_scale:
        text_width = cv2.getTextSize(
            rendered,
            cv2.FONT_HERSHEY_SIMPLEX,
            fitted_scale,
            thickness,
        )[0][0]
        if text_width <= max_width:
            break
        fitted_scale = max(minimum_scale, fitted_scale - 0.04)
    text_width = cv2.getTextSize(
        rendered,
        cv2.FONT_HERSHEY_SIMPLEX,
        fitted_scale,
        thickness,
    )[0][0]
    if text_width > max_width and len(rendered) > 3:
        suffix = "..."
        low, high = 0, len(rendered)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = rendered[:middle].rstrip() + suffix
            candidate_width = cv2.getTextSize(
                candidate,
                cv2.FONT_HERSHEY_SIMPLEX,
                fitted_scale,
                thickness,
            )[0][0]
            if candidate_width <= max_width:
                low = middle
            else:
                high = middle - 1
        rendered = rendered[:low].rstrip() + suffix if low else suffix
    _put(frame, rendered, position, fitted_scale, color, thickness)


def _draw_probability_bars(
    frame: np.ndarray,
    labels: list[str],
    probabilities: np.ndarray,
    left: int,
    top: int,
    width: int,
    count: int = 5,
    target_index: int | None = None,
    row_height: int = 31,
) -> None:
    import cv2

    if count <= 0 or width <= 0:
        return
    order = list(np.argsort(probabilities)[::-1][:count])
    if target_index is not None and target_index not in order:
        order[-1] = target_index
    row_height = max(20, row_height)
    label_width = max(90, int(width * 0.48))
    bar_left = left + label_width
    bar_width = max(24, width - label_width - 5)
    bar_top_offset = max(3, row_height // 6)
    bar_bottom_offset = min(row_height - 3, bar_top_offset + 16)
    for row, index in enumerate(order):
        y = top + row * row_height
        probability = float(probabilities[index])
        _put_fit(
            frame,
            f"{labels[index]} {probability:.1%}",
            (left, y + min(row_height - 6, 17)),
            label_width - 6,
            min(0.52, max(0.40, row_height / 55.0)),
        )
        color = (70, 200, 90) if index == int(np.argmax(probabilities)) else (60, 150, 230)
        if target_index is not None and index == target_index:
            color = (60, 210, 240)
        cv2.rectangle(
            frame,
            (bar_left, y + bar_top_offset),
            (bar_left + bar_width, y + bar_bottom_offset),
            (55, 55, 55),
            -1,
        )
        cv2.rectangle(
            frame,
            (bar_left, y + bar_top_offset),
            (bar_left + int(bar_width * probability), y + bar_bottom_offset),
            color,
            -1,
        )


def build_jetson_csi_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """Construye un pipeline CSI fijo sin interpolar texto arbitrario.

    ``nvarguscamerasrc`` es el origen de cámaras CSI en Jetson. Exigir enteros
    acotados mantiene este helper libre de fragmentos GStreamer aportados por el
    usuario; ``run_live`` también admite un pipeline explícito para integraciones
    avanzadas.
    """

    parameters = {
        "sensor_id": (sensor_id, 0, 15),
        "width": (width, 1, 8192),
        "height": (height, 1, 8192),
        "fps": (fps, 1, 240),
        "flip_method": (flip_method, 0, 7),
    }
    for name, (value, minimum, maximum) in parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"{name} debe ser un entero entre {minimum} y {maximum}."
            )
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def run_live(
    run_dir: Path,
    fruit: str | None = None,
    camera: int = 0,
    device_request: str = "cuda",
    confidence_threshold: float | None = None,
    smoothing: float = 0.30,
    mirror_display: bool = True,
    prediction_fps: float | None = None,
    camera_pipeline: str | None = None,
    width: int = 1280,
    height: int = 720,
    checkpoint_profile: str = "full",
    weights_only: bool = False,
) -> None:
    import cv2

    if not 0.0 < smoothing <= 1.0:
        raise ValueError("--smoothing debe estar en el intervalo (0, 1].")
    if prediction_fps is not None and (
        not np.isfinite(prediction_fps) or prediction_fps <= 0.0
    ):
        raise ValueError("--prediction-fps debe ser finito y mayor que cero.")
    if width < 1 or height < 1:
        raise ValueError("El ancho y alto de cámara deben ser mayores que cero.")
    if camera_pipeline is not None and not camera_pipeline.strip():
        raise ValueError("camera_pipeline no puede estar vacío.")
    device = resolve_device(device_request)
    members, ensemble = load_ensemble(
        run_dir,
        device,
        weights_only=weights_only,
        checkpoint_profile=checkpoint_profile,
    )
    if device == "cuda":
        import torch

        inference_device = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        inference_device = device
    labels: list[str] = ensemble["labels"]
    target_label = resolve_label(fruit, labels) if fruit else None
    target_index = labels.index(target_label) if target_label else None
    threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else float(ensemble.get("recommended_confidence_threshold", 0.55))
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold debe estar entre 0 y 1.")
    cv_metrics_applicable = bool(ensemble.get("_cv_metrics_applicable", True))
    cv_metrics_path = run_dir / ensemble.get("cv_metrics", "cv_metrics.json")
    raw_cv_metrics: Any = (
        read_json(cv_metrics_path)
        if cv_metrics_applicable and cv_metrics_path.is_file()
        else {}
    )
    cv_metrics = raw_cv_metrics if isinstance(raw_cv_metrics, dict) else {}
    global_accuracy = _unit_fraction(cv_metrics.get("accuracy"))
    macro_precision = _unit_fraction(cv_metrics.get("macro_precision"))
    raw_per_class = cv_metrics.get("per_class")
    per_class = raw_per_class if isinstance(raw_per_class, dict) else {}
    global_oof_metrics_available = (
        cv_metrics_applicable
        and global_accuracy is not None
        and macro_precision is not None
    )
    per_class_oof_available = cv_metrics_applicable and bool(per_class)
    if cv_metrics_applicable and not global_oof_metrics_available:
        print(
            "Advertencia: no se encontraron métricas OOF globales válidas; "
            "la interfaz no mostrará una exactitud de 0 % ficticia."
        )

    sequence_length = int(ensemble["sequence_length"])
    target_fps = float(ensemble["target_fps"])
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise RuntimeError("El ensemble contiene target_fps inválido.")
    max_side = int(ensemble.get("max_side", 960))
    # Cinco segundos conservan el mismo rango temporal que los videos raw. Las
    # vistas por fase lo reducen después a sequence_length para el clasificador.
    buffer_seconds = 5.0
    buffer_capacity = max(
        sequence_length,
        int(np.ceil(target_fps * buffer_seconds)) + 1,
    )
    feature_buffer: deque[np.ndarray] = deque(maxlen=buffer_capacity)
    pose_hands_buffer: deque[np.ndarray] = deque(maxlen=buffer_capacity)
    timestamp_buffer: deque[float] = deque(maxlen=buffer_capacity)
    capture_source = f"cámara {camera}"
    if camera_pipeline is None:
        capture = cv2.VideoCapture(camera)
    else:
        capture_source = "pipeline GStreamer"
        capture = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir {capture_source}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    smoothed: np.ndarray | None = None
    last_sample = 0.0
    last_prediction = 0.0
    attempts = 0
    correct_attempts = 0
    last_message = ""
    last_message_until = 0.0
    prediction_ready = False
    transfer_required = any(
        isinstance(member.model, ASLTransferFusion) for member in members
    )
    title = "PFrutas - reconocimiento de senas"
    print(f"Backend del clasificador: {inference_device} ({device})")
    print(
        f"Perfil de checkpoints: {checkpoint_profile} "
        f"({len(members)} miembro(s))"
    )
    print("Controles: Q/Esc salir | R reiniciar | Espacio registrar intento y limpiar buffer")

    try:
        with HolisticFeatureExtractor() as extractor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Se perdió la señal de la cámara.")
                now = time.monotonic()
                if now - last_sample >= 1.0 / target_fps:
                    last_sample = now
                    frame_height, frame_width = frame.shape[:2]
                    largest = max(frame_height, frame_width)
                    inference_frame = frame
                    if largest > max_side:
                        scale = max_side / largest
                        inference_frame = cv2.resize(
                            frame,
                            (
                                int(round(frame_width * scale)),
                                int(round(frame_height * scale)),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )
                    rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    if transfer_required:
                        vector, _, pose_hands_xy = extractor.process_multimodal(rgb)
                        pose_hands_buffer.append(pose_hands_xy)
                    else:
                        vector, _ = extractor.process_rgb(rgb)
                    feature_buffer.append(vector)
                    timestamp_buffer.append(now)
                    while (
                        len(timestamp_buffer) > 1
                        and timestamp_buffer[-1] - timestamp_buffer[0]
                        > buffer_seconds
                    ):
                        feature_buffer.popleft()
                        if transfer_required:
                            pose_hands_buffer.popleft()
                        timestamp_buffer.popleft()
                    minimum = max(12, int(sequence_length * 0.75))
                    prediction_due = (
                        prediction_fps is None
                        or now - last_prediction >= 1.0 / prediction_fps
                    )
                    if len(feature_buffer) >= minimum and prediction_due:
                        last_prediction = now
                        probability = ensemble_predict(
                            members,
                            np.stack(feature_buffer),
                            device,
                            timestamps=np.asarray(timestamp_buffer, dtype=np.float64),
                            pose_hands_xy=(
                                np.stack(pose_hands_buffer)
                                if transfer_required
                                else None
                            ),
                        )
                        smoothed = (
                            probability
                            if smoothed is None
                            else smoothing * probability + (1.0 - smoothing) * smoothed
                        )
                        smoothed = smoothed / smoothed.sum()
                        prediction_ready = True

                display = cv2.flip(frame, 1) if mirror_display else frame.copy()
                panel: PredictionPanel | None = None
                if prediction_ready and smoothed is not None:
                    panel = build_prediction_panel(
                        labels=labels,
                        probabilities=smoothed,
                        threshold=threshold,
                        per_class=per_class,
                        target_index=target_index,
                        attempts=attempts,
                        correct_attempts=correct_attempts,
                        show_cv_metrics=per_class_oof_available,
                        show_global_metrics=global_oof_metrics_available,
                    )

                show_probability_bars = bool(
                    panel is not None and panel.show_probability_bars
                )
                show_global_metrics = (
                    panel.show_global_metrics
                    if panel is not None
                    else target_index is None and global_oof_metrics_available
                )
                layout = build_live_overlay_layout(
                    display.shape[1],
                    display.shape[0],
                    prediction_ready=panel is not None,
                    target_mode=target_index is not None,
                    show_probability_bars=show_probability_bars,
                    show_global_metrics=show_global_metrics,
                )

                # El HUD se limita a una cabecera y tarjetas en las esquinas;
                # la zona central queda despejada para que la persona se vea y
                # pueda verificar el encuadre de la seña.
                _draw_translucent_rect(display, layout.header, alpha=0.56)
                cv2.rectangle(
                    display,
                    (layout.header.x, layout.header.y),
                    (layout.header.right - 1, layout.header.bottom - 1),
                    (64, 87, 110),
                    1,
                )
                mode = f"OBJETIVO: {target_label}" if target_label else "GENERAL"
                left_header_width = max(
                    80,
                    int(display.shape[1] * 0.46) - 2 * layout.padding,
                )
                header_first_line = max(20, layout.header.height // 2)
                header_second_line = layout.header.bottom - max(6, layout.padding // 2)
                _put_fit(
                    display,
                    f"Modo: {mode}",
                    (layout.padding, header_first_line),
                    left_header_width,
                    0.66,
                    (90, 220, 255),
                    2,
                )
                _put_fit(
                    display,
                    f"Buffer: {len(feature_buffer)}/{buffer_capacity} | "
                    f"Modelo: {device.upper()} / {checkpoint_profile}",
                    (layout.padding, header_second_line),
                    left_header_width,
                    0.50,
                    (190, 195, 205),
                )
                controls_x = int(display.shape[1] * 0.49)
                _put_fit(
                    display,
                    "ESPACIO: intento | R: reiniciar | Q: salir",
                    (controls_x, header_first_line + 5),
                    display.shape[1] - controls_x - layout.padding,
                    0.50,
                    (175, 180, 190),
                )

                _draw_translucent_rect(display, layout.card, alpha=0.68)
                cv2.rectangle(
                    display,
                    (layout.card.x, layout.card.y),
                    (layout.card.right - 1, layout.card.bottom - 1),
                    (64, 87, 110),
                    1,
                )
                card_text_x = layout.card.x + layout.padding
                card_text_width = layout.card.width - 2 * layout.padding
                if panel is None:
                    card_lines: list[
                        tuple[str | None, float, tuple[int, int, int], int]
                    ] = [
                        (
                            "Realiza una sena para llenar el buffer...",
                            0.62,
                            (100, 200, 255),
                            1,
                        )
                    ]
                elif target_index is None:
                    prediction_color = (
                        (80, 225, 100) if panel.positive else (80, 175, 255)
                    )
                    card_lines = [
                        (panel.primary_text, 0.78, prediction_color, 2),
                        (panel.confidence_text, 0.56, (245, 245, 245), 1),
                        (panel.cv_text, 0.52, (205, 210, 220), 1),
                    ]
                else:
                    result_color = (
                        (80, 230, 105) if panel.positive else (70, 110, 255)
                    )
                    card_lines = [
                        (panel.primary_text, 0.70, (245, 245, 245), 2),
                        (panel.confidence_text, 0.56, (245, 245, 245), 1),
                        (panel.cv_text, 0.50, (205, 210, 220), 1),
                        (panel.status_text, 0.66, result_color, 2),
                        (panel.session_text, 0.54, (240, 215, 110), 1),
                    ]

                for line_index, (text, scale, color, thickness) in enumerate(card_lines):
                    if text is None:
                        continue
                    line_y = (
                        layout.card.y
                        + layout.padding
                        + (line_index + 1) * layout.line_height
                        - 5
                    )
                    _put_fit(
                        display,
                        text,
                        (card_text_x, line_y),
                        card_text_width,
                        scale,
                        color,
                        thickness,
                    )

                if layout.global_metrics_in_card:
                    assert global_accuracy is not None
                    assert macro_precision is not None
                    metric_line_y = (
                        layout.card.y
                        + layout.padding
                        + (len(card_lines) + 1) * layout.line_height
                        - 5
                    )
                    _put_fit(
                        display,
                        build_general_oof_metrics_text(
                            global_accuracy,
                            macro_precision,
                        ),
                        (card_text_x, metric_line_y),
                        card_text_width,
                        0.50,
                        (180, 190, 205),
                    )

                if (
                    panel is not None
                    and panel.show_probability_bars
                    and smoothed is not None
                    and layout.probability_bar_count > 0
                ):
                    bar_top = (
                        layout.card.y
                        + layout.padding
                        + (
                            len(card_lines) + int(layout.global_metrics_in_card)
                        )
                        * layout.line_height
                    )
                    _draw_probability_bars(
                        display,
                        labels,
                        smoothed,
                        left=card_text_x,
                        top=bar_top,
                        width=card_text_width,
                        count=layout.probability_bar_count,
                        row_height=layout.bar_height,
                    )

                if layout.metrics is not None:
                    assert global_accuracy is not None
                    assert macro_precision is not None
                    _draw_translucent_rect(display, layout.metrics, alpha=0.64)
                    cv2.rectangle(
                        display,
                        (layout.metrics.x, layout.metrics.y),
                        (layout.metrics.right - 1, layout.metrics.bottom - 1),
                        (64, 87, 110),
                        1,
                    )
                    _put_fit(
                        display,
                        build_general_oof_metrics_text(
                            global_accuracy,
                            macro_precision,
                        ),
                        (
                            layout.metrics.x + layout.padding,
                            layout.metrics.y + layout.metrics.height - 13,
                        ),
                        layout.metrics.width - 2 * layout.padding,
                        0.52,
                        (180, 210, 230),
                    )

                if last_message and now < last_message_until:
                    if layout.toast is not None:
                        _draw_translucent_rect(display, layout.toast, alpha=0.65)
                        cv2.rectangle(
                            display,
                            (layout.toast.x, layout.toast.y),
                            (layout.toast.right - 1, layout.toast.bottom - 1),
                            (80, 150, 190),
                            1,
                        )
                        _put_fit(
                            display,
                            last_message,
                            (
                                layout.toast.x + layout.padding,
                                layout.toast.y + layout.toast.height - 12,
                            ),
                            layout.toast.width - 2 * layout.padding,
                            0.58,
                            (70, 230, 255),
                            2,
                        )
                    else:
                        _put_fit(
                            display,
                            last_message,
                            (controls_x, header_second_line),
                            display.shape[1] - controls_x - layout.padding,
                            0.50,
                            (70, 230, 255),
                            2,
                        )

                cv2.imshow(title, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    feature_buffer.clear()
                    pose_hands_buffer.clear()
                    timestamp_buffer.clear()
                    smoothed = None
                    prediction_ready = False
                    attempts = 0
                    correct_attempts = 0
                    last_message = "Sesion reiniciada"
                    last_message_until = now + 1.5
                if key == 32:
                    if not prediction_ready or smoothed is None:
                        last_message = "Espera a que el buffer este listo"
                    elif target_index is None:
                        predicted_index = int(np.argmax(smoothed))
                        last_message = (
                            f"Registrado: {labels[predicted_index]} "
                            f"({float(smoothed[predicted_index]):.1%})"
                        )
                        feature_buffer.clear()
                        pose_hands_buffer.clear()
                        timestamp_buffer.clear()
                        smoothed = None
                        prediction_ready = False
                    else:
                        predicted_index = int(np.argmax(smoothed))
                        confidence = float(smoothed[predicted_index])
                        attempts += 1
                        if predicted_index == target_index and confidence >= threshold:
                            correct_attempts += 1
                            last_message = "Intento CORRECTO"
                        else:
                            last_message = "Objetivo no detectado; repite el intento"
                        feature_buffer.clear()
                        pose_hands_buffer.clear()
                        timestamp_buffer.clear()
                        smoothed = None
                        prediction_ready = False
                    last_message_until = now + 2.0
    finally:
        capture.release()
        cv2.destroyAllWindows()
