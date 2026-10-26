from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Tronco, cabeza y brazos. Las manos detalladas se incorporan por separado.
POSE_INDICES = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)

# Contornos compactos de labios, ojos y cejas para conservar señales no manuales
# sin introducir los 478 puntos completos de Face Mesh en un dataset pequeño.
FACE_INDICES = tuple(
    sorted(
        {
            10,
            13,
            14,
            17,
            33,
            37,
            39,
            40,
            46,
            52,
            53,
            55,
            61,
            63,
            65,
            66,
            70,
            78,
            81,
            82,
            84,
            87,
            88,
            91,
            95,
            105,
            107,
            133,
            145,
            146,
            159,
            160,
            161,
            163,
            181,
            185,
            191,
            246,
            249,
            263,
            267,
            269,
            270,
            276,
            282,
            283,
            285,
            291,
            293,
            295,
            296,
            300,
            308,
            311,
            312,
            314,
            317,
            318,
            321,
            324,
            334,
            336,
            362,
            374,
            375,
            386,
            387,
            388,
            390,
            405,
            409,
            415,
            466,
        }
    )
)

SCHEMA_VERSION = 2
FEATURE_LAYOUT_VERSION = "mask_aware_motion_v2"
MAX_INTERPOLATION_GAP = 3
PRESENCE_DIM = 4
SEGMENT_PADDING_SECONDS = 0.25
SEGMENT_MINIMUM_SECONDS = 1.0
SEGMENT_ENERGY_THRESHOLD = 0.20
SEGMENT_CLOSE_GAP_FRAMES = 3
TEMPORAL_VIEW_EXPONENTS = (0.90, 1.0, 1.10)
TRAIN_SPEED_WARP_RANGE = (0.80, 1.25)
TRAIN_MIN_CROP_FRACTION = 0.88
TRAIN_MIRROR_PROBABILITY = 0.50
TRAIN_ROTATION_DEGREES = 7.0
TRAIN_SCALE_RANGE = (0.90, 1.10)
TRAIN_NOISE_STD = 0.004
STANDARDIZER_CLIP = 6.0

# Representación cruda utilizada por el adaptador de transferencia ASL.  El
# orden coincide con el contrato pose_hands_xy_v1 del manifiesto: pose
# completa, mano derecha y mano izquierda.
POSE_HANDS_XY_LAYOUT_VERSION = "pose_hands_xy_v1"
POSE_HANDS_XY_POINT_COUNT = 33 + 21 + 21

POSE_XYZ_DIM = len(POSE_INDICES) * 3
HAND_REPRESENTATION_DIM = 21 * 3 * 2
HAND_COORDINATE_DIM = 21 * 3
FACE_XYZ_DIM = len(FACE_INDICES) * 3
GEOMETRY_DIM = POSE_XYZ_DIM + 2 * HAND_REPRESENTATION_DIM + FACE_XYZ_DIM
POSE_VISIBILITY_DIM = len(POSE_INDICES)

POSE_XYZ_SLICE = slice(0, POSE_XYZ_DIM)
LEFT_HAND_SLICE = slice(POSE_XYZ_SLICE.stop, POSE_XYZ_SLICE.stop + HAND_REPRESENTATION_DIM)
RIGHT_HAND_SLICE = slice(LEFT_HAND_SLICE.stop, LEFT_HAND_SLICE.stop + HAND_REPRESENTATION_DIM)
FACE_XYZ_SLICE = slice(RIGHT_HAND_SLICE.stop, RIGHT_HAND_SLICE.stop + FACE_XYZ_DIM)
POSE_VISIBILITY_SLICE = slice(GEOMETRY_DIM, GEOMETRY_DIM + POSE_VISIBILITY_DIM)
PRESENCE_SLICE = slice(POSE_VISIBILITY_SLICE.stop, POSE_VISIBILITY_SLICE.stop + PRESENCE_DIM)

POSE_PRESENCE = 0
FACE_PRESENCE = 1
LEFT_HAND_PRESENCE = 2
RIGHT_HAND_PRESENCE = 3

MODALITY_GEOMETRY: tuple[tuple[slice, int], ...] = (
    (POSE_XYZ_SLICE, POSE_PRESENCE),
    (LEFT_HAND_SLICE, LEFT_HAND_PRESENCE),
    (RIGHT_HAND_SLICE, RIGHT_HAND_PRESENCE),
    (FACE_XYZ_SLICE, FACE_PRESENCE),
)

LEFT_HAND_GLOBAL_SLICE = slice(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.start + HAND_COORDINATE_DIM)
LEFT_HAND_LOCAL_SLICE = slice(LEFT_HAND_GLOBAL_SLICE.stop, LEFT_HAND_SLICE.stop)
RIGHT_HAND_GLOBAL_SLICE = slice(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.start + HAND_COORDINATE_DIM)
RIGHT_HAND_LOCAL_SLICE = slice(RIGHT_HAND_GLOBAL_SLICE.stop, RIGHT_HAND_SLICE.stop)

_POSE_MIRROR_PAIRS = {
    7: 8,
    8: 7,
    9: 10,
    10: 9,
    11: 12,
    12: 11,
    13: 14,
    14: 13,
    15: 16,
    16: 15,
    23: 24,
    24: 23,
}
POSE_MIRROR_PERMUTATION = tuple(
    POSE_INDICES.index(_POSE_MIRROR_PAIRS.get(index, index))
    for index in POSE_INDICES
)


def static_feature_dim() -> int:
    return GEOMETRY_DIM + POSE_VISIBILITY_DIM + PRESENCE_DIM


def model_feature_dim() -> int:
    # Geometría/auxiliares estáticos + velocidad válida de la geometría.
    return static_feature_dim() + GEOMETRY_DIM


def feature_layout() -> dict[str, Any]:
    """Contrato serializable utilizado por manifiestos y fingerprints."""
    return {
        "version": FEATURE_LAYOUT_VERSION,
        "static_dim": static_feature_dim(),
        "geometry_dim": GEOMETRY_DIM,
        "model_dim": model_feature_dim(),
        "pose_xyz": [POSE_XYZ_SLICE.start, POSE_XYZ_SLICE.stop],
        "left_hand": [LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop],
        "right_hand": [RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop],
        "face_xyz": [FACE_XYZ_SLICE.start, FACE_XYZ_SLICE.stop],
        "pose_visibility": [POSE_VISIBILITY_SLICE.start, POSE_VISIBILITY_SLICE.stop],
        "presence": [PRESENCE_SLICE.start, PRESENCE_SLICE.stop],
        "max_interpolation_gap_frames": MAX_INTERPOLATION_GAP,
        "delta_definition": "geometry diff only when both adjacent frames are valid",
        "segment": {
            "energy_threshold": SEGMENT_ENERGY_THRESHOLD,
            "close_gap_frames": SEGMENT_CLOSE_GAP_FRAMES,
            "padding_seconds": SEGMENT_PADDING_SECONDS,
            "minimum_seconds": SEGMENT_MINIMUM_SECONDS,
        },
        "temporal_views": list(TEMPORAL_VIEW_EXPONENTS),
        "training_augmentation": {
            "minimum_crop_fraction": TRAIN_MIN_CROP_FRACTION,
            "speed_warp_range": list(TRAIN_SPEED_WARP_RANGE),
            "mirror_probability": TRAIN_MIRROR_PROBABILITY,
            "rotation_degrees": TRAIN_ROTATION_DEGREES,
            "scale_range": list(TRAIN_SCALE_RANGE),
            "noise_std": TRAIN_NOISE_STD,
        },
        "standardization": {
            "valid_groups_only": True,
            "clip": [-STANDARDIZER_CLIP, STANDARDIZER_CLIP],
        },
    }


def _rotate_xy(points: np.ndarray, angle: float) -> np.ndarray:
    result = points.copy()
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    x = result[..., 0].copy()
    y = result[..., 1].copy()
    result[..., 0] = cosine * x - sine * y
    result[..., 1] = sine * x + cosine * y
    return result


def _landmarks_to_array(landmark_list: Any, expected: int) -> np.ndarray | None:
    if landmark_list is None:
        return None
    landmarks = landmark_list.landmark
    if len(landmarks) < expected:
        return None
    return np.asarray([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)


def _pose_to_array(landmark_list: Any) -> np.ndarray | None:
    if landmark_list is None:
        return None
    landmarks = landmark_list.landmark
    if len(landmarks) < 33:
        return None
    return np.asarray(
        [(p.x, p.y, p.z, p.visibility) for p in landmarks],
        dtype=np.float32,
    )


class HolisticFeatureExtractor:
    """Extrae rasgos geométricos invariantes a posición y escala del cuerpo."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticFeatureExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def process_rgb(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Conserva el contrato v2 histórico usando una sola pasada Holistic."""
        features, quality, _ = self.process_multimodal(rgb_frame)
        return features, quality

    def process_multimodal(
        self,
        rgb_frame: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        """Extrae simultáneamente features v2 y pose/manos crudas para transfer.

        ``pose_hands_xy`` contiene coordenadas MediaPipe normalizadas respecto a
        la imagen, sin la normalización corporal de las features v2. Su forma es
        ``[75, 2]`` y el orden es pose33 + right_hand21 + left_hand21. Una
        modalidad no detectada ocupa ceros, de modo que una segunda ejecución
        de MediaPipe no es necesaria.
        """
        results = self._holistic.process(rgb_frame)
        pose = _pose_to_array(results.pose_landmarks)
        left_hand = _landmarks_to_array(results.left_hand_landmarks, 21)
        right_hand = _landmarks_to_array(results.right_hand_landmarks, 21)
        face = _landmarks_to_array(results.face_landmarks, 468)

        pose_hands_xy = np.zeros(
            (POSE_HANDS_XY_POINT_COUNT, 2),
            dtype=np.float32,
        )
        if pose is not None:
            pose_hands_xy[:33] = pose[:, :2]
        if right_hand is not None:
            pose_hands_xy[33:54] = right_hand[:, :2]
        if left_hand is not None:
            pose_hands_xy[54:75] = left_hand[:, :2]
        # MediaPipe normalmente entrega valores finitos. Este saneamiento evita
        # que un frame excepcional inutilice el cache completo.
        np.nan_to_num(
            pose_hands_xy,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        body_angle = 0.0
        if pose is not None and min(pose[11, 3], pose[12, 3]) > 0.25:
            body_origin = (pose[11, :3] + pose[12, :3]) / 2.0
            shoulder_axis = pose[12, :2] - pose[11, :2]
            body_scale = float(np.linalg.norm(shoulder_axis))
            body_scale = max(body_scale, 0.08)
            body_angle = -float(np.arctan2(shoulder_axis[1], shoulder_axis[0]))
        else:
            body_origin = np.asarray((0.5, 0.5, 0.0), dtype=np.float32)
            body_scale = 0.25

        pose_xyz = np.zeros((len(POSE_INDICES), 3), dtype=np.float32)
        pose_visibility = np.zeros(len(POSE_INDICES), dtype=np.float32)
        if pose is not None:
            selected = pose[list(POSE_INDICES)].copy()
            pose_xyz = _rotate_xy(
                (selected[:, :3] - body_origin) / body_scale,
                body_angle,
            )
            pose_visibility = selected[:, 3]

        hand_features: list[np.ndarray] = []
        for hand in (left_hand, right_hand):
            if hand is None:
                hand_features.extend(
                    (
                        np.zeros((21, 3), dtype=np.float32),
                        np.zeros((21, 3), dtype=np.float32),
                    )
                )
                continue
            global_hand = _rotate_xy(
                (hand - body_origin) / body_scale,
                body_angle,
            )
            palm_scale = float(
                np.mean([np.linalg.norm(hand[index, :2] - hand[0, :2]) for index in (5, 9, 13, 17)])
            )
            palm_scale = max(palm_scale, 0.025)
            local_hand = _rotate_xy((hand - hand[0]) / palm_scale, body_angle)
            hand_features.extend((global_hand, local_hand))

        face_features = np.zeros((len(FACE_INDICES), 3), dtype=np.float32)
        if face is not None:
            left_eye_center = (face[33] + face[133]) / 2.0
            right_eye_center = (face[263] + face[362]) / 2.0
            face_origin = (left_eye_center + right_eye_center) / 2.0
            face_scale = max(float(np.linalg.norm(left_eye_center[:2] - right_eye_center[:2])), 0.04)
            eye_axis = right_eye_center[:2] - left_eye_center[:2]
            face_angle = -float(np.arctan2(eye_axis[1], eye_axis[0]))
            face_features = _rotate_xy(
                (face[list(FACE_INDICES)] - face_origin) / face_scale,
                face_angle,
            )

        presence = np.asarray(
            (
                float(pose is not None),
                float(face is not None),
                float(left_hand is not None),
                float(right_hand is not None),
            ),
            dtype=np.float32,
        )
        features = np.concatenate(
            (
                pose_xyz.reshape(-1),
                *(hand.reshape(-1) for hand in hand_features),
                face_features.reshape(-1),
                pose_visibility,
                presence,
            )
        ).astype(np.float32)
        if features.shape != (static_feature_dim(),):
            raise RuntimeError(
                f"Dimensión inesperada: {features.shape}; se esperaba {(static_feature_dim(),)}"
            )
        quality = {
            "pose": float(presence[0]),
            "face": float(presence[1]),
            "left_hand": float(presence[2]),
            "right_hand": float(presence[3]),
        }
        return features, quality, pose_hands_xy


def resample_sequence(
    sequence: np.ndarray,
    length: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Remuestrea geometría válida sin crear landmarks fantasma."""
    if sequence.ndim != 2:
        raise ValueError("La secuencia debe tener forma [tiempo, características].")
    if sequence.shape[1] != static_feature_dim():
        raise ValueError(
            f"La secuencia debe tener {static_feature_dim()} características estáticas."
        )
    if len(sequence) == 0:
        raise ValueError("No se puede remuestrear una secuencia vacía.")
    if length < 1:
        raise ValueError("La longitud de salida debe ser al menos 1.")

    repaired = fill_short_landmark_gaps(sequence)
    source = np.arange(len(sequence), dtype=np.float32)
    if positions is None:
        positions = np.linspace(0, len(sequence) - 1, length, dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (length,):
            raise ValueError("positions debe tener exactamente un valor por frame de salida.")
    positions = np.clip(positions, 0, len(sequence) - 1)
    nearest = np.clip(np.floor(positions + 0.5).astype(int), 0, len(sequence) - 1)

    result = np.zeros((length, static_feature_dim()), dtype=np.float32)
    source_presence = repaired[:, PRESENCE_SLICE] >= 0.5
    target_presence = source_presence[nearest]
    result[:, PRESENCE_SLICE] = target_presence.astype(np.float32)

    for geometry_slice, presence_index in MODALITY_GEOMETRY:
        valid = source_presence[:, presence_index]
        if np.any(valid):
            for column in range(geometry_slice.start, geometry_slice.stop):
                result[:, column] = np.interp(
                    positions,
                    source[valid],
                    repaired[valid, column],
                )
        result[~target_presence[:, presence_index], geometry_slice] = 0.0

    pose_valid = source_presence[:, POSE_PRESENCE]
    if np.any(pose_valid):
        for column in range(POSE_VISIBILITY_SLICE.start, POSE_VISIBILITY_SLICE.stop):
            result[:, column] = np.interp(
                positions,
                source[pose_valid],
                repaired[pose_valid, column],
            )
    result[~target_presence[:, POSE_PRESENCE], POSE_VISIBILITY_SLICE] = 0.0
    return _zero_missing_geometry(result)


def _zero_missing_geometry(sequence: np.ndarray) -> np.ndarray:
    result = np.asarray(sequence, dtype=np.float32).copy()
    presence = result[:, PRESENCE_SLICE] >= 0.5
    result[:, PRESENCE_SLICE] = presence.astype(np.float32)
    for geometry_slice, presence_index in MODALITY_GEOMETRY:
        result[~presence[:, presence_index], geometry_slice] = 0.0
    result[~presence[:, POSE_PRESENCE], POSE_VISIBILITY_SLICE] = 0.0
    return result


def fill_short_landmark_gaps(
    sequence: np.ndarray,
    timestamps: np.ndarray | None = None,
    max_gap: int = MAX_INTERPOLATION_GAP,
) -> np.ndarray:
    """Interpola solamente ausencias internas cortas y conserva gaps largos en cero."""
    if sequence.ndim != 2 or sequence.shape[1] != static_feature_dim():
        raise ValueError("Secuencia estática v2 inválida.")
    if max_gap < 0:
        raise ValueError("max_gap no puede ser negativo.")
    result = _zero_missing_geometry(sequence)
    if len(result) < 3 or max_gap == 0:
        return result
    time_axis = (
        np.asarray(timestamps, dtype=np.float64)
        if timestamps is not None
        else np.arange(len(result), dtype=np.float64)
    )
    if time_axis.shape != (len(result),) or np.any(np.diff(time_axis) <= 0):
        raise ValueError("Los timestamps deben ser crecientes y coincidir con la secuencia.")

    for geometry_slice, presence_index in MODALITY_GEOMETRY:
        presence_column = PRESENCE_SLICE.start + presence_index
        observed = result[:, presence_column] >= 0.5
        index = 0
        while index < len(result):
            if observed[index]:
                index += 1
                continue
            start = index
            while index < len(result) and not observed[index]:
                index += 1
            stop = index
            gap_length = stop - start
            if (
                start == 0
                or stop == len(result)
                or gap_length > max_gap
                or not observed[start - 1]
                or not observed[stop]
            ):
                continue
            left = start - 1
            right = stop
            denominator = max(time_axis[right] - time_axis[left], 1e-8)
            columns = [geometry_slice]
            if presence_index == POSE_PRESENCE:
                columns.append(POSE_VISIBILITY_SLICE)
            for current in range(start, stop):
                alpha = float((time_axis[current] - time_axis[left]) / denominator)
                for column_slice in columns:
                    result[current, column_slice] = (
                        (1.0 - alpha) * result[left, column_slice]
                        + alpha * result[right, column_slice]
                    )
                result[current, presence_column] = 1.0
        # La máscara puede haber cambiado; el siguiente grupo se procesa por separado.
    return _zero_missing_geometry(result)


def _motion_energy(
    sequence: np.ndarray,
    timestamps: np.ndarray | None,
) -> np.ndarray:
    repaired = fill_short_landmark_gaps(sequence, timestamps=timestamps)
    count = len(repaired)
    if count < 2:
        return np.zeros(count, dtype=np.float32)
    if timestamps is None:
        delta_time = np.ones(count - 1, dtype=np.float32)
    else:
        time_axis = np.asarray(timestamps, dtype=np.float32)
        delta_time = np.maximum(np.diff(time_axis), 1e-3)
    presence = repaired[:, PRESENCE_SLICE] >= 0.5
    streams: list[np.ndarray] = []

    pose = repaired[:, POSE_XYZ_SLICE].reshape(count, len(POSE_INDICES), 3)
    arm_slots = [POSE_INDICES.index(index) for index in (13, 14, 15, 16)]
    pose_speed = np.linalg.norm(np.diff(pose[:, arm_slots], axis=0), axis=-1).mean(axis=1)
    pose_pairs = presence[1:, POSE_PRESENCE] & presence[:-1, POSE_PRESENCE]
    streams.append(np.where(pose_pairs, pose_speed / delta_time, 0.0))

    fingertips = np.asarray((4, 8, 12, 16, 20))
    for hand_slice, presence_index in (
        (LEFT_HAND_SLICE, LEFT_HAND_PRESENCE),
        (RIGHT_HAND_SLICE, RIGHT_HAND_PRESENCE),
    ):
        hand = repaired[:, hand_slice].reshape(count, 2, 21, 3)
        points = np.concatenate(
            (hand[:, 0, :1], hand[:, 1, fingertips]),
            axis=1,
        )
        speed = np.linalg.norm(np.diff(points, axis=0), axis=-1).mean(axis=1)
        valid_pairs = presence[1:, presence_index] & presence[:-1, presence_index]
        streams.append(np.where(valid_pairs, speed / delta_time, 0.0))

    energy = np.zeros(count, dtype=np.float32)
    energy[1:] = np.max(np.stack(streams), axis=0)
    window = min(5, count)
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / window
        energy = np.convolve(energy, kernel, mode="same").astype(np.float32)
    return energy


def _close_boolean_gaps(mask: np.ndarray, maximum: int = 3) -> np.ndarray:
    result = mask.astype(bool).copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= maximum:
            result[start:index] = True
    return result


def segment_active_sequence(
    sequence: np.ndarray,
    timestamps: np.ndarray | None = None,
    padding_seconds: float = SEGMENT_PADDING_SECONDS,
    minimum_seconds: float = SEGMENT_MINIMUM_SECONDS,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Recorta reposo externo usando energía robusta; ante duda conserva todo."""
    if len(sequence) < 4:
        return sequence, timestamps, {"fallback": True, "reason": "too_short"}
    time_axis = None if timestamps is None else np.asarray(timestamps, dtype=np.float32)
    energy = _motion_energy(sequence, time_axis)
    low, high = np.percentile(energy, (20, 90))
    span = float(high - low)
    if not np.isfinite(span) or span < 1e-5:
        return sequence, timestamps, {"fallback": True, "reason": "low_motion"}
    normalized = np.clip((energy - low) / (span + 1e-8), 0.0, 1.0)
    active = _close_boolean_gaps(
        normalized >= SEGMENT_ENERGY_THRESHOLD,
        maximum=SEGMENT_CLOSE_GAP_FRAMES,
    )
    # Eliminar islas de un solo frame, típicamente jitter del detector.
    for index in range(1, len(active) - 1):
        if active[index] and not active[index - 1] and not active[index + 1]:
            active[index] = False
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return sequence, timestamps, {"fallback": True, "reason": "no_activity"}
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if time_axis is not None and len(time_axis) == len(sequence):
        start_time = float(time_axis[start] - padding_seconds)
        stop_time = float(time_axis[stop - 1] + padding_seconds)
        start = max(0, int(np.searchsorted(time_axis, start_time, side="left")))
        stop = min(len(sequence), int(np.searchsorted(time_axis, stop_time, side="right")))
        while stop - start > 1 and time_axis[stop - 1] - time_axis[start] < minimum_seconds:
            if start > 0:
                start -= 1
            if stop < len(sequence):
                stop += 1
            if start == 0 and stop == len(sequence):
                break
    else:
        padding = max(2, int(round(len(sequence) * 0.07)))
        start = max(0, start - padding)
        stop = min(len(sequence), stop + padding)
    if stop - start < 4:
        return sequence, timestamps, {"fallback": True, "reason": "segment_too_short"}
    segmented_timestamps = None if time_axis is None else time_axis[start:stop]
    return (
        sequence[start:stop],
        segmented_timestamps,
        {
            "fallback": False,
            "start_frame": start,
            "stop_frame": stop,
            "retained_fraction": float((stop - start) / len(sequence)),
            "energy_p20": float(low),
            "energy_p90": float(high),
        },
    )


def mirror_sequence(sequence: np.ndarray) -> np.ndarray:
    """Refleja toda la geometría e intercambia articuladores izquierdos/derechos."""
    result = _zero_missing_geometry(sequence)
    pose = result[:, POSE_XYZ_SLICE].reshape(len(result), len(POSE_INDICES), 3)
    pose = pose[:, POSE_MIRROR_PERMUTATION].copy()
    pose[..., 0] *= -1.0
    result[:, POSE_XYZ_SLICE] = pose.reshape(len(result), -1)
    visibility = result[:, POSE_VISIBILITY_SLICE].copy()
    result[:, POSE_VISIBILITY_SLICE] = visibility[:, POSE_MIRROR_PERMUTATION]

    left = result[:, LEFT_HAND_SLICE].copy().reshape(len(result), 2, 21, 3)
    right = result[:, RIGHT_HAND_SLICE].copy().reshape(len(result), 2, 21, 3)
    left[..., 0] *= -1.0
    right[..., 0] *= -1.0
    result[:, LEFT_HAND_SLICE] = right.reshape(len(result), -1)
    result[:, RIGHT_HAND_SLICE] = left.reshape(len(result), -1)

    face = result[:, FACE_XYZ_SLICE].reshape(len(result), len(FACE_INDICES), 3)
    face[..., 0] *= -1.0
    result[:, FACE_XYZ_SLICE] = face.reshape(len(result), -1)
    presence = result[:, PRESENCE_SLICE].copy()
    result[:, PRESENCE_SLICE.start + LEFT_HAND_PRESENCE] = presence[:, RIGHT_HAND_PRESENCE]
    result[:, PRESENCE_SLICE.start + RIGHT_HAND_PRESENCE] = presence[:, LEFT_HAND_PRESENCE]
    return _zero_missing_geometry(result)


def _augment_spatial(
    sequence: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    force_mirror: bool | None = None,
) -> np.ndarray:
    result = sequence.copy()
    should_mirror = (
        float(rng.random()) < TRAIN_MIRROR_PROBABILITY
        if force_mirror is None
        else bool(force_mirror)
    )
    if should_mirror:
        result = mirror_sequence(result)
    angle = float(
        rng.uniform(-TRAIN_ROTATION_DEGREES, TRAIN_ROTATION_DEGREES)
        * np.pi
        / 180.0
    )
    scale = float(rng.uniform(*TRAIN_SCALE_RANGE))
    for geometry_slice, presence_index in MODALITY_GEOMETRY:
        points = result[:, geometry_slice].reshape(len(result), -1, 3)
        points = _rotate_xy(points, angle)
        if geometry_slice in (POSE_XYZ_SLICE, LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
            # La escala residual enseña variación de amplitud sin borrar ubicación corporal.
            points[..., :2] *= scale
        valid = result[:, PRESENCE_SLICE.start + presence_index] >= 0.5
        if noise_std > 0 and np.any(valid):
            points[valid] += rng.normal(
                0.0,
                noise_std,
                size=points[valid].shape,
            ).astype(np.float32)
        result[:, geometry_slice] = points.reshape(len(result), -1)
    return _zero_missing_geometry(result)


def random_temporal_view(
    sequence: np.ndarray,
    length: int,
    rng: np.random.Generator,
    timestamps: np.ndarray | None = None,
    min_fraction: float = TRAIN_MIN_CROP_FRACTION,
    noise_std: float = TRAIN_NOISE_STD,
    force_mirror: bool | None = None,
    speed_warp: float | None = None,
    return_plan: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    active, _, segment_metadata = segment_active_sequence(
        sequence,
        timestamps=timestamps,
    )
    active_start = (
        0
        if bool(segment_metadata.get("fallback", True))
        else int(segment_metadata["start_frame"])
    )
    total = len(active)
    fraction = float(rng.uniform(min_fraction, 1.0))
    crop_length = max(2, min(total, int(round(total * fraction))))
    max_start = max(0, total - crop_length)
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    crop = active[start : start + crop_length]

    phase = np.linspace(0.0, 1.0, length, dtype=np.float32)
    exponent = (
        float(rng.uniform(*TRAIN_SPEED_WARP_RANGE))
        if speed_warp is None
        else float(speed_warp)
    )
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("speed_warp debe ser finito y mayor que cero.")
    positions = np.power(phase, exponent) * max(len(crop) - 1, 0)
    view = resample_sequence(crop, length, positions)
    augmented = _augment_spatial(
        view,
        rng,
        noise_std=noise_std,
        force_mirror=force_mirror,
    )
    if not return_plan:
        return augmented
    return augmented, {
        "source_start": active_start + start,
        "source_stop": active_start + start + crop_length,
        "speed_warp": exponent,
        "mirror": force_mirror,
    }


def deterministic_views(
    sequence: np.ndarray,
    length: int,
    timestamps: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Tres vistas por fase para tolerar ejecuciones más rápidas o lentas."""
    active, _, _ = segment_active_sequence(sequence, timestamps=timestamps)
    phase = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return [
        resample_sequence(active, length, np.power(phase, exponent) * (len(active) - 1))
        for exponent in TEMPORAL_VIEW_EXPONENTS
    ]


def add_temporal_deltas(sequence: np.ndarray) -> np.ndarray:
    """Añade velocidad por fase únicamente entre observaciones válidas."""
    sequence = _zero_missing_geometry(sequence)
    geometry = sequence[:, :GEOMETRY_DIM]
    delta = np.diff(geometry, axis=0, prepend=geometry[:1])
    presence = sequence[:, PRESENCE_SLICE] >= 0.5
    previous = np.vstack((np.zeros((1, PRESENCE_DIM), dtype=bool), presence[:-1]))
    for geometry_slice, presence_index in MODALITY_GEOMETRY:
        valid_pair = presence[:, presence_index] & previous[:, presence_index]
        delta[~valid_pair, geometry_slice] = 0.0
    return np.concatenate((sequence, delta), axis=1).astype(np.float32)


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray
    layout_version: str = FEATURE_LAYOUT_VERSION

    @classmethod
    def fit(cls, matrices: list[np.ndarray]) -> "FeatureStandardizer":
        if not matrices:
            raise ValueError("Se necesita al menos una matriz para ajustar el scaler.")
        if any(matrix.ndim != 2 or matrix.shape[1] != model_feature_dim() for matrix in matrices):
            raise ValueError("Todas las matrices deben usar el layout de modelo v2.")
        mean = np.zeros(model_feature_dim(), dtype=np.float64)
        std = np.ones(model_feature_dim(), dtype=np.float64)
        delta_offset = static_feature_dim()
        for geometry_slice, presence_index in MODALITY_GEOMETRY:
            static_values: list[np.ndarray] = []
            delta_values: list[np.ndarray] = []
            for matrix in matrices:
                presence = matrix[:, PRESENCE_SLICE.start + presence_index] >= 0.5
                if np.any(presence):
                    static_values.append(matrix[presence, geometry_slice])
                previous = np.r_[False, presence[:-1]]
                valid_delta = presence & previous
                if np.any(valid_delta):
                    delta_slice = slice(
                        delta_offset + geometry_slice.start,
                        delta_offset + geometry_slice.stop,
                    )
                    delta_values.append(matrix[valid_delta, delta_slice])
            for values, destination in (
                (static_values, geometry_slice),
                (
                    delta_values,
                    slice(
                        delta_offset + geometry_slice.start,
                        delta_offset + geometry_slice.stop,
                    ),
                ),
            ):
                if not values:
                    continue
                joined = np.concatenate(values, axis=0).astype(np.float64)
                group_mean = joined.mean(axis=0)
                group_std = joined.std(axis=0)
                mean[destination] = group_mean
                std[destination] = np.where(group_std < 1e-4, 1.0, group_std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError(
                f"Dimensión incompatible con scaler {self.layout_version}: "
                f"{matrix.shape}."
            )
        transformed = np.asarray(matrix, dtype=np.float32).copy()
        delta_offset = static_feature_dim()
        for geometry_slice, presence_index in MODALITY_GEOMETRY:
            presence = matrix[:, PRESENCE_SLICE.start + presence_index] >= 0.5
            transformed[:, geometry_slice] = 0.0
            if np.any(presence):
                transformed[presence, geometry_slice] = (
                    matrix[presence, geometry_slice] - self.mean[geometry_slice]
                ) / self.std[geometry_slice]
            previous = np.r_[False, presence[:-1]]
            valid_delta = presence & previous
            delta_slice = slice(
                delta_offset + geometry_slice.start,
                delta_offset + geometry_slice.stop,
            )
            transformed[:, delta_slice] = 0.0
            if np.any(valid_delta):
                transformed[valid_delta, delta_slice] = (
                    matrix[valid_delta, delta_slice] - self.mean[delta_slice]
                ) / self.std[delta_slice]
        pose_presence = (
            matrix[:, PRESENCE_SLICE.start + POSE_PRESENCE] >= 0.5
        )
        transformed[:, POSE_VISIBILITY_SLICE] = np.clip(
            transformed[:, POSE_VISIBILITY_SLICE],
            0.0,
            1.0,
        )
        transformed[~pose_presence, POSE_VISIBILITY_SLICE] = 0.0
        transformed[:, PRESENCE_SLICE] = (
            transformed[:, PRESENCE_SLICE] >= 0.5
        ).astype(np.float32)
        return np.clip(
            transformed,
            -STANDARDIZER_CLIP,
            STANDARDIZER_CLIP,
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "layout_version": self.layout_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStandardizer":
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
            str(payload.get("layout_version", "legacy_v1")),
        )
