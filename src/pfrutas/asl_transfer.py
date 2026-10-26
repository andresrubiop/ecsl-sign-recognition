"""Adaptador de transferencia pose-a-pose desde ASL Citizen v1.

La arquitectura ST-GCN y el preprocesamiento base de este modulo estan
adaptados del repositorio oficial ``microsoft/ASL-citizen-code``:
https://github.com/microsoft/ASL-citizen-code

Portions Copyright (c) Microsoft Corporation, usadas bajo licencia MIT
(SPDX-License-Identifier: MIT). Este proyecto no redistribuye pesos ni videos
de ASL Citizen; quien aporte un checkpoint debe comprobar por separado los
terminos aplicables al dataset y a ese archivo.

El camino determinista reproduce las decisiones del baseline oficial que son
importantes para compatibilidad de pesos: remuestreo por descarte, padding a
128, normalizacion global por hombros (incluyendo el padding), reordenamiento
de manos y seleccion del grafo de 27 nodos. Todo tensor entregado al modelo se
convierte explicitamente a ``float32``.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ASL_CITIZEN_DATASET_VERSION = "1.0"
ASL_CITIZEN_MODEL_NAME = "ASL Citizen v1 ST-GCN"
ASL_CITIZEN_CODE_REPOSITORY = "https://github.com/microsoft/ASL-citizen-code"
ASL_CITIZEN_CODE_LICENSE = "MIT"
ASL_CITIZEN_DATASET_PAGE = (
    "https://www.microsoft.com/en-us/research/project/asl-citizen/"
)
ASL_CITIZEN_CHECKPOINT_URL = (
    "https://github.com/microsoft/ASL-citizen-code/releases/download/"
    "checkpoints_v1/ASL_citizen_stgcn_weights.zip"
)
ASL_CITIZEN_ARCHIVE_SHA256 = (
    "4b52578ed24ed975756c6447c56ba9bb90576eb302518c9b3aef955bfb20eb01"
)
ASL_CITIZEN_CHECKPOINT_SHA256 = (
    "b08d84afb5a0fdf4c723fd7be2db897b2490a921ccf538cff1b345f905934d68"
)
ASL_CITIZEN_OFFICIAL_CHECKPOINT_SHA256 = (
    "b08d84afb5a0fdf4c723fd7be2db897b2490a921ccf538cff1b345f905934d68"
)

ASL_RAW_POSE_POINTS = 33
ASL_RAW_HAND_POINTS = 21
ASL_RAW_POINT_COUNT = ASL_RAW_POSE_POINTS + 2 * ASL_RAW_HAND_POINTS
TRANSFER_CACHE_LAYOUT_VERSION = "pose_hands_xy_v1"
ASL_GRAPH_SEQUENCE_LENGTH = 128
ASL_GRAPH_NUM_NODES = 27
ASL_GRAPH_INPUT_CHANNELS = 2
ASL_GRAPH_EMBEDDING_DIM = 256

# Indices publicados por Microsoft despues de reordenar
# pose33 + right21 + left21 -> pose33 + left21 + right21.
ASL_CITIZEN_GRAPH_INDICES = (
    0,
    2,
    5,
    11,
    12,
    13,
    14,
    33,
    37,
    38,
    41,
    42,
    45,
    46,
    49,
    50,
    53,
    54,
    58,
    59,
    62,
    63,
    66,
    67,
    70,
    71,
    74,
)

# Los mismos nodos expresados directamente en el layout crudo. Es util para
# auditoria, aunque la implementacion hace el reordenamiento de forma explicita.
ASL_GRAPH_RAW_INDICES = (
    0,
    2,
    5,
    11,
    12,
    13,
    14,
    54,
    58,
    59,
    62,
    63,
    66,
    67,
    70,
    71,
    74,
    33,
    37,
    38,
    41,
    42,
    45,
    46,
    49,
    50,
    53,
)

ASL_CITIZEN_INWARD_EDGES = (
    (2, 0),
    (1, 0),
    (0, 3),
    (0, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 17),
    (7, 8),
    (7, 9),
    (9, 10),
    (7, 11),
    (11, 12),
    (7, 13),
    (13, 14),
    (7, 15),
    (15, 16),
    (17, 18),
    (17, 19),
    (19, 20),
    (17, 21),
    (21, 22),
    (17, 23),
    (23, 24),
    (17, 25),
    (25, 26),
)

# Permutacion semantica de los 33 landmarks de MediaPipe Pose al reflejar.
_POSE_MIRROR_PAIRS = {
    1: 4,
    2: 5,
    3: 6,
    4: 1,
    5: 2,
    6: 3,
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
    17: 18,
    18: 17,
    19: 20,
    20: 19,
    21: 22,
    22: 21,
    23: 24,
    24: 23,
    25: 26,
    26: 25,
    27: 28,
    28: 27,
    29: 30,
    30: 29,
    31: 32,
    32: 31,
}
ASL_POSE_MIRROR_PERMUTATION = tuple(
    _POSE_MIRROR_PAIRS.get(index, index) for index in range(ASL_RAW_POSE_POINTS)
)

ASL_GRAPH_CONTRACT: dict[str, Any] = {
    "name": ASL_CITIZEN_MODEL_NAME,
    "dataset_version": ASL_CITIZEN_DATASET_VERSION,
    "cache_layout_version": TRANSFER_CACHE_LAYOUT_VERSION,
    "raw_layout": "pose_0..32 + right_hand_0..20 + left_hand_0..20",
    "raw_shape": ["frames", ASL_RAW_POINT_COUNT, ASL_GRAPH_INPUT_CHANNELS],
    "raw_coordinates": "mediapipe_image_normalized_xy; missing modalities are zero",
    "graph_shape": [
        ASL_GRAPH_INPUT_CHANNELS,
        ASL_GRAPH_SEQUENCE_LENGTH,
        ASL_GRAPH_NUM_NODES,
    ],
    "graph_indices_after_pose_left_right_reorder": list(
        ASL_CITIZEN_GRAPH_INDICES
    ),
    "temporal_preprocessing": "official discard-downsample then zero-pad to 128",
    "normalization": "mean shoulder midpoint and mean shoulder distance over all 128 frames, padding included",
    "dtype": "float32",
    "embedding_dim": ASL_GRAPH_EMBEDDING_DIM,
    "code_source": ASL_CITIZEN_CODE_REPOSITORY,
    "code_license": ASL_CITIZEN_CODE_LICENSE,
    "weights_bundled": False,
}


def _as_raw_pose_hands(raw_pose_hands_xy: np.ndarray) -> np.ndarray:
    try:
        data = np.asarray(raw_pose_hands_xy, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "pose_hands_xy debe ser un arreglo numerico [T, 75, 2]."
        ) from exc
    if data.ndim != 3 or data.shape[1:] != (ASL_RAW_POINT_COUNT, 2):
        raise ValueError(
            "pose_hands_xy debe tener forma [T, 75, 2] en orden "
            "pose33 + right_hand21 + left_hand21."
        )
    if len(data) == 0:
        raise ValueError("pose_hands_xy no puede estar vacio.")
    if not np.isfinite(data).all():
        raise ValueError("pose_hands_xy contiene valores NaN o infinitos.")
    return np.ascontiguousarray(data)


def _modality_is_present(block: np.ndarray) -> np.ndarray:
    """Mascara por frame compatible con el cero usado por MediaPipe Holistic."""
    return np.any(block != 0.0, axis=(1, 2))


def mirror_pose_hands_xy(raw_pose_hands_xy: np.ndarray) -> np.ndarray:
    """Refleja x e intercambia lados sin convertir landmarks ausentes en x=1.

    La operacion es involutiva. En Pose se permutan todos los pares anatomicos;
    las manos conservan sus indices internos y se intercambian como bloques.
    """
    source = _as_raw_pose_hands(raw_pose_hands_xy)
    result = np.zeros_like(source)

    pose = source[:, :ASL_RAW_POSE_POINTS]
    pose_present = _modality_is_present(pose)
    mirrored_pose = pose[:, ASL_POSE_MIRROR_PERMUTATION].copy()
    mirrored_pose[..., 0] = 1.0 - mirrored_pose[..., 0]
    result[pose_present, :ASL_RAW_POSE_POINTS] = mirrored_pose[pose_present]

    right = source[:, ASL_RAW_POSE_POINTS : ASL_RAW_POSE_POINTS + ASL_RAW_HAND_POINTS]
    left = source[:, ASL_RAW_POSE_POINTS + ASL_RAW_HAND_POINTS :]
    right_present = _modality_is_present(right)
    left_present = _modality_is_present(left)

    mirrored_left = right.copy()
    mirrored_left[..., 0] = 1.0 - mirrored_left[..., 0]
    left_slice = slice(
        ASL_RAW_POSE_POINTS + ASL_RAW_HAND_POINTS,
        ASL_RAW_POINT_COUNT,
    )
    result[right_present, left_slice] = mirrored_left[right_present]

    mirrored_right = left.copy()
    mirrored_right[..., 0] = 1.0 - mirrored_right[..., 0]
    right_slice = slice(
        ASL_RAW_POSE_POINTS,
        ASL_RAW_POSE_POINTS + ASL_RAW_HAND_POINTS,
    )
    result[left_present, right_slice] = mirrored_right[left_present]
    return result.astype(np.float32)


def _official_downsample(frames: np.ndarray, max_frames: int) -> np.ndarray:
    """Port literal del descarte temporal publicado por Microsoft."""
    length = frames.shape[0]
    increment = max_frames / length
    if increment > 1.0:
        increment = 1.0
    curr_increment = 0.0
    curr_frame = 0
    new_frames: list[np.ndarray] = []
    for frame in frames:
        curr_increment += increment
        if curr_increment > curr_frame:
            curr_frame += 1
            new_frames.append(frame)
    result = np.asarray(new_frames)
    if len(result) > max_frames:
        result = result[:max_frames]
    return result


def asl_citizen_temporal_fit(
    raw_pose_hands_xy: np.ndarray,
    length: int = ASL_GRAPH_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Aplica exactamente el downsample/pad del dataset oficial."""
    if length < 1:
        raise ValueError("length debe ser positivo.")
    data = _as_raw_pose_hands(raw_pose_hands_xy)
    original_length = data.shape[0]
    if original_length > length:
        data = _official_downsample(data, length)
    if original_length < length:
        data = np.pad(data, ((0, length - original_length), (0, 0), (0, 0)))
    if data.shape != (length, ASL_RAW_POINT_COUNT, 2):
        raise RuntimeError(
            f"El ajuste temporal produjo {data.shape}; se esperaba "
            f"{(length, ASL_RAW_POINT_COUNT, 2)}."
        )
    return np.asarray(data, dtype=np.float32)


def normalize_asl_citizen_shoulders(fitted_pose_hands_xy: np.ndarray) -> np.ndarray:
    """Normalizacion oficial global, deliberadamente incluyendo el padding."""
    data = np.asarray(fitted_pose_hands_xy, dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != (ASL_RAW_POINT_COUNT, 2):
        raise ValueError("La secuencia ajustada debe tener forma [T, 75, 2].")
    if not np.isfinite(data).all():
        raise ValueError("La secuencia ajustada contiene NaN o infinitos.")

    shoulder_l = data[:, 11, :]
    shoulder_r = data[:, 12, :]

    # Equivalente al bucle del codigo oficial; todos los frames, incluso los
    # ceros agregados por padding, contribuyen al centro y a mean_dist.
    center = np.zeros(2, dtype=np.float64)
    for index in range(len(shoulder_l)):
        center_i = (shoulder_r[index] + shoulder_l[index]) / 2.0
        center = center + center_i
    center = center / shoulder_l.shape[0]

    mean_dist = float(
        np.mean(np.sqrt(((shoulder_l - shoulder_r) ** 2).sum(-1)))
    )
    if mean_dist != 0.0:
        data = (data - center) * (1.0 / mean_dist)
    return np.asarray(data, dtype=np.float32)


def select_asl_graph_keypoints(pose_hands_xy: np.ndarray) -> np.ndarray:
    """Reordena pose+right+left a pose+left+right y selecciona 27 nodos."""
    data = np.asarray(pose_hands_xy)
    if data.ndim < 2 or data.shape[-2:] != (ASL_RAW_POINT_COUNT, 2):
        raise ValueError("La entrada debe terminar en dimensiones [75, 2].")
    posedata = data[..., 0:ASL_RAW_POSE_POINTS, :]
    rhdata = data[
        ...,
        ASL_RAW_POSE_POINTS : ASL_RAW_POSE_POINTS + ASL_RAW_HAND_POINTS,
        :,
    ]
    lhdata = data[..., ASL_RAW_POSE_POINTS + ASL_RAW_HAND_POINTS :, :]
    reordered = np.concatenate((posedata, lhdata, rhdata), axis=-2)
    return np.asarray(
        reordered[..., list(ASL_CITIZEN_GRAPH_INDICES), :],
        dtype=np.float32,
    )


def _validated_timestamps(
    timestamps: np.ndarray | None,
    length: int,
) -> np.ndarray | None:
    if timestamps is None:
        return None
    time_axis = np.asarray(timestamps, dtype=np.float64)
    if time_axis.shape != (length,):
        raise ValueError("timestamps debe tener un valor por frame.")
    if not np.isfinite(time_axis).all() or np.any(np.diff(time_axis) <= 0.0):
        raise ValueError("timestamps debe ser finito y estrictamente creciente.")
    return time_axis


def _active_raw_view(
    raw: np.ndarray,
    static_sequence: np.ndarray | None,
    timestamps: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    time_axis = _validated_timestamps(timestamps, len(raw))
    if static_sequence is None:
        return raw, time_axis
    static = np.asarray(static_sequence, dtype=np.float32)
    if static.ndim != 2 or len(static) != len(raw):
        raise ValueError(
            "static_sequence debe ser [T, features] y coincidir con pose_hands_xy."
        )

    # Se usa el mismo detector de actividad de la rama v2 para recortar ambos
    # streams con indices identicos; el preprocesamiento Microsoft empieza
    # despues de este recorte opcional.
    from .features import segment_active_sequence

    _, _, metadata = segment_active_sequence(static, timestamps=time_axis)
    if bool(metadata.get("fallback", True)):
        return raw, time_axis
    start = int(metadata["start_frame"])
    stop = int(metadata["stop_frame"])
    cropped_times = None if time_axis is None else time_axis[start:stop]
    return raw[start:stop], cropped_times


def _phase_warp_nearest(raw: np.ndarray, exponent: float) -> np.ndarray:
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("phase_exponent debe ser finito y mayor que cero.")
    if len(raw) < 2 or exponent == 1.0:
        return raw
    phase = np.linspace(0.0, 1.0, len(raw), dtype=np.float64)
    positions = np.power(phase, exponent) * (len(raw) - 1)
    # Nearest conserva exactamente los ceros que codifican modalidades ausentes.
    indices = np.floor(positions + 0.5).astype(np.int64)
    return raw[np.clip(indices, 0, len(raw) - 1)]


def prepare_asl_graph(
    raw_pose_hands_xy: np.ndarray,
    static_sequence: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    mirror: bool = False,
    phase_exponent: float = 1.0,
    frame_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    """Prepara un tensor ASL Citizen ``[2, 128, 27]`` en ``float32``.

    Con los defaults y sin ``static_sequence`` el camino reproduce el baseline
    oficial. El recorte de actividad, mirror y phase warp son vistas opcionales
    aplicadas antes del downsample/pad oficial.
    """
    raw = _as_raw_pose_hands(raw_pose_hands_xy)
    if frame_bounds is None:
        raw, _ = _active_raw_view(raw, static_sequence, timestamps)
    else:
        start, stop = (int(value) for value in frame_bounds)
        if not 0 <= start < stop <= len(raw):
            raise ValueError(
                f"frame_bounds inválido para {len(raw)} frames: {(start, stop)}."
            )
        raw = raw[start:stop]
    raw = _phase_warp_nearest(raw, float(phase_exponent))
    if mirror:
        raw = mirror_pose_hands_xy(raw)

    fitted = asl_citizen_temporal_fit(raw, ASL_GRAPH_SEQUENCE_LENGTH)
    normalized = normalize_asl_citizen_shoulders(fitted)
    graph = select_asl_graph_keypoints(normalized)
    graph = np.transpose(graph, (2, 0, 1))
    expected = (
        ASL_GRAPH_INPUT_CHANNELS,
        ASL_GRAPH_SEQUENCE_LENGTH,
        ASL_GRAPH_NUM_NODES,
    )
    if graph.shape != expected:
        raise RuntimeError(f"Tensor ASL inesperado: {graph.shape}; esperado {expected}.")
    return np.ascontiguousarray(graph, dtype=np.float32)


def prepare_asl_graph_augmented(
    raw_pose_hands_xy: np.ndarray,
    static_sequence: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    mirror_probability: float = 0.5,
    phase_exponent_range: tuple[float, float] = (0.80, 1.25),
    shear_std: float = 0.10,
    rotation_std: float = 0.10,
    force_mirror: bool | None = None,
    phase_exponent: float | None = None,
    frame_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    """Vista de entrenamiento: lateralidad/velocidad y transforms oficiales.

    Shear y rotacion se aplican despues de normalizar, igual que en el Dataset
    oficial. ``rng`` es obligatorio para que cada fold pueda ser reproducible.
    """
    if rng is None:
        raise ValueError("prepare_asl_graph_augmented requiere un rng explicito.")
    if not 0.0 <= mirror_probability <= 1.0:
        raise ValueError("mirror_probability debe estar entre 0 y 1.")
    low, high = (float(value) for value in phase_exponent_range)
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high):
        raise ValueError("phase_exponent_range debe ser positivo y ordenado.")
    if shear_std < 0.0 or rotation_std < 0.0:
        raise ValueError("Los std de augmentacion no pueden ser negativos.")

    should_mirror = (
        bool(rng.random() < mirror_probability)
        if force_mirror is None
        else bool(force_mirror)
    )
    exponent = (
        float(rng.uniform(low, high))
        if phase_exponent is None
        else float(phase_exponent)
    )
    graph = prepare_asl_graph(
        raw_pose_hands_xy,
        static_sequence=static_sequence,
        timestamps=timestamps,
        mirror=should_mirror,
        phase_exponent=exponent,
        frame_bounds=frame_bounds,
    )
    tvc = np.transpose(graph, (1, 2, 0)).astype(np.float32)

    shear = float(rng.normal(0.0, shear_std)) if shear_std else 0.0
    shear_matrix = np.asarray(((1.0, shear), (0.0, 1.0)), dtype=np.float32)
    tvc = np.matmul(tvc, shear_matrix)

    angle = float(rng.normal(0.0, rotation_std)) if rotation_std else 0.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation_matrix = np.asarray(
        ((cosine, -sine), (sine, cosine)),
        dtype=np.float32,
    )
    tvc = np.matmul(tvc, rotation_matrix)
    return np.ascontiguousarray(np.transpose(tvc, (2, 0, 1)), dtype=np.float32)


def _normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, 0)
    inverse = np.zeros_like(adjacency)
    for index in range(adjacency.shape[0]):
        if degree[index] > 0:
            inverse[index, index] = degree[index] ** -1
    return np.dot(adjacency, inverse)


def _hop_distance(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
    max_hop: int = 1,
) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes))
    for start, stop in edges:
        adjacency[stop, start] = 1
        adjacency[start, stop] = 1
    distances = np.zeros((num_nodes, num_nodes)) + np.inf
    transfer = [np.linalg.matrix_power(adjacency, hop) for hop in range(max_hop + 1)]
    arrivals = np.stack(transfer) > 0
    for hop in range(max_hop, -1, -1):
        distances[arrivals[hop]] = hop
    return distances


def _partitioned_adjacency(
    num_nodes: int,
    center: int,
    inward_edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    self_edges = tuple((index, index) for index in range(num_nodes))
    distances = _hop_distance(num_nodes, self_edges + inward_edges, max_hop=1)
    adjacency = np.zeros((num_nodes, num_nodes))
    for hop in range(2):
        adjacency[distances == hop] = 1
    normalized = _normalize_digraph(adjacency)

    partitions: list[np.ndarray] = []
    for hop in range(2):
        root = np.zeros((num_nodes, num_nodes))
        close = np.zeros((num_nodes, num_nodes))
        further = np.zeros((num_nodes, num_nodes))
        for start in range(num_nodes):
            for stop in range(num_nodes):
                if distances[stop, start] != hop:
                    continue
                if distances[stop, center] == distances[start, center]:
                    root[stop, start] = normalized[stop, start]
                elif distances[stop, center] > distances[start, center]:
                    close[stop, start] = normalized[stop, start]
                else:
                    further[stop, start] = normalized[stop, start]
        if hop == 0:
            partitions.append(root)
        else:
            partitions.extend((root + close, further))
    return np.stack(partitions)


class _ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        t_kernel_size: int = 1,
        t_stride: int = 1,
        t_padding: int = 0,
        t_dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adjacency.size(0) != self.kernel_size:
            raise ValueError("Particiones de adyacencia incompatibles.")
        x = self.conv(x)
        batch, channels, frames, vertices = x.size()
        x = x.view(
            batch,
            self.kernel_size,
            channels // self.kernel_size,
            frames,
            vertices,
        )
        x = torch.einsum("nkctv,kvw->nctw", (x, adjacency))
        return x.contiguous(), adjacency


class _STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if len(kernel_size) != 2 or kernel_size[0] % 2 != 1:
            raise ValueError("kernel_size temporal debe ser impar.")
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = _ConvTemporalGraphical(
            in_channels,
            out_channels,
            kernel_size[1],
        )
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda value: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda value: value
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual(x)
        x, adjacency = self.gcn(x, adjacency)
        x = self.tcn(x) + residual
        return self.relu(x), adjacency


class ASLCitizenSTGCN(nn.Module):
    """Encoder ST-GCN oficial de 10 bloques y embedding de 256."""

    def __init__(self) -> None:
        super().__init__()
        adjacency = torch.tensor(
            _partitioned_adjacency(
                ASL_GRAPH_NUM_NODES,
                center=0,
                inward_edges=ASL_CITIZEN_INWARD_EDGES,
            ),
            dtype=torch.float32,
            requires_grad=False,
        )
        self.register_buffer("A", adjacency)
        spatial_kernel_size = adjacency.size(0)
        kernel_size = (9, spatial_kernel_size)
        self.n_out_features = ASL_GRAPH_EMBEDDING_DIM
        self.data_bn = nn.BatchNorm1d(
            ASL_GRAPH_INPUT_CHANNELS * ASL_GRAPH_NUM_NODES
        )
        self.st_gcn_networks = nn.ModuleList(
            (
                _STGCNBlock(2, 64, kernel_size, 1, residual=False),
                _STGCNBlock(64, 64, kernel_size, 1),
                _STGCNBlock(64, 64, kernel_size, 1),
                _STGCNBlock(64, 64, kernel_size, 1),
                _STGCNBlock(64, 128, kernel_size, 2),
                _STGCNBlock(128, 128, kernel_size, 1),
                _STGCNBlock(128, 128, kernel_size, 1),
                _STGCNBlock(128, 256, kernel_size, 2),
                _STGCNBlock(256, 256, kernel_size, 1),
                _STGCNBlock(256, ASL_GRAPH_EMBEDDING_DIM, kernel_size, 1),
            )
        )
        self.edge_importance = nn.ParameterList(
            [
                nn.Parameter(torch.ones_like(self.A, dtype=torch.float32))
                for _ in self.st_gcn_networks
            ]
        )
        # Protege el contrato aun si el dtype global de Torch fue alterado.
        self.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2 or x.shape[3] != ASL_GRAPH_NUM_NODES:
            raise ValueError("ASLCitizenSTGCN espera [N, 2, T, 27].")
        x = x.to(dtype=torch.float32)
        batch, channels, frames, vertices = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, vertices * channels, frames)
        x = self.data_bn(x)
        x = x.view(batch, vertices, channels, frames)
        x = x.permute(0, 2, 3, 1).contiguous()

        if len(self.st_gcn_networks) != len(self.edge_importance):
            raise RuntimeError("Las capas ST-GCN y sus pesos de arista no coinciden.")
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, x.size()[2:])
        return x.view(batch, -1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_from_payload(payload: Any) -> Mapping[str, Any]:
    candidates: list[Any] = [payload]
    if isinstance(payload, Mapping):
        candidates.extend(
            payload.get(key)
            for key in ("state_dict", "model_state", "model_state_dict")
            if key in payload
        )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if any(
            isinstance(key, str) and key.startswith("encoder.")
            for key in candidate
        ):
            return candidate
    raise ValueError(
        "El checkpoint no contiene un state_dict con claves encoder.*."
    )


def load_asl_citizen_encoder(
    model: nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Carga solo ``encoder.*`` con SHA-256, ``weights_only`` y strict exacto.

    No existe fallback a pickle inseguro. Los tensores de punto flotante se
    desprenden, mueven a CPU y convierten a fp32 antes de cargar.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint ASL: {path}")
    actual_sha256 = _sha256(path)
    normalized_expected: str | None = None
    if expected_sha256 is not None:
        normalized_expected = expected_sha256.strip().casefold()
        if re.fullmatch(r"[0-9a-f]{64}", normalized_expected) is None:
            raise ValueError("expected_sha256 debe contener 64 digitos hexadecimales.")
        if not hmac.compare_digest(actual_sha256, normalized_expected):
            raise ValueError(
                "SHA-256 del checkpoint ASL no coincide con el valor esperado."
            )

    # Deliberadamente no se captura el error para reintentar con
    # weights_only=False: esa degradacion invalidaria el contrato de seguridad.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source_state = _state_dict_from_payload(payload)
    encoder_state: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if not isinstance(key, str) or not key.startswith("encoder."):
            continue
        # ``str.removeprefix`` se introdujo en Python 3.9. La condición del
        # bucle ya comprobó el prefijo, de modo que el slicing es equivalente y
        # permite la inferencia experimental en JetPack 5/Python 3.8.
        stripped = key[len("encoder.") :]
        if not stripped or stripped in encoder_state:
            raise ValueError(f"Clave encoder invalida o duplicada: {key!r}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"La clave {key!r} no contiene un torch.Tensor.")
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        encoder_state[stripped] = tensor

    expected_keys = set(model.state_dict())
    loaded_keys = set(encoder_state)
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("faltan: " + ", ".join(missing[:8]))
        if unexpected:
            details.append("sobran: " + ", ".join(unexpected[:8]))
        raise RuntimeError(
            "Estado encoder incompatible con carga strict (" + "; ".join(details) + ")."
        )
    model.load_state_dict(encoder_state, strict=True)
    model.float()

    return {
        "model": ASL_CITIZEN_MODEL_NAME,
        "dataset_version": ASL_CITIZEN_DATASET_VERSION,
        "checkpoint_path": str(path.resolve()),
        "sha256": actual_sha256,
        "expected_sha256": normalized_expected,
        "sha256_verified": normalized_expected is not None,
        "encoder_prefix": "encoder.",
        "loaded_tensor_count": len(encoder_state),
        "dtype": "float32",
        "strict": True,
        "safe_weights_only": True,
        "code_source": ASL_CITIZEN_CODE_REPOSITORY,
        "code_license": ASL_CITIZEN_CODE_LICENSE,
        "weights_bundled": False,
    }
