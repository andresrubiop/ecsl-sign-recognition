from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .data import VideoRecord, load_manifest, transfer_feature_layout
from .features import (
    FeatureStandardizer,
    POSE_HANDS_XY_LAYOUT_VERSION,
    TRAIN_MIRROR_PROBABILITY,
    TRAIN_SPEED_WARP_RANGE,
    TEMPORAL_VIEW_EXPONENTS,
    add_temporal_deltas,
    deterministic_views,
    feature_layout,
    mirror_sequence,
    model_feature_dim,
    random_temporal_view,
)
from .model import (
    ASLTransferFusion,
    ASLTransferModelConfig,
    DEFAULT_MODEL_PRESET,
    ModelConfig,
    ModelConfiguration,
    SignModel,
    SignLSTM,
    build_model,
    is_asl_transfer_config,
    load_checkpoint,
    model_config_for_preset,
    save_checkpoint,
)
from .utils import read_json, resolve_device, seed_everything, write_json


VALIDATION_PROTOCOL = "nested_leave_one_signer_out_v1"
EXPERIMENT_IMPLEMENTATION_VERSION = "nested_loso_cuda_amp_mask_v2_compact_v2"
TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION = (
    "nested_loso_cuda_amp_asl_citizen_stgcn_fusion_v1"
)
SUPPORTED_EXPERIMENT_IMPLEMENTATION_VERSIONS = frozenset(
    {
        EXPERIMENT_IMPLEMENTATION_VERSION,
        TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION,
    }
)


@lru_cache(maxsize=256)
def _load_sequence_cached(path_text: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as payload:
        return (
            payload["features"].astype(np.float32),
            payload["timestamps"].astype(np.float32),
        )


def sequence_for(
    record: VideoRecord,
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    return _load_sequence_cached(str((processed_dir / record.sequence).resolve()))


@lru_cache(maxsize=256)
def _load_pose_hands_cached(path_text: str) -> np.ndarray:
    with np.load(path_text, allow_pickle=False) as payload:
        try:
            coordinates = payload["pose_hands_xy"].astype(np.float32)
        except KeyError as exc:
            raise RuntimeError(
                "La secuencia no contiene pose_hands_xy. Ejecuta primero "
                "scripts/05_prepare_asl_transfer.py."
            ) from exc
    if (
        coordinates.ndim != 3
        or coordinates.shape[1:] != (75, 2)
        or not np.all(np.isfinite(coordinates))
    ):
        raise RuntimeError(
            f"pose_hands_xy inválido en {path_text}: {coordinates.shape}."
        )
    return coordinates


def pose_hands_for(record: VideoRecord, processed_dir: Path) -> np.ndarray:
    return _load_pose_hands_cached(
        str((processed_dir / record.sequence).resolve())
    )


class AugmentedSequenceDataset:
    """Dataset compatible con PyTorch, con recortes temporales en cada acceso."""

    def __init__(
        self,
        records: list[VideoRecord],
        processed_dir: Path,
        sequence_length: int,
        standardizer: FeatureStandardizer,
        samples_per_video: int,
        seed: int,
        model_config: ModelConfiguration,
    ) -> None:
        self.records = records
        self.processed_dir = processed_dir
        self.sequence_length = sequence_length
        self.standardizer = standardizer
        self.samples_per_video = samples_per_video
        self.seed = seed
        self.model_config = model_config
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_video

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        record = self.records[index % len(self.records)]
        rng = np.random.default_rng(
            np.random.SeedSequence((self.seed, self.epoch, index))
        )
        sequence, timestamps = sequence_for(record, self.processed_dir)
        transfer_mirror: bool | None = None
        transfer_speed: float | None = None
        if is_asl_transfer_config(self.model_config):
            transfer_mirror = bool(rng.random() < TRAIN_MIRROR_PROBABILITY)
            transfer_speed = float(rng.uniform(*TRAIN_SPEED_WARP_RANGE))
        temporal_result = random_temporal_view(
            sequence,
            self.sequence_length,
            rng,
            timestamps=timestamps,
            force_mirror=transfer_mirror,
            speed_warp=transfer_speed,
            return_plan=is_asl_transfer_config(self.model_config),
        )
        if is_asl_transfer_config(self.model_config):
            view, temporal_plan = temporal_result
        else:
            view = temporal_result
            temporal_plan = None
        model_input = self.standardizer.transform(add_temporal_deltas(view))
        tensor_input: Any = torch.from_numpy(model_input)
        if is_asl_transfer_config(self.model_config):
            from .asl_transfer import prepare_asl_graph_augmented

            pose_hands = pose_hands_for(record, self.processed_dir)
            if len(pose_hands) != len(sequence):
                raise RuntimeError(
                    f"Longitudes multimodales distintas en {record.video_id}: "
                    f"{len(sequence)} != {len(pose_hands)}."
                )
            graph = prepare_asl_graph_augmented(
                pose_hands,
                static_sequence=sequence,
                timestamps=timestamps,
                rng=rng,
                force_mirror=transfer_mirror,
                phase_exponent=transfer_speed,
                frame_bounds=(
                    int(temporal_plan["source_start"]),
                    int(temporal_plan["source_stop"]),
                ),
            )
            tensor_input = (tensor_input, torch.from_numpy(graph))
        return tensor_input, torch.tensor(record.label_index, dtype=torch.long)


def fit_standardizer(
    records: list[VideoRecord],
    processed_dir: Path,
    sequence_length: int,
) -> FeatureStandardizer:
    matrices: list[np.ndarray] = []
    for record in records:
        sequence, timestamps = sequence_for(record, processed_dir)
        for view in deterministic_views(
            sequence,
            sequence_length,
            timestamps=timestamps,
        ):
            matrices.append(add_temporal_deltas(view))
            # La reflexión forma parte de la distribución de entrenamiento; incluirla
            # al ajustar el scaler evita sesgar los canales X hacia una lateralidad.
            matrices.append(add_temporal_deltas(mirror_sequence(view)))
    return FeatureStandardizer.fit(matrices)


def make_folds(
    records: list[VideoRecord],
    n_splits: int,
) -> list[tuple[list[VideoRecord], list[VideoRecord], list[str]]]:
    groups = sorted({record.signer_id for record in records})
    if n_splits < 2:
        raise ValueError("--folds debe ser al menos 2.")
    if n_splits > len(groups):
        raise ValueError(
            f"Solo existen {len(groups)} participantes; --folds no puede ser {n_splits}."
        )
    if n_splits == len(groups):
        validation_groups = [[group] for group in groups]
    else:
        from sklearn.model_selection import GroupKFold

        indices = np.arange(len(records))
        group_array = np.asarray([record.signer_id for record in records])
        splitter = GroupKFold(n_splits=n_splits)
        validation_groups = []
        for _, validation_indices in splitter.split(indices, groups=group_array):
            validation_groups.append(sorted(set(group_array[validation_indices])))

    folds = []
    for held_out in validation_groups:
        held_out_set = set(held_out)
        train = [record for record in records if record.signer_id not in held_out_set]
        validation = [record for record in records if record.signer_id in held_out_set]
        folds.append((train, validation, held_out))
    return folds


def make_inner_folds(
    outer_train_records: list[VideoRecord],
) -> list[tuple[list[VideoRecord], list[VideoRecord], list[str]]]:
    """LOSO interno usando exclusivamente participantes del train externo."""
    signer_count = len({record.signer_id for record in outer_train_records})
    if signer_count < 2:
        raise ValueError("Se necesitan al menos dos participantes para validación interna.")
    return make_folds(outer_train_records, signer_count)


def _class_weights(records: list[VideoRecord], num_classes: int) -> np.ndarray:
    counts = Counter(record.label_index for record in records)
    weights = np.asarray(
        [math.sqrt(len(records) / (num_classes * max(counts.get(index, 0), 1))) for index in range(num_classes)],
        dtype=np.float32,
    )
    return weights / weights.mean()


def predict_records(
    model: SignModel,
    standardizer: FeatureStandardizer,
    records: list[VideoRecord],
    processed_dir: Path,
    sequence_length: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    model.eval()
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device == "cuda",
    ):
        for record in records:
            sequence, timestamps = sequence_for(record, processed_dir)
            matrices = [
                standardizer.transform(add_temporal_deltas(view))
                for view in deterministic_views(
                    sequence,
                    sequence_length,
                    timestamps=timestamps,
                )
            ]
            graph_matrices: list[np.ndarray] | None = None
            if isinstance(model, ASLTransferFusion):
                from .asl_transfer import prepare_asl_graph

                pose_hands = pose_hands_for(record, processed_dir)
                if len(pose_hands) != len(sequence):
                    raise RuntimeError(
                        f"Longitudes multimodales distintas en {record.video_id}."
                    )
                graph_matrices = [
                    prepare_asl_graph(
                        pose_hands,
                        static_sequence=sequence,
                        timestamps=timestamps,
                        phase_exponent=exponent,
                    )
                    for exponent in TEMPORAL_VIEW_EXPONENTS
                ]
                if len(graph_matrices) != len(matrices):
                    raise RuntimeError("Las vistas de landmarks y grafo no coinciden.")
            view_probabilities: list[np.ndarray] = []
            for start in range(0, len(matrices), batch_size):
                batch = torch.from_numpy(np.stack(matrices[start : start + batch_size])).to(device)
                if graph_matrices is None:
                    logits = model(batch)
                else:
                    graph_batch = torch.from_numpy(
                        np.stack(graph_matrices[start : start + batch_size])
                    ).to(device)
                    logits = model(batch, graph_batch)
                probabilities = torch.softmax(logits.float(), dim=-1)
                view_probabilities.append(probabilities.cpu().numpy())
            all_probabilities.append(np.concatenate(view_probabilities).mean(axis=0))
    return np.stack(all_probabilities)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        f1_score,
        top_k_accuracy_score,
    )

    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    present = np.unique(y_true)
    selection_f1 = f1_score(
        y_true,
        y_pred,
        labels=present,
        average="macro",
        zero_division=0,
    )
    top_k = min(3, len(labels))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "selection_macro_f1": float(selection_f1),
        "top3_accuracy": float(
            top_k_accuracy_score(
                y_true,
                probabilities,
                k=top_k,
                labels=np.arange(len(labels)),
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(labels)),
        ).tolist(),
        "predictions": y_pred.tolist(),
    }


def _validate_roc_inputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Valida que las probabilidades OOF permitan curvas ROC OVR completas."""
    true = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=np.float64)
    class_count = len(labels)
    if class_count < 2:
        raise ValueError("ROC OVR requiere al menos dos clases.")
    if len(set(labels)) != class_count:
        raise ValueError("Las etiquetas para ROC deben ser únicas.")
    if true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true para ROC debe ser un vector no vacío.")
    if scores.shape != (len(true), class_count):
        raise ValueError(
            "Las probabilidades para ROC deben tener forma "
            f"({len(true)}, {class_count}), no {scores.shape}."
        )
    if not np.issubdtype(true.dtype, np.integer):
        raise ValueError("y_true para ROC debe contener índices enteros de clase.")
    if np.any(true < 0) or np.any(true >= class_count):
        raise ValueError("y_true contiene un índice de clase fuera de rango.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Las probabilidades para ROC contienen valores no finitos.")
    if np.any(scores < -1e-6) or np.any(scores > 1.0 + 1e-6):
        raise ValueError("Las probabilidades para ROC deben estar en el intervalo [0, 1].")
    if not np.allclose(scores.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("Cada fila de probabilidades para ROC debe sumar 1.")
    for class_index, label in enumerate(labels):
        support = int(np.count_nonzero(true == class_index))
        if support == 0 or support == len(true):
            raise ValueError(
                "No se puede calcular una curva ROC OVR para "
                f"{label}: se requieren ejemplos positivos y negativos."
            )
    return true.astype(np.int64, copy=False), scores


def save_roc_artifacts(
    run_dir: Path,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Guarda curvas ROC OVR y sus AUC a partir de predicciones OOF.

    Las curvas se calculan sobre una sola predicción externa por video. No usan
    el refit final, por lo que sus AUC tienen la misma procedencia que las
    métricas de validación cruzada reportadas en ``cv_metrics.json``.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    true, scores = _validate_roc_inputs(y_true, probabilities, labels)
    run_dir.mkdir(parents=True, exist_ok=True)
    class_count = len(labels)
    binary_true = (true[:, None] == np.arange(class_count)[None, :]).astype(np.int8)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_class: dict[str, dict[str, Any]] = {}
    auc_values: list[float] = []
    supports: list[int] = []

    for class_index, label in enumerate(labels):
        targets = binary_true[:, class_index]
        fpr, tpr, _ = roc_curve(targets, scores[:, class_index])
        auc_ovr = float(roc_auc_score(targets, scores[:, class_index]))
        support = int(targets.sum())
        curves[label] = (fpr, tpr)
        per_class[label] = {
            "auc_ovr": auc_ovr,
            "support": support,
            "negative_support": int(len(targets) - support),
        }
        auc_values.append(auc_ovr)
        supports.append(support)

    micro_fpr, micro_tpr, _ = roc_curve(binary_true.ravel(), scores.ravel())
    micro_auc = float(roc_auc_score(binary_true.ravel(), scores.ravel()))
    common_fpr = np.unique(np.concatenate([curve[0] for curve in curves.values()]))
    macro_tpr = np.mean(
        np.stack(
            [np.interp(common_fpr, fpr, tpr) for fpr, tpr in curves.values()],
            axis=0,
        ),
        axis=0,
    )
    macro_auc = float(np.mean(auc_values))
    weighted_auc = float(np.average(auc_values, weights=supports))
    summary = {
        "evaluation_unit": "video",
        "prediction_source": "out-of-fold predicted probabilities",
        "curve_type": "one-vs-rest",
        "video_count": int(len(true)),
        "class_count": class_count,
        "macro_ovr_auc": macro_auc,
        "weighted_ovr_auc": weighted_auc,
        "micro_ovr_auc": micro_auc,
        "per_class": per_class,
        "artifacts": {
            "figure": "roc_curves.png",
            "per_class_auc": "roc_auc_per_class.csv",
            "summary": "roc_metrics.json",
        },
    }

    with (run_dir / "roc_auc_per_class.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("fruit", "auc_ovr", "support", "negative_support"))
        for label in labels:
            row = per_class[label]
            writer.writerow(
                (label, row["auc_ovr"], row["support"], row["negative_support"])
            )
    write_json(run_dir / "roc_metrics.json", summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(14, 10))
        color_map = plt.get_cmap("tab20", class_count)
        for class_index, label in enumerate(labels):
            fpr, tpr = curves[label]
            row = per_class[label]
            axis.plot(
                fpr,
                tpr,
                color=color_map(class_index),
                linewidth=1.25,
                alpha=0.9,
                label=(
                    f"{label} (AUC={row['auc_ovr']:.3f}, "
                    f"n={row['support']})"
                ),
            )
        axis.plot(
            common_fpr,
            macro_tpr,
            color="navy",
            linestyle="--",
            linewidth=2.6,
            label=f"Macro OVR (AUC={macro_auc:.3f})",
        )
        axis.plot(
            micro_fpr,
            micro_tpr,
            color="black",
            linestyle=":",
            linewidth=2.6,
            label=f"Micro OVR (AUC={micro_auc:.3f})",
        )
        axis.plot((0.0, 1.0), (0.0, 1.0), color="gray", linestyle="--", linewidth=1.0)
        axis.set(
            xlim=(-0.01, 1.0),
            ylim=(0.0, 1.01),
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="Curvas ROC one-vs-rest — predicciones out-of-fold por video",
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        figure.tight_layout()
        figure.savefig(run_dir / "roc_curves.png", dpi=170, bbox_inches="tight")
        plt.close(figure)
    except Exception as exc:
        raise RuntimeError(f"No se pudo generar roc_curves.png: {exc}") from exc

    return summary


def _recommended_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Umbral que separa mejor predicciones correctas de incorrectas en OOF."""
    from sklearn.metrics import f1_score

    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(np.int64)
    if len(np.unique(correct)) < 2:
        return 0.55
    candidates = np.linspace(0.30, 0.90, 121)
    scores = [f1_score(correct, confidence >= threshold, zero_division=0) for threshold in candidates]
    best = float(candidates[int(np.argmax(scores))])
    return float(np.clip(best, 0.40, 0.80))


def _save_metrics_artifacts(
    run_dir: Path,
    metrics: dict[str, Any],
    labels: list[str],
) -> None:
    write_json(run_dir / "cv_metrics.json", metrics)
    with (run_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("fruit", "precision", "recall", "f1", "support"))
        for label in labels:
            row = metrics["per_class"][label]
            writer.writerow((label, row["precision"], row["recall"], row["f1"], row["support"]))
    with (run_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("actual/predicted", *labels))
        matrix_rows = metrics["confusion_matrix"]
        if len(matrix_rows) != len(labels):
            raise ValueError("La matriz de confusión no coincide con las etiquetas.")
        for label, row in zip(labels, matrix_rows):
            writer.writerow((label, *row))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = np.asarray(metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(13, 11))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Etiqueta real")
        axis.set_title("Matriz de confusión — predicciones out-of-fold por video")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, int(matrix[row, column]), ha="center", va="center", fontsize=7)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(run_dir / "confusion_matrix.png", dpi=170)
        plt.close(figure)
    except Exception as exc:
        print(f"Aviso: no se pudo generar confusion_matrix.png: {exc}")


def _build_training_stage(
    records: list[VideoRecord],
    processed_dir: Path,
    labels: list[str],
    sequence_length: int,
    batch_size: int,
    samples_per_video: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    num_workers: int,
    model_config: ModelConfiguration,
    pretrained_checkpoint: Path | None,
) -> tuple[SignModel, FeatureStandardizer, Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    seed_everything(seed)
    standardizer = fit_standardizer(records, processed_dir, sequence_length)
    model = build_model(
        model_config,
        pretrained_checkpoint=pretrained_checkpoint,
    ).to(device)
    weights = torch.from_numpy(_class_weights(records, len(labels))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.06)
    optimizer_parameters: Any = (
        model.optimizer_parameter_groups(learning_rate)
        if isinstance(model, ASLTransferFusion)
        else model.parameters()
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    dataset = AugmentedSequenceDataset(
        records,
        processed_dir,
        sequence_length,
        standardizer,
        samples_per_video,
        seed,
        model_config,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "generator": generator,
    }
    if num_workers > 0:
        # Los workers se recrean cada época para recibir dataset.epoch y producir
        # augmentations distintas pero deterministas en cada pasada.
        loader_options.update(prefetch_factor=2)
    loader = DataLoader(dataset, **loader_options)
    return model, standardizer, optimizer, criterion, scaler, loader


def _train_one_epoch(
    model: SignModel,
    optimizer: Any,
    criterion: Any,
    scaler: Any,
    loader: Any,
    device: str,
    epoch: int,
) -> tuple[float, float]:
    import torch

    if isinstance(model, ASLTransferFusion):
        model.configure_epoch(epoch)
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for inputs, targets in loader:
        if isinstance(inputs, (list, tuple)):
            device_inputs = tuple(
                tensor.to(device, non_blocking=True) for tensor in inputs
            )
        else:
            device_inputs = (inputs.to(device, non_blocking=True),)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device == "cuda",
        ):
            logits = model(*device_inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(targets)
        total_examples += len(targets)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
    return (
        total_loss / max(total_examples, 1),
        correct / max(total_examples, 1),
    )


def _fit_inner_selection(
    train_records: list[VideoRecord],
    validation_records: list[VideoRecord],
    processed_dir: Path,
    labels: list[str],
    sequence_length: int,
    epochs: int,
    batch_size: int,
    samples_per_video: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    num_workers: int,
    model_config: ModelConfiguration,
    pretrained_checkpoint: Path | None,
    stage_name: str,
) -> dict[str, Any]:
    model, standardizer, optimizer, criterion, scaler, loader = _build_training_stage(
        train_records,
        processed_dir,
        labels,
        sequence_length,
        batch_size,
        samples_per_video,
        learning_rate,
        weight_decay,
        seed,
        device,
        num_workers,
        model_config,
        pretrained_checkpoint,
    )
    validation_true = np.asarray(
        [record.label_index for record in validation_records],
        dtype=np.int64,
    )
    best_score = -math.inf
    best_epoch = 1
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss, train_accuracy = _train_one_epoch(
            model,
            optimizer,
            criterion,
            scaler,
            loader,
            device,
            epoch,
        )
        probabilities = predict_records(
            model,
            standardizer,
            validation_records,
            processed_dir,
            sequence_length,
            device,
            batch_size,
        )
        validation_metrics = compute_metrics(validation_true, probabilities, labels)
        score = float(validation_metrics["selection_macro_f1"])
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": float(validation_metrics["accuracy"]),
            "val_macro_f1": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "y_true": validation_true.tolist(),
            "predictions": probabilities.argmax(axis=1).tolist(),
        }
        history.append(row)
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
        print(
            f"    {stage_name} epoch {epoch:03d} | loss {loss:.4f} | "
            f"train acc {train_accuracy:.3f} | val acc {row['val_accuracy']:.3f} | "
            f"val F1 {score:.3f}"
        )
    return {
        "validation_protocol": VALIDATION_PROTOCOL,
        "train_signers": sorted({record.signer_id for record in train_records}),
        "validation_signers": sorted(
            {record.signer_id for record in validation_records}
        ),
        "train_videos": len(train_records),
        "validation_videos": len(validation_records),
        "best_epoch": best_epoch,
        "best_selection_macro_f1": best_score,
        "history": history,
    }


def select_epoch_from_inner_histories(
    selections: list[dict[str, Any]],
) -> tuple[int, list[dict[str, float]]]:
    """Escoge época por predicciones OOF internas agrupadas, sin promediar folds."""
    from sklearn.metrics import accuracy_score, f1_score

    if not selections or any(not item.get("history") for item in selections):
        raise ValueError("No hay historiales internos suficientes para escoger época.")
    lengths = {len(item["history"]) for item in selections}
    if len(lengths) != 1:
        raise ValueError(
            "Todos los subfolds internos deben recorrer la misma cantidad de épocas."
        )
    common_length = lengths.pop()
    curve: list[dict[str, float]] = []
    for index in range(common_length):
        y_true = np.concatenate(
            [np.asarray(item["history"][index]["y_true"]) for item in selections]
        )
        y_pred = np.concatenate(
            [np.asarray(item["history"][index]["predictions"]) for item in selections]
        )
        fold_scores = [
            float(item["history"][index]["val_macro_f1"])
            for item in selections
        ]
        curve.append(
            {
                "epoch": index + 1,
                "pooled_inner_macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "pooled_inner_accuracy": float(accuracy_score(y_true, y_pred)),
                "mean_fold_macro_f1": float(np.mean(fold_scores)),
                "std_fold_macro_f1": float(np.std(fold_scores)),
            }
        )
    # Desempate fijado de antemano: mayor F1, luego accuracy y época más temprana.
    selected_index = max(
        range(len(curve)),
        key=lambda index: (
            curve[index]["pooled_inner_macro_f1"],
            curve[index]["pooled_inner_accuracy"],
            -curve[index]["epoch"],
        ),
    )
    return selected_index + 1, curve


def _fit_fixed_epochs(
    records: list[VideoRecord],
    processed_dir: Path,
    labels: list[str],
    sequence_length: int,
    epochs: int,
    batch_size: int,
    samples_per_video: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    num_workers: int,
    model_config: ModelConfiguration,
    pretrained_checkpoint: Path | None,
    stage_name: str,
) -> tuple[SignModel, FeatureStandardizer, Any, list[dict[str, float]]]:
    model, standardizer, optimizer, criterion, scaler, loader = _build_training_stage(
        records,
        processed_dir,
        labels,
        sequence_length,
        batch_size,
        samples_per_video,
        learning_rate,
        weight_decay,
        seed,
        device,
        num_workers,
        model_config,
        pretrained_checkpoint,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss, train_accuracy = _train_one_epoch(
            model,
            optimizer,
            criterion,
            scaler,
            loader,
            device,
            epoch,
        )
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_accuracy": train_accuracy,
        }
        history.append(row)
        print(
            f"    {stage_name} epoch {epoch:03d} | loss {loss:.4f} | "
            f"train acc {train_accuracy:.3f}"
        )
    return model, standardizer, optimizer, history


def _fit_final_model(
    records: list[VideoRecord],
    processed_dir: Path,
    run_dir: Path,
    labels: list[str],
    manifest: dict[str, Any],
    sequence_length: int,
    epochs: int,
    batch_size: int,
    samples_per_video: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    num_workers: int,
    cv_metrics: dict[str, Any],
    run_fingerprint: str,
    dataset_content_fingerprint: str,
    runtime: dict[str, Any],
    model_config: ModelConfiguration,
    pretrained_checkpoint: Path | None,
    implementation_version: str,
    resume: bool,
) -> Path:
    """Reajusta con todos los videos usando épocas escogidas sin tocar test externo."""
    checkpoint_path = run_dir / "final_model.pt"
    complete_path = run_dir / "final_complete.json"
    if resume and checkpoint_path.exists() and complete_path.exists():
        _, _, payload = load_checkpoint(checkpoint_path, device)
        preprocessing = payload.get("preprocessing", {})
        completion = read_json(complete_path)
        reusable = (
            preprocessing.get("validation_protocol") == VALIDATION_PROTOCOL
            and preprocessing.get("run_fingerprint") == run_fingerprint
            and preprocessing.get("dataset_content_fingerprint")
            == dataset_content_fingerprint
            and preprocessing.get("runtime", {}).get("device") == runtime["device"]
            and preprocessing.get("runtime", {}).get("mixed_precision")
            == runtime["mixed_precision"]
            and preprocessing.get("role") == "full_data_refit"
            and payload.get("model_config") == asdict(model_config)
            and int(payload.get("epoch", -1)) == epochs
            and completion.get("status") == "complete"
            and completion.get("validation_protocol") == VALIDATION_PROTOCOL
            and completion.get("run_fingerprint") == run_fingerprint
        )
        if reusable:
            print(f"Reutilizando modelo final: {checkpoint_path}")
            return checkpoint_path
        print(
            "El modelo final no coincide íntegramente con esta corrida; "
            "se reajustará desde cero."
        )
    if resume and checkpoint_path.exists() and not complete_path.exists():
        print("Modelo final incompleto detectado; se reajustará desde cero.")

    print(
        f"\nReajuste final | train={len(records)} videos | "
        f"épocas escogidas por CV anidada={epochs}"
    )
    model, standardizer, optimizer, history = _fit_fixed_epochs(
        records,
        processed_dir,
        labels,
        sequence_length,
        epochs,
        batch_size,
        samples_per_video,
        learning_rate,
        weight_decay,
        seed + 10_007,
        device,
        num_workers,
        model_config,
        pretrained_checkpoint,
        "final",
    )
    training_probabilities = predict_records(
        model,
        standardizer,
        records,
        processed_dir,
        sequence_length,
        device,
        batch_size,
    )
    training_true = np.asarray([record.label_index for record in records], dtype=np.int64)
    training_accuracy = float(
        np.mean(training_probabilities.argmax(axis=1) == training_true)
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        standardizer,
        labels,
        sequence_length,
        -1,
        epochs,
        {
            "training_accuracy": training_accuracy,
            "cv_accuracy_reference": cv_metrics["accuracy"],
            "note": "Training accuracy es diagnóstica; la generalización se mide con OOF externo.",
        },
        {
            "target_fps": manifest["target_fps"],
            "max_side": manifest["max_side"],
            "schema_version": manifest["schema_version"],
            "feature_layout": feature_layout(),
            **(
                {
                    "transfer_contract": manifest["transfer_contract"],
                    "transfer_layout": manifest["transfer_layout"],
                }
                if is_asl_transfer_config(model_config)
                else {}
            ),
            "model_config": asdict(model_config),
            "pretraining": getattr(model, "pretraining_metadata", None),
            "role": "full_data_refit",
            "validation_protocol": VALIDATION_PROTOCOL,
            "implementation_version": implementation_version,
            "run_fingerprint": run_fingerprint,
            "dataset_content_fingerprint": dataset_content_fingerprint,
            "runtime": runtime,
        },
    )
    write_json(run_dir / "final_history.json", history)
    return checkpoint_path


def _training_fingerprint(
    manifest: dict[str, Any],
    records: list[VideoRecord],
    configuration: dict[str, Any],
    model_config: ModelConfiguration,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "validation_protocol": VALIDATION_PROTOCOL,
        "schema_version": manifest["schema_version"],
        "feature_dim": manifest["feature_dim"],
        "feature_layout": manifest["feature_layout"],
        "target_fps": manifest["target_fps"],
        "max_side": manifest["max_side"],
        "include_excluded": bool(manifest.get("include_excluded", False)),
        "excluded_videos": manifest.get("excluded_videos", []),
        "labels": manifest["labels"],
        "model_config": asdict(model_config),
        "training": configuration,
        "videos": [
            {
                "source": record.source,
                "video_id": record.video_id,
                "label": record.label,
                "label_index": record.label_index,
                "signer_id": record.signer_id,
                "tags": record.tags,
                "excluded": record.excluded,
                "sequence": record.sequence,
                "frames": record.frames,
                "duration": record.duration,
                "source_size": record.source_size,
                "source_mtime_ns": record.source_mtime_ns,
            }
            for record in sorted(records, key=lambda item: item.video_id)
        ],
    }
    if manifest.get("transfer_contract") is not None:
        payload["transfer_contract"] = manifest.get("transfer_contract")
        payload["transfer_layout"] = manifest.get("transfer_layout")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def dataset_content_fingerprint(
    processed_dir: Path,
    manifest: dict[str, Any],
    records: list[VideoRecord],
) -> tuple[str, dict[str, Any]]:
    """Huella reproducible del contrato de extracción y de cada secuencia real."""
    sequences: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.video_id):
        path = processed_dir / record.sequence
        if not path.is_file():
            raise FileNotFoundError(f"No existe la secuencia procesada: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        sequences.append(
            {
                "video_id": record.video_id,
                "sequence": record.sequence,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    payload = {
        "schema_version": manifest["schema_version"],
        "feature_dim": manifest["feature_dim"],
        "feature_layout": manifest["feature_layout"],
        "target_fps": manifest["target_fps"],
        "max_side": manifest["max_side"],
        "include_excluded": bool(manifest.get("include_excluded", False)),
        "labels": manifest["labels"],
        "sequences": sequences,
    }
    if manifest.get("transfer_contract") is not None:
        payload["transfer_contract"] = manifest.get("transfer_contract")
        payload["transfer_layout"] = manifest.get("transfer_layout")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _require_all_classes(
    records: list[VideoRecord],
    labels: list[str],
    stage_name: str,
) -> None:
    present = {record.label_index for record in records}
    missing = [label for index, label in enumerate(labels) if index not in present]
    if missing:
        raise RuntimeError(
            f"{stage_name} no contiene todas las clases. Faltan: {', '.join(missing)}"
        )


def _validate_disjoint(
    first: list[VideoRecord],
    second: list[VideoRecord],
    description: str,
) -> None:
    first_ids = {record.video_id for record in first}
    second_ids = {record.video_id for record in second}
    overlap = first_ids & second_ids
    if overlap:
        raise RuntimeError(
            f"Fuga de datos en {description}: {len(overlap)} videos compartidos."
        )
    first_signers = {record.signer_id for record in first}
    second_signers = {record.signer_id for record in second}
    signer_overlap = first_signers & second_signers
    if signer_overlap:
        raise RuntimeError(
            f"Fuga de participantes en {description}: {sorted(signer_overlap)}"
        )


def _runtime_metadata(device: str) -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {
        "device": device,
        "mixed_precision": "float16" if device == "cuda" else "float32",
        "torch_version": torch.__version__,
    }
    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu_index": index,
                "gpu_name": properties.name,
                "gpu_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return metadata


def train_cross_validation(
    processed_dir: Path,
    run_dir: Path,
    folds: int = 5,
    sequence_length: int = 48,
    epochs: int = 30,
    batch_size: int = 32,
    samples_per_video: int = 4,
    learning_rate: float = 8e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device_request: str = "cuda",
    num_workers: int = 2,
    model_preset: str = DEFAULT_MODEL_PRESET,
    pretrained_checkpoint: Path | None = None,
    only_fold: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Nested LOSO:

    1. Un participante externo se reserva como test y no se consulta al seleccionar.
    2. Los participantes restantes forman subfolds LOSO internos para escoger época.
    3. Se reajusta desde cero con todo el train externo durante esa época.
    4. El test externo se evalúa una única vez después del reajuste.
    """
    seed_everything(seed)
    device = resolve_device(device_request)
    runtime = _runtime_metadata(device)
    print(
        "Backend de entrenamiento: "
        f"{runtime.get('gpu_name', device)} | device={device} | "
        f"precision={runtime['mixed_precision']}"
    )
    manifest, records = load_manifest(processed_dir)
    labels = manifest["labels"]
    model_config = model_config_for_preset(
        input_dim=model_feature_dim(),
        num_classes=len(labels),
        preset=model_preset,
    )
    implementation_version = (
        TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION
        if is_asl_transfer_config(model_config)
        else EXPERIMENT_IMPLEMENTATION_VERSION
    )
    pretraining: dict[str, Any] | None = None
    if is_asl_transfer_config(model_config):
        from .asl_transfer import (
            ASL_CITIZEN_CHECKPOINT_SHA256,
            ASL_GRAPH_CONTRACT,
        )

        if (
            manifest.get("transfer_contract") != POSE_HANDS_XY_LAYOUT_VERSION
            or manifest.get("transfer_layout") != transfer_feature_layout()
            or manifest.get("status") != "complete"
            or bool(manifest.get("limited_for_testing", False))
            or manifest.get("max_videos") is not None
            or int(manifest.get("inventory_video_count", -1)) != len(records)
            or int(manifest.get("expected_video_count", -1)) != len(records)
            or int(manifest.get("processed_video_count", -1)) != len(records)
            or bool(manifest.get("failures"))
        ):
            raise RuntimeError(
                "El dataset no contiene un caché ASL exacto y completo. Ejecuta "
                "scripts/05_prepare_asl_transfer.py hasta que termine sin fallos."
            )
        if pretrained_checkpoint is None:
            raise ValueError(
                "El preset ASL requiere --pretrained-checkpoint. Ejecuta antes "
                "scripts/06_download_asl_citizen_weights.py."
            )
        pretrained_checkpoint = pretrained_checkpoint.resolve()
        if not pretrained_checkpoint.is_file():
            raise FileNotFoundError(
                f"No existe el checkpoint preentrenado: {pretrained_checkpoint}"
            )
        digest = hashlib.sha256()
        with pretrained_checkpoint.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        checkpoint_sha256 = digest.hexdigest()
        if checkpoint_sha256 != ASL_CITIZEN_CHECKPOINT_SHA256:
            raise RuntimeError(
                "SHA256 inesperado para el checkpoint ASL Citizen: "
                f"{checkpoint_sha256}."
            )
        pretraining = {
            "provider": "Microsoft ASL Citizen",
            "architecture": "ST-GCN",
            "source_modality": "MediaPipe Holistic pose+hands xy",
            "source_classes": 2731,
            "classifier_reused": False,
            "checkpoint_file": pretrained_checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "cache_layout_version": POSE_HANDS_XY_LAYOUT_VERSION,
            "input_contract": ASL_GRAPH_CONTRACT,
            "target_adaptation": {
                "active_segment": "shared mask_aware_motion_v2 bounds",
                "deterministic_phase_exponents": list(TEMPORAL_VIEW_EXPONENTS),
                "training_phase_range": list(TRAIN_SPEED_WARP_RANGE),
                "training_mirror_probability": TRAIN_MIRROR_PROBABILITY,
                "mirror_synchronized_with_landmark_branch": True,
            },
        }
    elif pretrained_checkpoint is not None:
        raise ValueError(
            "--pretrained-checkpoint solo corresponde al preset de transferencia ASL."
        )
    if len(records) < len(labels) * 2:
        raise RuntimeError("Hay muy pocos videos procesados para entrenar con folds.")
    signer_count = len({record.signer_id for record in records})
    if folds != signer_count:
        raise ValueError(
            "Este protocolo es leave-one-signer-out: --folds debe coincidir con "
            f"los {signer_count} participantes detectados."
        )
    outer_folds = make_folds(records, folds)
    if only_fold is not None and not 0 <= only_fold < len(outer_folds):
        raise ValueError(
            f"--only-fold debe estar entre 0 y {len(outer_folds) - 1}."
        )

    fingerprint_configuration = {
        "folds": folds,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "samples_per_video": samples_per_video,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "num_workers": num_workers,
        "model_preset": model_preset,
        "model_config": asdict(model_config),
        "feature_layout": feature_layout(),
        "device": device,
        "mixed_precision": runtime["mixed_precision"],
    }
    if pretraining is not None:
        fingerprint_configuration["pretraining"] = pretraining
    fingerprint, fingerprint_payload = _training_fingerprint(
        manifest,
        records,
        fingerprint_configuration,
        model_config,
    )
    dataset_fingerprint, dataset_fingerprint_payload = dataset_content_fingerprint(
        processed_dir,
        manifest,
        records,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = run_dir / "training_config.json"
    if resume and configuration_path.exists():
        existing_configuration = read_json(configuration_path)
        if (
            existing_configuration.get("implementation_version")
            != implementation_version
        ):
            raise RuntimeError(
                "La implementación del experimento cambió; no es seguro reutilizar "
                "checkpoints con --resume. Usa otro --run-dir."
            )
        existing_fingerprint = existing_configuration.get("run_fingerprint")
        if existing_fingerprint != fingerprint:
            raise RuntimeError(
                "La configuración o el dataset cambiaron; no es seguro mezclar "
                "checkpoints con --resume. Usa otro --run-dir."
            )
        existing_dataset_fingerprint = existing_configuration.get(
            "dataset_content_fingerprint"
        )
        if (
            existing_dataset_fingerprint is not None
            and existing_dataset_fingerprint != dataset_fingerprint
        ):
            raise RuntimeError(
                "Las secuencias procesadas cambiaron; no es seguro reutilizar "
                "checkpoints con --resume. Usa otro --run-dir."
            )
        if existing_dataset_fingerprint is None:
            raise RuntimeError(
                "La corrida existente no tiene una huella del contenido procesado; "
                "no puede reanudarse de forma segura. Usa otro --run-dir."
            )
    run_configuration = {
        **fingerprint_configuration,
        "implementation_version": implementation_version,
        "processed_dir": str(processed_dir.resolve()),
        "runtime": runtime,
        "model_preset": model_preset,
        "model_config": asdict(model_config),
        "feature_layout": feature_layout(),
        "labels": labels,
        "validation_protocol": VALIDATION_PROTOCOL,
        "inner_validation": "leave-one-signer-out sobre todo el train externo",
        "epoch_selection": "F1 macro pooled de predicciones OOF internas",
        "run_fingerprint": fingerprint,
        "dataset_content_fingerprint": dataset_fingerprint,
    }
    if pretrained_checkpoint is not None:
        run_configuration["pretrained_checkpoint_path"] = str(
            pretrained_checkpoint
        )
    write_json(configuration_path, run_configuration)
    write_json(run_dir / "run_fingerprint_payload.json", fingerprint_payload)
    write_json(
        run_dir / "dataset_fingerprint_payload.json",
        dataset_fingerprint_payload,
    )

    # Materializar y auditar todos los splits antes de entrenar.
    split_manifest: list[dict[str, Any]] = []
    for outer_index, (outer_train, outer_test, outer_test_signers) in enumerate(outer_folds):
        _validate_disjoint(outer_train, outer_test, f"outer fold {outer_index}")
        _require_all_classes(outer_train, labels, f"outer train {outer_index}")
        inner_manifest: list[dict[str, Any]] = []
        outer_train_ids = {record.video_id for record in outer_train}
        inner_validation_ids: set[str] = set()
        for inner_index, (inner_train, inner_validation, inner_signers) in enumerate(
            make_inner_folds(outer_train)
        ):
            _validate_disjoint(
                inner_train,
                inner_validation,
                f"outer {outer_index} / inner {inner_index}",
            )
            _require_all_classes(
                inner_train,
                labels,
                f"outer {outer_index} / inner train {inner_index}",
            )
            inner_validation_ids.update(record.video_id for record in inner_validation)
            inner_manifest.append(
                {
                    "inner_fold": inner_index,
                    "validation_signers": inner_signers,
                    "train_video_ids": [record.video_id for record in inner_train],
                    "validation_video_ids": [
                        record.video_id for record in inner_validation
                    ],
                }
            )
        if inner_validation_ids != outer_train_ids:
            raise RuntimeError(
                f"Los subfolds internos del outer {outer_index} no cubren exactamente su train."
            )
        split_manifest.append(
            {
                "outer_fold": outer_index,
                "test_signers": outer_test_signers,
                "train_signers": sorted(
                    {record.signer_id for record in outer_train}
                ),
                "train_video_ids": [record.video_id for record in outer_train],
                "test_video_ids": [record.video_id for record in outer_test],
                "inner_folds": inner_manifest,
            }
        )
    write_json(
        run_dir / "splits.json",
        {
            "validation_protocol": VALIDATION_PROTOCOL,
            "implementation_version": implementation_version,
            "run_fingerprint": fingerprint,
            "dataset_content_fingerprint": dataset_fingerprint,
            "runtime": runtime,
            "outer_folds": split_manifest,
        },
    )

    all_true: list[int] = []
    all_probabilities: list[np.ndarray] = []
    all_video_ids: list[str] = []
    checkpoints: list[str] = []
    fold_summaries: list[dict[str, Any]] = []

    for outer_index, (outer_train, outer_test, outer_test_signers) in enumerate(outer_folds):
        if only_fold is not None and outer_index != only_fold:
            continue
        fold_seed = seed + outer_index * 10_000
        fold_dir = run_dir / f"fold_{outer_index}"
        checkpoint_path = fold_dir / "best.pt"
        complete_path = fold_dir / "complete.json"
        selection_path = fold_dir / "selection.json"
        outer_train_signers = sorted({record.signer_id for record in outer_train})
        print(
            f"\nOuter fold {outer_index + 1}/{len(outer_folds)} | "
            f"test aislado={','.join(outer_test_signers)} | "
            f"dev={','.join(outer_train_signers)}"
        )

        reused_outer = resume and checkpoint_path.exists() and complete_path.exists()
        if reused_outer:
            model, standardizer, payload = load_checkpoint(checkpoint_path, device)
            preprocessing = payload.get("preprocessing", {})
            if (
                preprocessing.get("validation_protocol") != VALIDATION_PROTOCOL
                or preprocessing.get("run_fingerprint") != fingerprint
                or preprocessing.get("dataset_content_fingerprint")
                != dataset_fingerprint
                or preprocessing.get("runtime", {}).get("device") != device
                or preprocessing.get("runtime", {}).get("mixed_precision")
                != runtime["mixed_precision"]
                or preprocessing.get("role") != "nested_outer_refit"
                or preprocessing.get("outer_test_signers") != outer_test_signers
            ):
                raise RuntimeError(
                    f"El checkpoint {checkpoint_path} no coincide con este fold anidado."
                )
            selected_epoch = int(payload["epoch"])
            selection_summary = read_json(selection_path)
            refit_history = read_json(fold_dir / "refit_history.json")
            print(f"  Reutilizando outer fold completo: {checkpoint_path}")
        else:
            inner_selections: list[dict[str, Any]] = []
            for inner_index, (inner_train, inner_validation, inner_signers) in enumerate(
                make_inner_folds(outer_train)
            ):
                inner_dir = fold_dir / f"inner_{inner_index}_{inner_signers[0]}"
                inner_path = inner_dir / "selection.json"
                if resume and inner_path.exists():
                    selection = read_json(inner_path)
                    if (
                        selection.get("status") != "complete"
                        or selection.get("validation_protocol") != VALIDATION_PROTOCOL
                        or selection.get("run_fingerprint") != fingerprint
                        or selection.get("outer_test_signers") != outer_test_signers
                        or selection.get("validation_signers") != inner_signers
                    ):
                        raise RuntimeError(
                            f"La selección interna {inner_path} no coincide con la corrida."
                        )
                    print(
                        f"  Reutilizando inner {inner_index + 1}: "
                        f"val={','.join(inner_signers)}"
                    )
                else:
                    print(
                        f"  Inner {inner_index + 1}/{len(outer_train_signers)} | "
                        f"val={','.join(inner_signers)} | "
                        f"train={len(inner_train)} videos | "
                        f"val={len(inner_validation)} videos"
                    )
                    selection = _fit_inner_selection(
                        inner_train,
                        inner_validation,
                        processed_dir,
                        labels,
                        sequence_length,
                        epochs,
                        batch_size,
                        samples_per_video,
                        learning_rate,
                        weight_decay,
                        fold_seed + (inner_index + 1) * 101,
                        device,
                        num_workers,
                        model_config,
                        pretrained_checkpoint,
                        f"inner {inner_index + 1}",
                    )
                    selection.update(
                        {
                            "status": "complete",
                            "implementation_version": implementation_version,
                            "run_fingerprint": fingerprint,
                            "outer_fold": outer_index,
                            "outer_test_signers": outer_test_signers,
                            "inner_fold": inner_index,
                        }
                    )
                    write_json(inner_path, selection)
                inner_selections.append(selection)

            selected_epoch, pooled_curve = select_epoch_from_inner_histories(
                inner_selections
            )
            selection_summary = {
                "status": "complete",
                "validation_protocol": VALIDATION_PROTOCOL,
                "implementation_version": implementation_version,
                "run_fingerprint": fingerprint,
                "outer_fold": outer_index,
                "outer_test_signers": outer_test_signers,
                "outer_train_signers": outer_train_signers,
                "selected_epoch": selected_epoch,
                "selection_rule": (
                    "máximo F1 macro pooled OOF interno; desempate por accuracy y época temprana"
                ),
                "pooled_curve": pooled_curve,
                "inner_folds": [
                    {
                        key: value
                        for key, value in selection.items()
                        if key != "history"
                    }
                    for selection in inner_selections
                ],
            }
            write_json(selection_path, selection_summary)
            selected_row = pooled_curve[selected_epoch - 1]
            print(
                f"  Época seleccionada sin consultar test externo: {selected_epoch} "
                f"(inner pooled F1={selected_row['pooled_inner_macro_f1']:.3f})"
            )
            print(
                f"  Refit outer con {len(outer_train)} videos durante "
                f"{selected_epoch} épocas"
            )
            model, standardizer, optimizer, refit_history = _fit_fixed_epochs(
                outer_train,
                processed_dir,
                labels,
                sequence_length,
                selected_epoch,
                batch_size,
                samples_per_video,
                learning_rate,
                weight_decay,
                fold_seed + 9_001,
                device,
                num_workers,
                model_config,
                pretrained_checkpoint,
                "outer refit",
            )
            write_json(fold_dir / "refit_history.json", refit_history)

        # Primera y única consulta al test externo durante una ejecución nueva.
        probabilities = predict_records(
            model,
            standardizer,
            outer_test,
            processed_dir,
            sequence_length,
            device,
            batch_size,
        )
        y_true = np.asarray(
            [record.label_index for record in outer_test],
            dtype=np.int64,
        )
        fold_metrics = compute_metrics(y_true, probabilities, labels)
        if not reused_outer:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                standardizer,
                labels,
                sequence_length,
                outer_index,
                selected_epoch,
                fold_metrics,
                {
                    "target_fps": manifest["target_fps"],
                    "max_side": manifest["max_side"],
                    "schema_version": manifest["schema_version"],
                    "feature_layout": feature_layout(),
                    **(
                        {
                            "transfer_contract": manifest["transfer_contract"],
                            "transfer_layout": manifest["transfer_layout"],
                        }
                        if is_asl_transfer_config(model_config)
                        else {}
                    ),
                    "model_config": asdict(model_config),
                    "pretraining": getattr(model, "pretraining_metadata", None),
                    "role": "nested_outer_refit",
                    "validation_protocol": VALIDATION_PROTOCOL,
                    "implementation_version": implementation_version,
                    "run_fingerprint": fingerprint,
                    "dataset_content_fingerprint": dataset_fingerprint,
                    "runtime": runtime,
                    "outer_test_signers": outer_test_signers,
                    "outer_train_signers": outer_train_signers,
                    "selected_epoch": selected_epoch,
                },
            )

        fold_summary = {
            "status": "complete",
            "validation_protocol": VALIDATION_PROTOCOL,
            "implementation_version": implementation_version,
            "run_fingerprint": fingerprint,
            "dataset_content_fingerprint": dataset_fingerprint,
            "fold": outer_index,
            "outer_test_signers": outer_test_signers,
            "outer_train_signers": outer_train_signers,
            "outer_train_videos": len(outer_train),
            "outer_test_videos": len(outer_test),
            "selected_epoch": selected_epoch,
            "selection": selection_summary,
            "metrics": fold_metrics,
        }
        write_json(fold_dir / "metrics.json", fold_summary)
        np.savez_compressed(
            fold_dir / "test_predictions.npz",
            y_true=y_true,
            probabilities=probabilities,
            video_ids=np.asarray([record.video_id for record in outer_test]),
        )
        write_json(
            complete_path,
            {
                "status": "complete",
                "validation_protocol": VALIDATION_PROTOCOL,
                "implementation_version": implementation_version,
                "run_fingerprint": fingerprint,
                "dataset_content_fingerprint": dataset_fingerprint,
                "runtime": runtime,
                "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            },
        )
        fold_summaries.append(fold_summary)
        checkpoints.append(str(checkpoint_path.relative_to(run_dir)))
        all_true.extend(y_true.tolist())
        all_probabilities.extend(probabilities)
        all_video_ids.extend(record.video_id for record in outer_test)

    if only_fold is not None:
        result = {
            "status": "partial",
            "validation_protocol": VALIDATION_PROTOCOL,
            "implementation_version": implementation_version,
            "run_fingerprint": fingerprint,
            "folds": fold_summaries,
            "checkpoints": checkpoints,
        }
        write_json(run_dir / "partial_metrics.json", result)
        return result
    if len(fold_summaries) != len(outer_folds):
        raise RuntimeError("No se completaron todos los folds externos.")
    expected_video_ids = sorted(record.video_id for record in records)
    if sorted(all_video_ids) != expected_video_ids or len(set(all_video_ids)) != len(records):
        raise RuntimeError(
            "Las predicciones OOF externas no cubren cada video exactamente una vez."
        )

    y_true_array = np.asarray(all_true, dtype=np.int64)
    probabilities_array = np.asarray(all_probabilities, dtype=np.float32)
    metrics = compute_metrics(y_true_array, probabilities_array, labels)
    metrics.update(
        {
            "evaluation_unit": "video",
            "validation": (
                "nested leave-one-signer-out: inner LOSO epoch selection, "
                "outer refit, single outer-test evaluation"
            ),
            "validation_protocol": VALIDATION_PROTOCOL,
            "implementation_version": implementation_version,
            "run_fingerprint": fingerprint,
            "dataset_content_fingerprint": dataset_fingerprint,
            "runtime": runtime,
            "video_count": len(y_true_array),
            "recommended_confidence_threshold": _recommended_threshold(
                y_true_array,
                probabilities_array,
            ),
            "fold_summary": [
                {
                    "fold": item["fold"],
                    "outer_test_signers": item["outer_test_signers"],
                    "selected_epoch": item["selected_epoch"],
                    "accuracy": item["metrics"]["accuracy"],
                    "macro_f1": item["metrics"]["macro_f1"],
                }
                for item in fold_summaries
            ],
        }
    )
    np.savez_compressed(
        run_dir / "oof_predictions.npz",
        y_true=y_true_array,
        probabilities=probabilities_array,
        video_ids=np.asarray(all_video_ids),
    )
    metrics["roc_auc"] = save_roc_artifacts(
        run_dir,
        y_true_array,
        probabilities_array,
        labels,
    )
    _save_metrics_artifacts(run_dir, metrics, labels)
    selected_epochs = max(
        1,
        int(
            round(
                float(
                    np.median([item["selected_epoch"] for item in fold_summaries])
                )
            )
        ),
    )
    final_checkpoint = _fit_final_model(
        records=records,
        processed_dir=processed_dir,
        run_dir=run_dir,
        labels=labels,
        manifest=manifest,
        sequence_length=sequence_length,
        epochs=selected_epochs,
        batch_size=batch_size,
        samples_per_video=samples_per_video,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        num_workers=num_workers,
        cv_metrics=metrics,
        run_fingerprint=fingerprint,
        dataset_content_fingerprint=dataset_fingerprint,
        runtime=runtime,
        model_config=model_config,
        pretrained_checkpoint=pretrained_checkpoint,
        implementation_version=implementation_version,
        resume=resume,
    )
    write_json(
        run_dir / "final_complete.json",
        {
            "status": "complete",
            "validation_protocol": VALIDATION_PROTOCOL,
            "implementation_version": implementation_version,
            "run_fingerprint": fingerprint,
            "dataset_content_fingerprint": dataset_fingerprint,
            "runtime": runtime,
            "checkpoint": str(final_checkpoint.relative_to(run_dir)),
        },
    )
    ensemble_checkpoints = [*checkpoints, str(final_checkpoint.relative_to(run_dir))]
    ensemble = {
        "labels": labels,
        "checkpoints": ensemble_checkpoints,
        "fold_checkpoints": checkpoints,
        "final_checkpoint": str(final_checkpoint.relative_to(run_dir)),
        "final_refit_epochs": selected_epochs,
        "sequence_length": sequence_length,
        "model_preset": model_preset,
        "model_config": asdict(model_config),
        "pretraining": pretraining,
        "feature_layout": feature_layout(),
        "target_fps": manifest["target_fps"],
        "max_side": manifest["max_side"],
        "recommended_confidence_threshold": metrics[
            "recommended_confidence_threshold"
        ],
        "cv_metrics": "cv_metrics.json",
        "validation_protocol": VALIDATION_PROTOCOL,
        "implementation_version": implementation_version,
        "run_fingerprint": fingerprint,
        "dataset_content_fingerprint": dataset_fingerprint,
        "runtime": runtime,
    }
    if is_asl_transfer_config(model_config):
        ensemble["transfer_contract"] = manifest["transfer_contract"]
        ensemble["transfer_layout"] = manifest["transfer_layout"]
    write_json(run_dir / "ensemble.json", ensemble)
    print(
        "\nValidación anidada terminada | "
        f"accuracy={metrics['accuracy']:.3f} | "
        f"precision macro={metrics['macro_precision']:.3f} | "
        f"F1 macro={metrics['macro_f1']:.3f}"
    )
    return metrics
