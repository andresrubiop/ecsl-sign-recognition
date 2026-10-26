"""Etapa 3: recalcula las métricas OOF generales o de una fruta."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pfrutas.data import load_manifest
from pfrutas.training import (
    SUPPORTED_EXPERIMENT_IMPLEMENTATION_VERSIONS,
    compute_metrics,
    dataset_content_fingerprint,
    save_roc_artifacts,
)
from pfrutas.utils import read_json, resolve_label, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed_v2"),
        help="Dataset v2 contra el que se validan IDs y huellas.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/lstm_compact_v2_gpu"),
        help="Corrida v2 cuyas predicciones OOF y huellas se verificarán.",
    )
    parser.add_argument("--fruit", default=None)
    parser.add_argument(
        "--write-roc-artifacts",
        action="store_true",
        help=(
            "Regenerar roc_curves.png, roc_auc_per_class.csv y roc_metrics.json "
            "a partir de las probabilidades OOF verificadas."
        ),
    )
    args = parser.parse_args()

    manifest, records = load_manifest(args.processed_dir)
    ensemble = read_json(args.run_dir / "ensemble.json")
    training_config = read_json(args.run_dir / "training_config.json")
    labels = ensemble["labels"]
    if labels != manifest["labels"]:
        raise RuntimeError("Las clases del modelo y del dataset no coinciden.")
    with np.load(args.run_dir / "oof_predictions.npz", allow_pickle=False) as payload:
        y_true = payload["y_true"]
        probabilities = payload["probabilities"]
        video_ids = payload["video_ids"]
    if (
        training_config.get("implementation_version")
        not in SUPPORTED_EXPERIMENT_IMPLEMENTATION_VERSIONS
    ):
        raise RuntimeError("La corrida pertenece a otra versión de la implementación.")
    current_dataset_fingerprint, _ = dataset_content_fingerprint(
        args.processed_dir,
        manifest,
        records,
    )
    if (
        current_dataset_fingerprint
        != training_config.get("dataset_content_fingerprint")
    ):
        raise RuntimeError(
            f"El contenido de {args.processed_dir} cambió desde que se entrenó "
            "la corrida."
        )
    if (
        ensemble.get("run_fingerprint") != training_config.get("run_fingerprint")
        or ensemble.get("dataset_content_fingerprint")
        != training_config.get("dataset_content_fingerprint")
        or ensemble.get("implementation_version")
        != training_config.get("implementation_version")
    ):
        raise RuntimeError("Las huellas del ensemble y la corrida no coinciden.")
    expected_by_id = {record.video_id: record.label_index for record in records}
    ids = video_ids.astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("Las predicciones OOF contienen videos duplicados.")
    if set(ids) != set(expected_by_id):
        raise RuntimeError(
            "Las predicciones OOF no cubren exactamente todos los videos del manifiesto."
        )
    expected_true = np.asarray([expected_by_id[video_id] for video_id in ids])
    if not np.array_equal(y_true, expected_true):
        raise RuntimeError("Las etiquetas OOF no coinciden con el manifiesto.")
    if probabilities.shape != (len(records), len(labels)):
        raise RuntimeError("La matriz de probabilidades OOF tiene una forma inválida.")
    metrics = compute_metrics(y_true, probabilities, labels)
    roc_summary = None
    if args.write_roc_artifacts:
        roc_summary = save_roc_artifacts(
            args.run_dir,
            y_true,
            probabilities,
            labels,
        )
        metrics_path = args.run_dir / "cv_metrics.json"
        existing_metrics = read_json(metrics_path) if metrics_path.is_file() else {}
        if not isinstance(existing_metrics, dict):
            raise RuntimeError("cv_metrics.json debe contener un objeto JSON.")
        existing_metrics["roc_auc"] = roc_summary
        write_json(metrics_path, existing_metrics)
    if args.fruit:
        label = resolve_label(args.fruit, labels)
        output = {"fruit": label, **metrics["per_class"][label]}
        if roc_summary is not None:
            output["roc_auc_ovr"] = roc_summary["per_class"][label]["auc_ovr"]
    else:
        output = {
            "videos": len(y_true),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "top3_accuracy": metrics["top3_accuracy"],
        }
        if roc_summary is not None:
            output["roc_auc"] = {
                key: roc_summary[key]
                for key in ("macro_ovr_auc", "weighted_ovr_auc", "micro_ovr_auc")
            }
            output["roc_artifacts"] = roc_summary["artifacts"]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
