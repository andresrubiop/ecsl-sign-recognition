import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import pfrutas.live as live_module
from pfrutas.features import (
    FeatureStandardizer,
    PRESENCE_SLICE,
    feature_layout,
    model_feature_dim,
    static_feature_dim,
)
from pfrutas.data import transfer_feature_layout
from pfrutas.asl_transfer import (
    ASL_CITIZEN_CHECKPOINT_SHA256,
    ASL_GRAPH_CONTRACT,
)
from pfrutas.live import (
    LoadedMember,
    build_general_oof_metrics_text,
    build_jetson_csi_pipeline,
    build_live_overlay_layout,
    build_prediction_panel,
    ensemble_predict,
    load_ensemble,
    run_live,
)
from pfrutas.model import (
    ASL_TRANSFER_MODEL_PRESET,
    SignLSTM,
    build_model,
    model_config_for_preset,
    save_checkpoint,
)
from pfrutas.training import (
    EXPERIMENT_IMPLEMENTATION_VERSION,
    TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION,
)


class _IdentityStandardizer:
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return matrix


class _ViewSensitiveModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs[:, 0, 0]
        return torch.stack((value, -value), dim=1)


def test_ensemble_predict_uses_real_timestamps_and_averages_views(monkeypatch) -> None:
    captured_timestamps: list[np.ndarray] = []

    def fake_views(
        sequence: np.ndarray,
        length: int,
        timestamps: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        assert sequence.shape == (4, static_feature_dim())
        assert length == 6
        assert timestamps is not None
        captured_timestamps.append(timestamps.copy())
        views = []
        for value in (0.0, 1.0, 2.0):
            view = np.zeros((length, static_feature_dim()), dtype=np.float32)
            view[:, 0] = value
            view[:, PRESENCE_SLICE.start] = 1.0
            views.append(view)
        return views

    monkeypatch.setattr(live_module, "deterministic_views", fake_views)
    member = LoadedMember(
        model=_ViewSensitiveModel(),  # type: ignore[arg-type]
        standardizer=_IdentityStandardizer(),
        payload={"sequence_length": 6},
    )
    sequence = np.zeros((4, static_feature_dim()), dtype=np.float32)
    timestamps = np.asarray(
        (1_000_000.0, 1_000_000.1, 1_000_000.4, 1_000_000.9),
        dtype=np.float64,
    )

    probability = ensemble_predict([member], sequence, "cpu", timestamps=timestamps)

    assert len(captured_timestamps) == 1
    assert np.allclose(captured_timestamps[0], (0.0, 0.1, 0.4, 0.9))
    logits = torch.tensor(((0.0, 0.0), (1.0, -1.0), (2.0, -2.0)))
    expected = torch.softmax(logits, dim=1).mean(dim=0).numpy()
    assert np.allclose(probability, expected, atol=1e-6)


def test_ensemble_predict_rejects_non_monotonic_timestamps() -> None:
    sequence = np.zeros((3, static_feature_dim()), dtype=np.float32)
    try:
        ensemble_predict(
            [],
            sequence,
            "cpu",
            timestamps=np.asarray((0.0, 0.1, 0.1)),
        )
    except ValueError as exc:
        assert "estrictamente crecientes" in str(exc)
    else:
        raise AssertionError("Los timestamps duplicados deben rechazarse.")


def test_target_panel_only_exposes_target_information() -> None:
    labels = ["MANGO", "PERA", "UVILLA"]
    probabilities = np.asarray((0.20, 0.70, 0.10), dtype=np.float32)
    per_class = {
        "MANGO": {"precision": 0.61, "recall": 0.52, "f1": 0.56},
        "PERA": {"precision": 0.99, "recall": 0.98, "f1": 0.97},
    }

    panel = build_prediction_panel(
        labels,
        probabilities,
        threshold=0.55,
        per_class=per_class,
        target_index=0,
        attempts=4,
        correct_attempts=3,
    )

    visible_text = " ".join(
        text
        for text in (
            panel.primary_text,
            panel.confidence_text,
            panel.cv_text,
            panel.status_text,
            panel.session_text,
        )
        if text is not None
    )
    assert "MANGO" in visible_text
    assert "20.0%" in visible_text
    assert "OOF de MANGO: P 61.0% | R 52.0% | F1 56.0%" in visible_text
    assert "Exactitud de sesion MANGO: 3/4 (75.0%)" in visible_text
    assert "NO DETECTADA" in visible_text
    assert "PERA" not in visible_text
    assert "UVILLA" not in visible_text
    assert "99.0%" not in visible_text
    assert "Exactitud OOF general" not in visible_text
    assert panel.show_probability_bars is False
    assert panel.show_global_metrics is False


def test_general_panel_preserves_winner_and_general_displays() -> None:
    panel = build_prediction_panel(
        ["MANGO", "PERA"],
        np.asarray((0.20, 0.80), dtype=np.float32),
        threshold=0.55,
        per_class={"PERA": {"precision": 0.75}},
    )

    assert panel.primary_text == "Prediccion: PERA"
    assert "80.0%" in panel.confidence_text
    assert panel.cv_text == "Precision OOF de PERA: 75.0%"
    assert panel.status_text is None
    assert panel.session_text is None
    assert panel.show_probability_bars is True
    assert panel.show_global_metrics is True


def test_general_oof_text_identifies_accuracy_as_out_of_fold() -> None:
    text = build_general_oof_metrics_text(0.8194444444, 0.8467261904)

    assert text == (
        "Exactitud OOF general: 81.9% | "
        "precision macro: 84.7%"
    )


def test_general_oof_text_rejects_invalid_metrics() -> None:
    try:
        build_general_oof_metrics_text(float("nan"), 0.8)
    except ValueError as exc:
        assert "OOF" in str(exc)
    else:
        raise AssertionError("Una metrica OOF NaN debe rechazarse.")


def test_reduced_profile_panel_hides_cv_metrics() -> None:
    panel = build_prediction_panel(
        ["MANGO", "PERA"],
        np.asarray((0.70, 0.30), dtype=np.float32),
        threshold=0.55,
        per_class={"MANGO": {"precision": 0.99}},
        show_cv_metrics=False,
    )

    assert panel.cv_text is None
    assert panel.show_global_metrics is False
    assert panel.primary_text == "Prediccion: MANGO"

    target_panel = build_prediction_panel(
        ["MANGO", "PERA"],
        np.asarray((0.70, 0.30), dtype=np.float32),
        threshold=0.55,
        per_class={"MANGO": {"precision": 0.99, "recall": 0.99, "f1": 0.99}},
        target_index=0,
        show_cv_metrics=False,
    )
    assert target_panel.cv_text is None
    assert target_panel.show_global_metrics is False
    assert "MANGO" in target_panel.primary_text


def test_panel_hides_missing_class_metrics_instead_of_showing_zero() -> None:
    general = build_prediction_panel(
        ["MANGO", "PERA"],
        np.asarray((0.70, 0.30), dtype=np.float32),
        threshold=0.55,
        per_class={},
        show_cv_metrics=True,
        show_global_metrics=True,
    )
    assert general.cv_text is None
    assert general.show_global_metrics is True

    target = build_prediction_panel(
        ["MANGO", "PERA"],
        np.asarray((0.70, 0.30), dtype=np.float32),
        threshold=0.55,
        per_class={"MANGO": {"precision": 0.7}},
        target_index=0,
        show_cv_metrics=True,
    )
    assert target.cv_text is None
    assert target.show_global_metrics is False


def _assert_rect_inside_frame(rect: object, width: int, height: int) -> None:
    assert getattr(rect, "x") >= 0
    assert getattr(rect, "y") >= 0
    assert getattr(rect, "right") <= width
    assert getattr(rect, "bottom") <= height


def _contains(rect: object, point: tuple[int, int]) -> bool:
    x, y = point
    return (
        getattr(rect, "x") <= x < getattr(rect, "right")
        and getattr(rect, "y") <= y < getattr(rect, "bottom")
    )


def test_general_live_layout_keeps_the_camera_center_clear() -> None:
    layout = build_live_overlay_layout(
        1280,
        720,
        prediction_ready=True,
        target_mode=False,
        show_probability_bars=True,
        show_global_metrics=True,
    )

    assert layout.header.height < 80
    assert layout.card.right <= 500
    assert layout.card.y > 400
    assert layout.metrics is not None
    assert layout.metrics.x > 600
    assert layout.probability_bar_count == 5
    assert layout.global_metrics_in_card is False
    for rect in (layout.header, layout.card, layout.metrics, layout.toast):
        if rect is not None:
            _assert_rect_inside_frame(rect, 1280, 720)
            assert not _contains(rect, (640, 360))


def test_target_live_layout_hides_global_ui_and_keeps_center_clear() -> None:
    layout = build_live_overlay_layout(
        1280,
        720,
        prediction_ready=True,
        target_mode=True,
        show_probability_bars=False,
        show_global_metrics=False,
    )

    assert layout.metrics is None
    assert layout.probability_bar_count == 0
    assert layout.global_metrics_in_card is False
    for rect in (layout.header, layout.card, layout.toast):
        if rect is not None:
            _assert_rect_inside_frame(rect, 1280, 720)
            assert not _contains(rect, (640, 360))


def test_live_layout_stays_inside_a_small_frame() -> None:
    layout = build_live_overlay_layout(
        320,
        240,
        prediction_ready=True,
        target_mode=False,
        show_probability_bars=True,
        show_global_metrics=True,
    )

    for rect in (layout.header, layout.card, layout.metrics, layout.toast):
        if rect is not None:
            _assert_rect_inside_frame(rect, 320, 240)
    assert layout.probability_bar_count == 0


def test_jetson_csi_pipeline_is_fixed_and_validated() -> None:
    pipeline = build_jetson_csi_pipeline(
        sensor_id=1,
        width=1920,
        height=1080,
        fps=30,
        flip_method=2,
    )

    assert pipeline.startswith("nvarguscamerasrc sensor-id=1 !")
    assert "width=(int)1920" in pipeline
    assert "height=(int)1080" in pipeline
    assert "framerate=(fraction)30/1" in pipeline
    assert "nvvidconv flip-method=2" in pipeline
    assert pipeline.endswith("appsink drop=true max-buffers=1 sync=false")

    for invalid in (-1, 8, "0"):
        try:
            build_jetson_csi_pipeline(flip_method=invalid)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("flip_method inválido debe rechazarse.")


def test_run_live_rejects_invalid_prediction_rate_before_loading() -> None:
    try:
        run_live(Path("inexistente"), prediction_fps=0.0)
    except ValueError as exc:
        assert "prediction-fps" in str(exc)
    else:
        raise AssertionError("prediction_fps=0 debe rechazarse.")


def test_transfer_ensemble_predict_requires_and_uses_raw_pose_hands() -> None:
    model = build_model(
        model_config_for_preset(
            model_feature_dim(),
            3,
            ASL_TRANSFER_MODEL_PRESET,
        )
    ).eval()
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    member = LoadedMember(
        model=model,
        standardizer=standardizer,
        payload={"sequence_length": 48},
    )
    sequence = np.zeros((20, static_feature_dim()), dtype=np.float32)
    raw = np.full((20, 75, 2), 0.5, dtype=np.float32)
    raw[:, 11, 0] = 0.25
    raw[:, 12, 0] = 0.75

    try:
        ensemble_predict([member], sequence, "cpu")
    except ValueError as exc:
        assert "pose_hands_xy" in str(exc)
    else:
        raise AssertionError("El ensemble transfer debe exigir pose_hands_xy.")

    probabilities = ensemble_predict(
        [member],
        sequence,
        "cpu",
        pose_hands_xy=raw,
    )
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)


def test_transfer_graph_tensor_is_created_once_for_shared_length(monkeypatch) -> None:
    model = build_model(
        model_config_for_preset(
            model_feature_dim(),
            3,
            ASL_TRANSFER_MODEL_PRESET,
        )
    ).eval()
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    member = LoadedMember(
        model=model,
        standardizer=standardizer,
        payload={"sequence_length": 48},
    )
    sequence = np.zeros((20, static_feature_dim()), dtype=np.float32)
    raw = np.full((20, 75, 2), 0.5, dtype=np.float32)
    raw[:, 11, 0] = 0.25
    raw[:, 12, 0] = 0.75

    tensor_shapes: list[tuple[int, ...]] = []
    original_from_numpy = torch.from_numpy

    def tracked_from_numpy(array: np.ndarray) -> torch.Tensor:
        tensor_shapes.append(tuple(array.shape))
        return original_from_numpy(array)

    monkeypatch.setattr(torch, "from_numpy", tracked_from_numpy)
    probabilities = ensemble_predict(
        [member, member],
        sequence,
        "cpu",
        pose_hands_xy=raw,
    )

    assert probabilities.shape == (3,)
    assert tensor_shapes.count((3, 2, 128, 27)) == 1
    assert tensor_shapes.count((3, 48, model_feature_dim())) == 2


def test_load_ensemble_rejects_incompatible_feature_layout(tmp_path: Path) -> None:
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
                "labels": ["MANGO"],
                "feature_layout": {"version": "otro_layout"},
                "model_config": {"input_dim": 1, "num_classes": 1},
                "sequence_length": 48,
                "checkpoints": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_ensemble(tmp_path, "cpu")
    except RuntimeError as exc:
        assert "feature layout v2" in str(exc)
    else:
        raise AssertionError("Un feature layout incompatible debe rechazarse.")


def test_load_ensemble_validates_checkpoint_feature_layout(tmp_path: Path) -> None:
    labels = ["MANGO"]
    config = model_config_for_preset(model_feature_dim(), len(labels))
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        SignLSTM(config),
        None,
        standardizer,
        labels,
        48,
        0,
        1,
        {},
        {
            "feature_layout": {"version": "incorrecto"},
            "model_config": asdict(config),
            "validation_protocol": "nested",
            "run_fingerprint": "run",
            "dataset_content_fingerprint": "dataset",
        },
    )
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
                "labels": labels,
                "feature_layout": feature_layout(),
                "model_config": asdict(config),
                "sequence_length": 48,
                "validation_protocol": "nested",
                "run_fingerprint": "run",
                "dataset_content_fingerprint": "dataset",
                "checkpoints": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_ensemble(tmp_path, "cpu")
    except RuntimeError as exc:
        assert "feature layout de preprocesamiento" in str(exc)
    else:
        raise AssertionError("El layout incompatible del checkpoint debe rechazarse.")


def test_load_ensemble_final_profile_uses_safe_checkpoint_and_drops_state(
    tmp_path: Path,
) -> None:
    labels = ["MANGO", "PERA"]
    config = model_config_for_preset(model_feature_dim(), len(labels))
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    preprocessing = {
        "feature_layout": feature_layout(),
        "model_config": asdict(config),
        "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
        "validation_protocol": "nested",
        "run_fingerprint": "run",
        "dataset_content_fingerprint": "dataset",
    }
    for relative, fold in (("fold.pt", 0), ("final.pt", -1)):
        save_checkpoint(
            tmp_path / relative,
            SignLSTM(config),
            None,
            standardizer,
            labels,
            48,
            fold,
            1,
            {},
            preprocessing,
        )
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "implementation_version": EXPERIMENT_IMPLEMENTATION_VERSION,
                "labels": labels,
                "feature_layout": feature_layout(),
                "model_config": asdict(config),
                "sequence_length": 48,
                "validation_protocol": "nested",
                "run_fingerprint": "run",
                "dataset_content_fingerprint": "dataset",
                "checkpoints": ["fold.pt", "final.pt"],
                "final_checkpoint": "final.pt",
            }
        ),
        encoding="utf-8",
    )

    members, ensemble = load_ensemble(
        tmp_path,
        "cpu",
        weights_only=True,
        checkpoint_profile="final",
    )

    assert len(members) == 1
    assert members[0].payload["fold"] == -1
    assert "model_state" not in members[0].payload
    assert "optimizer_state" not in members[0].payload
    assert ensemble["_checkpoint_profile"] == "final"
    assert ensemble["_cv_metrics_applicable"] is False

    try:
        load_ensemble(tmp_path, "cpu", checkpoint_profile="unknown")
    except ValueError as exc:
        assert "checkpoint_profile" in str(exc)
    else:
        raise AssertionError("Un perfil desconocido debe rechazarse.")


def test_load_transfer_ensemble_validates_each_checkpoint_contract(
    tmp_path: Path,
) -> None:
    labels = ["MANGO"]
    config = model_config_for_preset(
        model_feature_dim(),
        len(labels),
        ASL_TRANSFER_MODEL_PRESET,
    )
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    pretraining = {
        "sha256": ASL_CITIZEN_CHECKPOINT_SHA256,
        "expected_sha256": ASL_CITIZEN_CHECKPOINT_SHA256,
        "sha256_verified": True,
        "strict": True,
        "safe_weights_only": True,
    }
    save_checkpoint(
        tmp_path / "transfer.pt",
        build_model(config),
        None,
        standardizer,
        labels,
        48,
        0,
        1,
        {},
        {
            "feature_layout": feature_layout(),
            "transfer_contract": "contrato_incorrecto",
            "transfer_layout": transfer_feature_layout(),
            "model_config": asdict(config),
            "pretraining": pretraining,
            "implementation_version": TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION,
            "validation_protocol": "nested",
            "run_fingerprint": "run",
            "dataset_content_fingerprint": "dataset",
        },
    )
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "implementation_version": TRANSFER_EXPERIMENT_IMPLEMENTATION_VERSION,
                "labels": labels,
                "feature_layout": feature_layout(),
                "transfer_contract": "pose_hands_xy_v1",
                "transfer_layout": transfer_feature_layout(),
                "model_config": asdict(config),
                "pretraining": {
                    "checkpoint_sha256": ASL_CITIZEN_CHECKPOINT_SHA256,
                    "cache_layout_version": "pose_hands_xy_v1",
                    "input_contract": ASL_GRAPH_CONTRACT,
                },
                "sequence_length": 48,
                "validation_protocol": "nested",
                "run_fingerprint": "run",
                "dataset_content_fingerprint": "dataset",
                "checkpoints": ["transfer.pt"],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_ensemble(tmp_path, "cpu")
    except RuntimeError as exc:
        assert "contrato pose transfer" in str(exc)
    else:
        raise AssertionError("Cada miembro transfer debe validar su contrato pose.")
