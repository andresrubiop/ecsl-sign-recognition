from dataclasses import asdict

import numpy as np
import torch

from pfrutas.features import FeatureStandardizer, model_feature_dim
from pfrutas.model import (
    ASLTransferModelConfig,
    ASL_TRANSFER_MODEL_PRESET,
    ASLTransferFusion,
    COMPACT_MODEL_PRESET,
    ModelConfig,
    SignLSTM,
    build_model,
    load_checkpoint,
    model_config_for_preset,
)


NUM_CLASSES = 16
SEQUENCE_LENGTH = 48


def test_compact_v2_parameter_budget() -> None:
    config = model_config_for_preset(
        input_dim=model_feature_dim(),
        num_classes=NUM_CLASSES,
        preset=COMPACT_MODEL_PRESET,
    )
    model = SignLSTM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert config.projection_dim == 80
    assert config.hidden_dim == 40
    assert config.num_layers == 1
    assert config.dropout == 0.25
    assert config.bidirectional is True
    assert 100_000 <= parameter_count <= 150_000


def test_compact_v2_forward_and_attention() -> None:
    model = SignLSTM(
        model_config_for_preset(model_feature_dim(), NUM_CLASSES)
    ).eval()
    inputs = torch.randn(2, SEQUENCE_LENGTH, model_feature_dim())

    with torch.inference_mode():
        logits, attention = model(inputs, return_attention=True)

    assert logits.shape == (2, NUM_CLASSES)
    assert attention.shape == (2, SEQUENCE_LENGTH)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention).all()
    assert torch.all(attention >= 0)
    assert torch.allclose(
        attention.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_load_checkpoint_preserves_legacy_model_config(tmp_path) -> None:
    legacy_config = ModelConfig(
        input_dim=model_feature_dim(),
        num_classes=NUM_CLASSES,
    )
    legacy_model = SignLSTM(legacy_config).eval()
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state": legacy_model.state_dict(),
            "optimizer_state": None,
            "model_config": asdict(legacy_config),
            "feature_scaler": standardizer.to_dict(),
            "labels": [f"class_{index}" for index in range(NUM_CLASSES)],
            "sequence_length": SEQUENCE_LENGTH,
            "fold": 0,
            "epoch": 1,
            "metrics": {},
            "preprocessing": {},
        },
        checkpoint,
    )

    loaded_model, loaded_standardizer, payload = load_checkpoint(checkpoint, "cpu")

    assert loaded_model.config == legacy_config
    assert payload["model_config"] == asdict(legacy_config)
    assert "model_state" in payload
    assert "optimizer_state" in payload
    assert np.array_equal(loaded_standardizer.mean, standardizer.mean)
    assert np.array_equal(loaded_standardizer.std, standardizer.std)
    expected_states = list(legacy_model.state_dict().values())
    actual_states = list(loaded_model.state_dict().values())
    assert len(expected_states) == len(actual_states)
    for expected, actual in zip(expected_states, actual_states):
        assert torch.equal(expected, actual)


def test_unknown_model_preset_is_rejected() -> None:
    try:
        model_config_for_preset(model_feature_dim(), NUM_CLASSES, "unknown")
    except ValueError as exc:
        assert "Preset de modelo desconocido" in str(exc)
    else:
        raise AssertionError("Un preset desconocido debe producir ValueError.")


def test_asl_transfer_fusion_shape_gate_and_freeze_schedule() -> None:
    config = model_config_for_preset(
        model_feature_dim(),
        NUM_CLASSES,
        ASL_TRANSFER_MODEL_PRESET,
    )
    model = build_model(config)
    assert isinstance(model, ASLTransferFusion)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    warmup_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    model.eval()
    with torch.inference_mode():
        logits = model(
            torch.randn(2, SEQUENCE_LENGTH, model_feature_dim()),
            torch.randn(2, 2, 128, 27),
        )

    assert logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(logits).all()
    assert 3_200_000 <= total_parameters <= 3_400_000
    assert 150_000 <= warmup_parameters <= 200_000
    assert torch.allclose(
        torch.sigmoid(model.fusion_gate[-1].bias),
        torch.tensor((0.75,)),
        atol=1e-6,
    )

    model.configure_epoch(config.unfreeze_backbone_epoch)
    opened_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert 900_000 <= opened_parameters <= 1_000_000
    model.train()
    batch_norms = [
        module
        for module in model.asl_encoder.modules()
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)


def test_asl_transfer_rejects_non_official_graph_length() -> None:
    config = model_config_for_preset(
        model_feature_dim(),
        NUM_CLASSES,
        ASL_TRANSFER_MODEL_PRESET,
    )
    invalid = asdict(config)
    invalid["graph_sequence_length"] = 64
    try:
        ASLTransferModelConfig(**invalid)
    except ValueError as exc:
        assert "exactamente 128" in str(exc)
    else:
        raise AssertionError("ASL Citizen v1 debe fijar T=128.")
