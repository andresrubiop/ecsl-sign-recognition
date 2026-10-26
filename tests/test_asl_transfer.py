from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pfrutas.asl_transfer import (
    ASL_CITIZEN_CODE_LICENSE,
    ASL_CITIZEN_GRAPH_INDICES,
    ASL_GRAPH_CONTRACT,
    ASL_GRAPH_EMBEDDING_DIM,
    ASL_GRAPH_NUM_NODES,
    ASL_GRAPH_RAW_INDICES,
    ASL_GRAPH_SEQUENCE_LENGTH,
    ASLCitizenSTGCN,
    TRANSFER_CACHE_LAYOUT_VERSION,
    asl_citizen_temporal_fit,
    load_asl_citizen_encoder,
    mirror_pose_hands_xy,
    prepare_asl_graph,
    prepare_asl_graph_augmented,
    select_asl_graph_keypoints,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_asl_citizen_v1_contract_is_explicit() -> None:
    assert TRANSFER_CACHE_LAYOUT_VERSION == "pose_hands_xy_v1"
    assert ASL_GRAPH_SEQUENCE_LENGTH == 128
    assert ASL_GRAPH_NUM_NODES == 27
    assert ASL_GRAPH_EMBEDDING_DIM == 256
    assert len(ASL_CITIZEN_GRAPH_INDICES) == 27
    assert ASL_GRAPH_CONTRACT["graph_shape"] == [2, 128, 27]
    assert ASL_GRAPH_CONTRACT["dtype"] == "float32"
    assert ASL_GRAPH_CONTRACT["code_license"] == ASL_CITIZEN_CODE_LICENSE == "MIT"
    assert ASL_GRAPH_CONTRACT["weights_bundled"] is False


def test_graph_selection_reorders_right_and_left_hands_exactly() -> None:
    raw = np.zeros((2, 75, 2), dtype=np.float32)
    raw[:, :, 0] = np.arange(75, dtype=np.float32)
    raw[:, :, 1] = -np.arange(75, dtype=np.float32)

    graph = select_asl_graph_keypoints(raw)

    assert graph.shape == (2, 27, 2)
    assert np.array_equal(graph[..., 0], raw[:, ASL_GRAPH_RAW_INDICES, 0])
    assert np.array_equal(graph[..., 1], raw[:, ASL_GRAPH_RAW_INDICES, 1])
    # Primer wrist de grafo es mano izquierda raw54; el segundo es derecha raw33.
    assert graph[0, 7, 0] == 54
    assert graph[0, 17, 0] == 33


def test_raw_mirror_is_involutive_and_preserves_missing_modalities() -> None:
    rng = np.random.default_rng(17)
    raw = rng.uniform(0.05, 0.95, size=(4, 75, 2)).astype(np.float32)
    raw[1, 33:54] = 0.0  # mano derecha ausente
    raw[2, :33] = 0.0  # pose ausente
    raw[3, 54:75] = 0.0  # mano izquierda ausente

    mirrored = mirror_pose_hands_xy(raw)
    restored = mirror_pose_hands_xy(mirrored)

    assert mirrored.dtype == np.float32
    assert np.allclose(restored, raw, rtol=0.0, atol=1e-7)
    assert np.count_nonzero(mirrored[1, 54:75]) == 0
    assert np.count_nonzero(mirrored[2, :33]) == 0
    assert np.count_nonzero(mirrored[3, 33:54]) == 0
    assert np.allclose(mirrored[0, 54:75, 0], 1.0 - raw[0, 33:54, 0])
    assert np.allclose(mirrored[0, 33:54, 0], 1.0 - raw[0, 54:75, 0])


def test_official_temporal_downsample_uses_discard_algorithm() -> None:
    raw = np.zeros((256, 75, 2), dtype=np.float32)
    raw[:, :, 0] = np.arange(256, dtype=np.float32)[:, None]

    fitted = asl_citizen_temporal_fit(raw)

    assert fitted.shape == (128, 75, 2)
    assert fitted.dtype == np.float32
    assert np.array_equal(fitted[:, 0, 0], np.arange(0, 256, 2, dtype=np.float32))


def test_padding_precedes_shoulder_normalization_golden_values() -> None:
    # Un solo frame real y 127 frames de padding. El codigo Microsoft incluye
    # esos ceros al promediar: center=(.5,.5)/128, mean_dist=.5/128.
    raw = np.zeros((1, 75, 2), dtype=np.float32)
    raw[0, 0] = (0.50, 0.25)  # nariz
    raw[0, 11] = (0.25, 0.50)  # hombro izquierdo
    raw[0, 12] = (0.75, 0.50)  # hombro derecho

    graph = prepare_asl_graph(raw)

    assert graph.shape == (2, 128, 27)
    assert graph.dtype == np.float32
    assert np.array_equal(graph[:, 0, 0], np.asarray((127.0, 63.0), dtype=np.float32))
    assert np.array_equal(graph[:, 0, 3], np.asarray((63.0, 127.0), dtype=np.float32))
    assert np.array_equal(graph[:, 0, 4], np.asarray((191.0, 127.0), dtype=np.float32))
    # Padding y landmarks ausentes pasan a (-1,-1), no vuelven a cero despues.
    assert np.array_equal(graph[:, 1, 0], np.asarray((-1.0, -1.0), dtype=np.float32))
    assert np.array_equal(graph[:, 0, 7], np.asarray((-1.0, -1.0), dtype=np.float32))


def test_augmented_view_is_seed_reproducible() -> None:
    raw = np.full((24, 75, 2), 0.5, dtype=np.float32)
    raw[:, 11, 0] = 0.25
    raw[:, 12, 0] = 0.75

    first = prepare_asl_graph_augmented(raw, rng=np.random.default_rng(91))
    second = prepare_asl_graph_augmented(raw, rng=np.random.default_rng(91))

    assert first.shape == (2, 128, 27)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_stgcn_has_official_ten_blocks_and_float32_embedding() -> None:
    model = ASLCitizenSTGCN().eval()
    inputs = torch.randn(1, 2, ASL_GRAPH_SEQUENCE_LENGTH, 27, dtype=torch.float64)

    with torch.inference_mode():
        embedding = model(inputs)

    assert len(model.st_gcn_networks) == 10
    assert model.A.shape == (3, 27, 27)
    assert model.A.dtype == torch.float32
    assert embedding.shape == (1, ASL_GRAPH_EMBEDDING_DIM)
    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_safe_loader_strips_only_encoder_and_converts_floats(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).eval()
    state: dict[str, torch.Tensor] = {
        f"encoder.{key}": (
            value.double() if value.is_floating_point() else value.clone()
        )
        for key, value in source.state_dict().items()
    }
    state["decoder.classifier.weight"] = torch.randn(9, 4)
    checkpoint = tmp_path / "asl_synthetic.pt"
    torch.save({"state_dict": state, "epoch": 1}, checkpoint)

    target = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).eval()
    metadata = load_asl_citizen_encoder(
        target,
        checkpoint,
        expected_sha256=_sha256(checkpoint).upper(),
    )

    for key, expected in source.state_dict().items():
        actual = target.state_dict()[key]
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    assert metadata["sha256_verified"] is True
    assert metadata["safe_weights_only"] is True
    assert metadata["strict"] is True
    assert metadata["encoder_prefix"] == "encoder."
    assert metadata["loaded_tensor_count"] == len(source.state_dict())
    assert metadata["weights_bundled"] is False


def test_safe_loader_rejects_sha_mismatch_before_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad_sha.pt"
    torch.save({"encoder.weight": torch.ones(1)}, checkpoint)

    try:
        load_asl_citizen_encoder(
            nn.Linear(1, 1),
            checkpoint,
            expected_sha256="0" * 64,
        )
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("Un SHA-256 incorrecto debe rechazarse.")


def test_safe_loader_keeps_strict_state_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"encoder.weight": torch.ones((1, 1))}, checkpoint)

    try:
        load_asl_citizen_encoder(nn.Linear(1, 1), checkpoint)
    except RuntimeError as exc:
        assert "strict" in str(exc)
        assert "bias" in str(exc)
    else:
        raise AssertionError("Un encoder incompleto no debe cargarse.")
