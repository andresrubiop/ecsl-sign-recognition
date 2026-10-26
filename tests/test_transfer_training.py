from __future__ import annotations

from pathlib import Path

import numpy as np

from pfrutas.data import VideoRecord
from pfrutas.features import (
    FeatureStandardizer,
    POSE_VISIBILITY_SLICE,
    PRESENCE_SLICE,
    model_feature_dim,
    random_temporal_view,
    resample_sequence,
    static_feature_dim,
)
from pfrutas.asl_transfer import prepare_asl_graph
from pfrutas.model import ASL_TRANSFER_MODEL_PRESET, model_config_for_preset
from pfrutas.training import AugmentedSequenceDataset


def test_transfer_dataset_returns_reproducible_paired_inputs(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequences"
    sequence_dir.mkdir()
    frames = 24
    features = np.zeros((frames, static_feature_dim()), dtype=np.float32)
    features[:, PRESENCE_SLICE.start] = 1.0
    timestamps = np.arange(frames, dtype=np.float32) / 30.0
    pose_hands = np.full((frames, 75, 2), 0.5, dtype=np.float32)
    pose_hands[:, 11, 0] = 0.25
    pose_hands[:, 12, 0] = 0.75
    destination = sequence_dir / "sample.npz"
    np.savez_compressed(
        destination,
        features=features,
        timestamps=timestamps,
        pose_hands_xy=pose_hands,
    )
    record = VideoRecord(
        source="MANGO/Mango0.mp4",
        sequence="sequences/sample.npz",
        label="MANGO",
        label_index=0,
        signer_id="signer_0",
        video_id="mango_0",
        frames=frames,
        duration=float(timestamps[-1]),
        quality={"pose": 1.0},
    )
    standardizer = FeatureStandardizer(
        mean=np.zeros(model_feature_dim(), dtype=np.float32),
        std=np.ones(model_feature_dim(), dtype=np.float32),
    )
    config = model_config_for_preset(
        model_feature_dim(),
        2,
        ASL_TRANSFER_MODEL_PRESET,
    )
    dataset = AugmentedSequenceDataset(
        [record],
        tmp_path,
        sequence_length=48,
        standardizer=standardizer,
        samples_per_video=1,
        seed=42,
        model_config=config,
    )

    (landmarks_first, graph_first), target_first = dataset[0]
    (landmarks_second, graph_second), target_second = dataset[0]

    assert landmarks_first.shape == (48, model_feature_dim())
    assert graph_first.shape == (2, 128, 27)
    assert np.isfinite(landmarks_first.numpy()).all()
    assert np.isfinite(graph_first.numpy()).all()
    assert np.array_equal(landmarks_first.numpy(), landmarks_second.numpy())
    assert np.array_equal(graph_first.numpy(), graph_second.numpy())
    assert int(target_first) == int(target_second) == 0


def test_temporal_plan_selects_the_exact_same_source_frames_for_both_branches() -> None:
    frames = 30
    length = 13
    sequence = np.zeros((frames, static_feature_dim()), dtype=np.float32)
    sequence[:, PRESENCE_SLICE.start] = 1.0
    # La visibilidad no recibe transformaciones espaciales; funciona como ID
    # observable del frame que llegó a la rama BiLSTM.
    frame_ids = np.arange(frames, dtype=np.float32)
    sequence[:, POSE_VISIBILITY_SLICE.start] = frame_ids
    timestamps = frame_ids / 30.0

    view, plan = random_temporal_view(
        sequence,
        length,
        np.random.default_rng(2026),
        timestamps=timestamps,
        min_fraction=0.55,
        noise_std=0.0,
        force_mirror=False,
        speed_warp=1.0,
        return_plan=True,
    )
    start = int(plan["source_start"])
    stop = int(plan["source_stop"])
    positions = np.linspace(0.0, stop - start - 1, length, dtype=np.float32)
    expected_view = resample_sequence(sequence[start:stop], length, positions)
    assert np.allclose(
        view[:, POSE_VISIBILITY_SLICE.start],
        expected_view[:, POSE_VISIBILITY_SLICE.start],
    )

    raw = np.zeros((frames, 75, 2), dtype=np.float32)
    raw[:, 11] = (0.25, 0.50)
    raw[:, 12] = (0.75, 0.50)
    # Movimiento relativo a los hombros para que cada intervalo produzca un
    # grafo identificable después de la normalización oficial.
    raw[:, 0, 0] = 0.40 + frame_ids * 0.005
    raw[:, 0, 1] = 0.30 + frame_ids * 0.002
    bounded_graph = prepare_asl_graph(raw, frame_bounds=(start, stop))
    sliced_graph = prepare_asl_graph(raw[start:stop])
    assert np.array_equal(bounded_graph, sliced_graph)
    if stop < frames:
        shifted_graph = prepare_asl_graph(raw[start + 1 : stop + 1])
        assert not np.array_equal(bounded_graph, shifted_graph)
