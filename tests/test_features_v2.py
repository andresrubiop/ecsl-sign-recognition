import numpy as np

from pfrutas.features import (
    FACE_XYZ_SLICE,
    FEATURE_LAYOUT_VERSION,
    GEOMETRY_DIM,
    LEFT_HAND_GLOBAL_SLICE,
    LEFT_HAND_PRESENCE,
    LEFT_HAND_SLICE,
    MAX_INTERPOLATION_GAP,
    MODALITY_GEOMETRY,
    POSE_INDICES,
    POSE_PRESENCE,
    POSE_VISIBILITY_SLICE,
    POSE_XYZ_SLICE,
    PRESENCE_DIM,
    PRESENCE_SLICE,
    RIGHT_HAND_PRESENCE,
    RIGHT_HAND_SLICE,
    SCHEMA_VERSION,
    FeatureStandardizer,
    add_temporal_deltas,
    deterministic_views,
    feature_layout,
    fill_short_landmark_gaps,
    mirror_sequence,
    model_feature_dim,
    resample_sequence,
    segment_active_sequence,
    static_feature_dim,
)


def _presence_column(index: int) -> int:
    return PRESENCE_SLICE.start + index


def _set_modality(
    sequence: np.ndarray,
    frame: int,
    geometry_slice: slice,
    presence_index: int,
    value: float,
) -> None:
    sequence[frame, geometry_slice] = value
    sequence[frame, _presence_column(presence_index)] = 1.0


def _gesture_sequence(
    active_frames: int,
    rest_frames: int = 30,
    fps: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    total = 2 * rest_frames + active_frames
    sequence = np.zeros((total, static_feature_dim()), dtype=np.float32)
    sequence[:, _presence_column(POSE_PRESENCE)] = 1.0
    sequence[:, _presence_column(LEFT_HAND_PRESENCE)] = 1.0
    sequence[:, POSE_VISIBILITY_SLICE] = 1.0

    phase = np.linspace(0.0, 1.0, active_frames, dtype=np.float32)
    trajectory = np.sin(np.pi * phase).astype(np.float32)
    active_slice = slice(rest_frames, rest_frames + active_frames)

    pose = sequence[:, POSE_XYZ_SLICE].reshape(total, len(POSE_INDICES), 3)
    arm_slots = [POSE_INDICES.index(index) for index in (13, 14, 15, 16)]
    pose[active_slice, arm_slots, 0] = trajectory[:, None]
    pose[active_slice, arm_slots, 1] = (0.25 * trajectory)[:, None]

    hand = sequence[:, LEFT_HAND_SLICE].reshape(total, 2, 21, 3)
    hand[active_slice, 0, :, 0] = trajectory[:, None]
    hand[active_slice, 1, :, 0] = (0.5 * trajectory)[:, None]
    timestamps = np.arange(total, dtype=np.float32) / fps
    return sequence, timestamps


def test_v2_layout_and_dimensions_are_explicit() -> None:
    layout = feature_layout()

    assert SCHEMA_VERSION == 2
    assert FEATURE_LAYOUT_VERSION == "mask_aware_motion_v2"
    assert GEOMETRY_DIM == 510
    assert static_feature_dim() == 527
    assert model_feature_dim() == 1_037
    assert model_feature_dim() == static_feature_dim() + GEOMETRY_DIM
    assert PRESENCE_SLICE.stop - PRESENCE_SLICE.start == PRESENCE_DIM
    assert layout["version"] == FEATURE_LAYOUT_VERSION
    assert layout["static_dim"] == 527
    assert layout["geometry_dim"] == 510
    assert layout["model_dim"] == 1_037
    assert layout["max_interpolation_gap_frames"] == MAX_INTERPOLATION_GAP


def test_short_hand_gap_is_filled_but_long_gap_stays_missing() -> None:
    sequence = np.zeros((8, static_feature_dim()), dtype=np.float32)
    _set_modality(
        sequence,
        0,
        LEFT_HAND_SLICE,
        LEFT_HAND_PRESENCE,
        1.0,
    )
    _set_modality(
        sequence,
        4,
        LEFT_HAND_SLICE,
        LEFT_HAND_PRESENCE,
        5.0,
    )
    _set_modality(
        sequence,
        0,
        RIGHT_HAND_SLICE,
        RIGHT_HAND_PRESENCE,
        2.0,
    )
    _set_modality(
        sequence,
        5,
        RIGHT_HAND_SLICE,
        RIGHT_HAND_PRESENCE,
        7.0,
    )

    repaired = fill_short_landmark_gaps(sequence)

    assert np.array_equal(
        repaired[:5, _presence_column(LEFT_HAND_PRESENCE)],
        np.ones(5, dtype=np.float32),
    )
    assert np.allclose(repaired[1:4, LEFT_HAND_SLICE.start], [2.0, 3.0, 4.0])
    assert np.array_equal(
        repaired[1:5, _presence_column(RIGHT_HAND_PRESENCE)],
        np.zeros(4, dtype=np.float32),
    )
    assert np.count_nonzero(repaired[1:5, RIGHT_HAND_SLICE]) == 0

    resampled = resample_sequence(sequence, len(sequence))
    assert np.count_nonzero(resampled[1:5, RIGHT_HAND_SLICE]) == 0
    assert np.count_nonzero(
        resampled[1:5, _presence_column(RIGHT_HAND_PRESENCE)]
    ) == 0


def test_temporal_deltas_are_zero_at_presence_transitions() -> None:
    sequence = np.zeros((5, static_feature_dim()), dtype=np.float32)
    for frame, value in ((1, 10.0), (2, 12.0), (4, 20.0)):
        _set_modality(
            sequence,
            frame,
            LEFT_HAND_SLICE,
            LEFT_HAND_PRESENCE,
            value,
        )

    model_input = add_temporal_deltas(sequence)
    delta_slice = slice(
        static_feature_dim() + LEFT_HAND_SLICE.start,
        static_feature_dim() + LEFT_HAND_SLICE.stop,
    )

    assert np.count_nonzero(model_input[0, delta_slice]) == 0
    assert np.count_nonzero(model_input[1, delta_slice]) == 0
    assert np.allclose(model_input[2, delta_slice], 2.0)
    assert np.count_nonzero(model_input[3, delta_slice]) == 0
    assert np.count_nonzero(model_input[4, delta_slice]) == 0


def test_mask_aware_scaler_keeps_every_missing_modality_at_zero() -> None:
    sequence = np.zeros((4, static_feature_dim()), dtype=np.float32)
    for frame, value in ((0, 1.0), (2, 3.0), (3, 5.0)):
        _set_modality(
            sequence,
            frame,
            LEFT_HAND_SLICE,
            LEFT_HAND_PRESENCE,
            value,
        )
    matrix = add_temporal_deltas(sequence)
    scaler = FeatureStandardizer.fit([matrix])

    assert np.allclose(scaler.mean[LEFT_HAND_SLICE], 3.0)
    left_delta_slice = slice(
        static_feature_dim() + LEFT_HAND_SLICE.start,
        static_feature_dim() + LEFT_HAND_SLICE.stop,
    )
    assert np.allclose(scaler.mean[left_delta_slice], 2.0)

    probe = matrix.copy()
    missing_frame = 1
    probe[missing_frame, :GEOMETRY_DIM] = 123.0
    probe[missing_frame, static_feature_dim():] = 456.0
    probe[missing_frame, POSE_VISIBILITY_SLICE] = 0.75
    probe[missing_frame, PRESENCE_SLICE] = 0.0
    transformed = scaler.transform(probe)

    assert np.count_nonzero(transformed[missing_frame, :GEOMETRY_DIM]) == 0
    assert np.count_nonzero(transformed[missing_frame, static_feature_dim():]) == 0
    assert np.count_nonzero(transformed[missing_frame, POSE_VISIBILITY_SLICE]) == 0


def test_mirroring_is_an_involution_with_asymmetric_masks() -> None:
    rng = np.random.default_rng(17)
    sequence = rng.normal(size=(7, static_feature_dim())).astype(np.float32)
    masks = np.asarray(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    sequence[:, PRESENCE_SLICE] = masks
    sequence[:, POSE_VISIBILITY_SLICE] = rng.uniform(
        0.0,
        1.0,
        size=(len(sequence), len(POSE_INDICES)),
    )
    sequence = fill_short_landmark_gaps(sequence, max_gap=0)

    restored = mirror_sequence(mirror_sequence(sequence))

    assert restored.dtype == np.float32
    assert np.allclose(restored, sequence, atol=1e-6)


def test_rest_gesture_rest_is_segmented_without_losing_the_gesture() -> None:
    rest_frames = 30
    active_frames = 36
    sequence, timestamps = _gesture_sequence(active_frames, rest_frames)

    segmented, segmented_timestamps, metadata = segment_active_sequence(
        sequence,
        timestamps=timestamps,
        padding_seconds=0.25,
        minimum_seconds=0.8,
    )

    assert metadata["fallback"] is False
    assert rest_frames - 8 <= metadata["start_frame"] <= rest_frames + 4
    expected_stop = rest_frames + active_frames
    assert expected_stop - 4 <= metadata["stop_frame"] <= expected_stop + 8
    assert len(segmented) < len(sequence) * 0.75
    assert segmented_timestamps is not None
    assert len(segmented_timestamps) == len(segmented)
    assert np.max(segmented[:, LEFT_HAND_GLOBAL_SLICE.start]) > 0.95


def test_phase_views_are_approximately_speed_invariant() -> None:
    fast_sequence, fast_timestamps = _gesture_sequence(active_frames=24)
    slow_sequence, slow_timestamps = _gesture_sequence(active_frames=48)

    fast_view = deterministic_views(
        fast_sequence,
        48,
        timestamps=fast_timestamps,
    )[1]
    slow_view = deterministic_views(
        slow_sequence,
        48,
        timestamps=slow_timestamps,
    )[1]
    coordinate = LEFT_HAND_GLOBAL_SLICE.start
    fast_trajectory = fast_view[:, coordinate]
    slow_trajectory = slow_view[:, coordinate]

    assert np.max(fast_trajectory) > 0.95
    assert np.max(slow_trajectory) > 0.95
    assert np.corrcoef(fast_trajectory, slow_trajectory)[0, 1] > 0.97
    assert np.mean(np.abs(fast_trajectory - slow_trajectory)) < 0.10
