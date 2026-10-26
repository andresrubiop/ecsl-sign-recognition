from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import pfrutas.data as data_module
from pfrutas.data import load_manifest, prepare_transfer_dataset
from pfrutas.features import (
    HolisticFeatureExtractor,
    POSE_HANDS_XY_LAYOUT_VERSION,
    POSE_HANDS_XY_POINT_COUNT,
    static_feature_dim,
)


def _landmark_list(count: int, offset: float) -> SimpleNamespace:
    return SimpleNamespace(
        landmark=[
            SimpleNamespace(
                x=offset + index * 0.001,
                y=offset + index * 0.002,
                z=offset - index * 0.001,
                visibility=0.9,
            )
            for index in range(count)
        ]
    )


def test_multimodal_extractor_uses_one_holistic_pass_and_exact_order() -> None:
    calls = 0
    pose_landmarks = _landmark_list(33, 0.10)
    right_landmarks = _landmark_list(21, 0.30)
    result = SimpleNamespace(
        pose_landmarks=pose_landmarks,
        right_hand_landmarks=right_landmarks,
        left_hand_landmarks=None,
        face_landmarks=_landmark_list(468, 0.20),
    )

    class FakeHolistic:
        def process(self, frame: np.ndarray) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            return result

    extractor = HolisticFeatureExtractor.__new__(HolisticFeatureExtractor)
    extractor._holistic = FakeHolistic()  # type: ignore[attr-defined]
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    features, quality, pose_hands_xy = extractor.process_multimodal(rgb)
    assert calls == 1
    assert features.shape == (static_feature_dim(),)
    assert pose_hands_xy.shape == (POSE_HANDS_XY_POINT_COUNT, 2)
    assert pose_hands_xy.dtype == np.float32
    expected_pose = np.asarray(
        [(point.x, point.y) for point in pose_landmarks.landmark],
        dtype=np.float32,
    )
    expected_right = np.asarray(
        [(point.x, point.y) for point in right_landmarks.landmark],
        dtype=np.float32,
    )
    assert np.allclose(pose_hands_xy[:33], expected_pose)
    assert np.allclose(pose_hands_xy[33:54], expected_right)
    assert np.allclose(pose_hands_xy[54:75], 0.0)
    assert quality == {
        "pose": 1.0,
        "face": 1.0,
        "left_hand": 0.0,
        "right_hand": 1.0,
    }

    legacy_features, legacy_quality = extractor.process_rgb(rgb)
    assert calls == 2  # una sola llamada adicional, no dos
    assert np.allclose(legacy_features, features)
    assert legacy_quality == quality


def test_prepare_transfer_cache_contract_exclusion_and_validation() -> None:
    extraction_count = 0

    class FakeExtractor:
        def __enter__(self) -> "FakeExtractor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_extract_transfer_video(
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        nonlocal extraction_count
        extraction_count += 1
        frame_count = 4
        features = np.full(
            (frame_count, static_feature_dim()),
            extraction_count,
            dtype=np.float32,
        )
        timestamps = np.arange(frame_count, dtype=np.float32) / 30.0
        pose_hands_xy = np.full(
            (frame_count, POSE_HANDS_XY_POINT_COUNT, 2),
            0.25,
            dtype=np.float32,
        )
        metadata: dict[str, object] = {
            "duration": 1.0,
            "quality": {"pose": 1.0},
        }
        return features, timestamps, pose_hands_xy, metadata

    original_extractor = data_module.HolisticFeatureExtractor
    original_extract = data_module.extract_transfer_video
    data_module.HolisticFeatureExtractor = FakeExtractor  # type: ignore[assignment]
    data_module.extract_transfer_video = fake_extract_transfer_video  # type: ignore[assignment]
    try:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            output_dir = root / "processed_asl_transfer_v1"
            mango_dir = raw_dir / "MANGO"
            mango_dir.mkdir(parents=True)
            for name in ("Mango0.mp4", "Mango1_r.mp4", "Mango2_x.mp4"):
                (mango_dir / name).touch()

            manifest = prepare_transfer_dataset(raw_dir, output_dir)
            assert extraction_count == 2
            assert manifest["transfer_contract"] == POSE_HANDS_XY_LAYOUT_VERSION
            assert manifest["transfer_layout"]["shape"] == ["frames", 75, 2]
            assert manifest["status"] == "complete"
            assert manifest["inventory_video_count"] == 2
            assert manifest["expected_video_count"] == 2
            assert manifest["selected_video_count"] == 2
            assert manifest["processed_video_count"] == 2
            assert manifest["max_videos"] is None
            assert manifest["limited_for_testing"] is False
            assert [row["source"] for row in manifest["videos"]] == [
                "MANGO/Mango0.mp4",
                "MANGO/Mango1_r.mp4",
            ]
            assert manifest["excluded_videos"][0]["source"] == "MANGO/Mango2_x.mp4"

            loaded_manifest, records = load_manifest(output_dir)
            assert loaded_manifest["transfer_contract"] == POSE_HANDS_XY_LAYOUT_VERSION
            assert len(records) == 2
            first_sequence = output_dir / records[0].sequence
            with np.load(first_sequence, allow_pickle=False) as payload:
                assert set(payload.files) == {
                    "features",
                    "timestamps",
                    "pose_hands_xy",
                }
                assert payload["pose_hands_xy"].shape == (4, 75, 2)

            # Un segundo llamado reutiliza ambos caches válidos.
            prepare_transfer_dataset(raw_dir, output_dir)
            assert extraction_count == 2

            # Un tensor no finito invalida solo esa secuencia y fuerza reextracción.
            with np.load(first_sequence, allow_pickle=False) as payload:
                cached_features = payload["features"].copy()
                cached_timestamps = payload["timestamps"].copy()
                cached_pose = payload["pose_hands_xy"].copy()
            cached_pose[0, 0, 0] = np.nan
            with first_sequence.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    features=cached_features,
                    timestamps=cached_timestamps,
                    pose_hands_xy=cached_pose,
                )
            prepare_transfer_dataset(raw_dir, output_dir)
            assert extraction_count == 3

            # Timestamps duplicados/reordenados también invalidan el cache.
            with np.load(first_sequence, allow_pickle=False) as payload:
                cached_features = payload["features"].copy()
                cached_timestamps = payload["timestamps"].copy()
                cached_pose = payload["pose_hands_xy"].copy()
            cached_timestamps[1] = cached_timestamps[0]
            with first_sequence.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    features=cached_features,
                    timestamps=cached_timestamps,
                    pose_hands_xy=cached_pose,
                )
            prepare_transfer_dataset(raw_dir, output_dir)
            assert extraction_count == 4

            # --max-videos sirve solo para smoke tests: nunca puede presentarse
            # como un dataset completo y, por tanto, no es entrenable.
            limited_manifest = prepare_transfer_dataset(
                raw_dir,
                root / "processed_asl_transfer_limited",
                max_videos=1,
            )
            assert extraction_count == 5
            assert limited_manifest["status"] == "partial"
            assert limited_manifest["inventory_video_count"] == 2
            assert limited_manifest["expected_video_count"] == 2
            assert limited_manifest["selected_video_count"] == 1
            assert limited_manifest["processed_video_count"] == 1
            assert limited_manifest["max_videos"] == 1
            assert limited_manifest["limited_for_testing"] is True
    finally:
        data_module.HolisticFeatureExtractor = original_extractor
        data_module.extract_transfer_video = original_extract
