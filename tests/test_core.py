import csv
import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import pfrutas.data as data_module
from pfrutas.data import (
    VideoRecord,
    dataset_summary,
    discover_raw_videos,
    infer_signer_id,
    parse_video_filename,
    prepare_dataset,
)
from pfrutas.features import (
    add_temporal_deltas,
    model_feature_dim,
    resample_sequence,
    static_feature_dim,
)
from pfrutas.training import (
    make_folds,
    make_inner_folds,
    save_roc_artifacts,
    select_epoch_from_inner_histories,
)


def test_signer_inference_matches_dataset_convention() -> None:
    for signer in range(5):
        assert infer_signer_id(Path(f"Mango{signer}.mp4")) == f"signer_{signer}"
        assert infer_signer_id(Path(f"Mango{signer}.MP4")) == f"signer_{signer}"
    assert infer_signer_id(Path("Ciruela0_x.MP4")) == "signer_0"
    assert infer_signer_id(Path("Guaba1_r.mp4")) == "signer_1"


def test_video_name_parser_preserves_tags_and_validates_label() -> None:
    excluded = parse_video_filename(
        Path("Tomate_de_Árbol0_x.mp4"),
        expected_label="TOMATE_DE_ÁRBOL",
    )
    assert excluded.label_stem == "Tomate_de_Árbol"
    assert excluded.signer_id == "signer_0"
    assert excluded.tags == ("x",)
    assert excluded.excluded is True

    included = parse_video_filename(
        Path("Guaba1_R.MP4"),
        expected_label="GUABA",
    )
    assert included.signer_id == "signer_1"
    assert included.tags == ("r",)
    assert included.excluded is False


def _assert_value_error(callback: Callable[[], object]) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Se esperaba ValueError.")


def test_video_name_parser_rejects_ambiguous_or_inconsistent_names() -> None:
    invalid = (
        "Mango.mp4",
        "Mango5.mp4",
        "Mango01.mp4",
        "Mango0_.mp4",
        "Mango0_otro.mp4",
        "Mango0_x_x.mp4",
    )
    for name in invalid:
        _assert_value_error(lambda name=name: parse_video_filename(Path(name)))
    _assert_value_error(
        lambda: parse_video_filename(Path("Mango0.mp4"), expected_label="ORITO")
    )


def test_discovery_excludes_x_by_default_and_keeps_r_metadata() -> None:
    with TemporaryDirectory() as temporary:
        raw_dir = Path(temporary)
        mango_dir = raw_dir / "MANGO"
        mango_dir.mkdir()
        for name in ("Mango0.mp4", "Mango1_r.MP4", "Mango2_x.mp4"):
            (mango_dir / name).touch()

        labels, included = discover_raw_videos(raw_dir)
        assert labels == ["MANGO"]
        assert [row["source"] for row in included] == [
            "MANGO/Mango0.mp4",
            "MANGO/Mango1_r.MP4",
        ]
        assert included[1]["signer_id"] == "signer_1"
        assert included[1]["tags"] == ["r"]
        assert included[1]["excluded"] is False

        _, inventory = discover_raw_videos(raw_dir, include_excluded=True)
        assert len(inventory) == 3
        excluded = next(row for row in inventory if row["excluded"])
        assert excluded["source"] == "MANGO/Mango2_x.mp4"
        assert excluded["signer_id"] == "signer_2"
        assert excluded["tags"] == ["x"]

        summary = dataset_summary(raw_dir)
        assert summary["raw_video_count"] == 3
        assert summary["video_count"] == 2
        assert summary["included_video_count"] == 2
        assert summary["excluded_video_count"] == 1
        assert summary["videos_per_signer"] == {
            "signer_0": 1,
            "signer_1": 1,
        }
        assert summary["tag_counts"] == {"r": 1, "x": 1}
        assert summary["excluded_videos"][0]["tags"] == ["x"]


def test_discovery_rejects_filename_label_mismatch() -> None:
    with TemporaryDirectory() as temporary:
        class_dir = Path(temporary) / "MANGO"
        class_dir.mkdir()
        (class_dir / "Orito0.mp4").touch()
        _assert_value_error(lambda: discover_raw_videos(Path(temporary)))


def test_prepare_registers_excluded_files_without_processing_them() -> None:
    class FakeExtractor:
        def __enter__(self) -> "FakeExtractor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_extract_video(*args: object, **kwargs: object) -> tuple[object, ...]:
        features = np.zeros((4, static_feature_dim()), dtype=np.float32)
        timestamps = np.arange(4, dtype=np.float32)
        metadata = {
            "duration": 1.0,
            "quality": {"pose": 1.0},
        }
        return features, timestamps, metadata

    original_extractor = data_module.HolisticFeatureExtractor
    original_extract_video = data_module.extract_video
    data_module.HolisticFeatureExtractor = FakeExtractor  # type: ignore[assignment]
    data_module.extract_video = fake_extract_video  # type: ignore[assignment]
    try:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            output_dir = root / "processed"
            mango_dir = raw_dir / "MANGO"
            mango_dir.mkdir(parents=True)
            for name in ("Mango0.mp4", "Mango1_r.mp4", "Mango2_x.mp4"):
                (mango_dir / name).touch()

            manifest = prepare_dataset(raw_dir, output_dir)
            assert [row["source"] for row in manifest["videos"]] == [
                "MANGO/Mango0.mp4",
                "MANGO/Mango1_r.mp4",
            ]
            assert manifest["videos"][1]["tags"] == ["r"]
            assert manifest["excluded_videos"][0]["source"] == "MANGO/Mango2_x.mp4"
            assert manifest["excluded_videos"][0]["tags"] == ["x"]

            included_manifest = prepare_dataset(
                raw_dir,
                root / "processed_with_x",
                include_excluded=True,
            )
            assert len(included_manifest["videos"]) == 3
            x_record = next(
                row for row in included_manifest["videos"] if row["tags"] == ["x"]
            )
            assert x_record["signer_id"] == "signer_2"
            assert x_record["excluded"] is True
            assert included_manifest["excluded_videos"] == []
    finally:
        data_module.HolisticFeatureExtractor = original_extractor
        data_module.extract_video = original_extract_video


def test_resampling_and_deltas_have_expected_shape() -> None:
    source = np.random.default_rng(1).normal(size=(23, static_feature_dim())).astype(np.float32)
    sampled = resample_sequence(source, 48)
    with_deltas = add_temporal_deltas(sampled)
    assert sampled.shape == (48, static_feature_dim())
    assert with_deltas.shape == (48, model_feature_dim())
    assert np.allclose(with_deltas[0, static_feature_dim():], 0.0)


def _record(signer: int, label: int) -> VideoRecord:
    video_id = f"s{signer}_c{label}"
    return VideoRecord(
        source=f"{video_id}.mp4",
        sequence=f"{video_id}.npz",
        label=f"C{label}",
        label_index=label,
        signer_id=f"signer_{signer}",
        video_id=video_id,
        frames=32,
        duration=2.0,
        quality={},
    )


def test_nested_loso_covers_each_video_without_signer_leakage() -> None:
    records = [_record(signer, label) for signer in range(5) for label in range(2)]
    outer_test_ids: list[str] = []
    for outer_train, outer_test, _ in make_folds(records, 5):
        assert not ({r.signer_id for r in outer_train} & {r.signer_id for r in outer_test})
        inner_validation_ids: list[str] = []
        for inner_train, inner_validation, _ in make_inner_folds(outer_train):
            assert not (
                {r.signer_id for r in inner_train}
                & {r.signer_id for r in inner_validation}
            )
            inner_validation_ids.extend(r.video_id for r in inner_validation)
        assert sorted(inner_validation_ids) == sorted(r.video_id for r in outer_train)
        outer_test_ids.extend(r.video_id for r in outer_test)
    assert sorted(outer_test_ids) == sorted(r.video_id for r in records)


def test_inner_epoch_selection_uses_pooled_oof_predictions() -> None:
    selections = []
    for _ in range(2):
        selections.append(
            {
                "history": [
                    {
                        "val_macro_f1": 0.3,
                        "val_accuracy": 0.5,
                        "y_true": [0, 1],
                        "predictions": [0, 0],
                    },
                    {
                        "val_macro_f1": 1.0,
                        "val_accuracy": 1.0,
                        "y_true": [0, 1],
                        "predictions": [0, 1],
                    },
                    {
                        "val_macro_f1": 0.0,
                        "val_accuracy": 0.0,
                        "y_true": [0, 1],
                        "predictions": [1, 0],
                    },
                ]
            }
        )
    selected_epoch, curve = select_epoch_from_inner_histories(selections)
    assert selected_epoch == 2
    assert curve[1]["pooled_inner_macro_f1"] == 1.0


def test_save_roc_artifacts_exports_ovr_curves_and_summary() -> None:
    labels = ["A", "B", "C"]
    y_true = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.80, 0.10, 0.10],
            [0.05, 0.90, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.05, 0.90],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float32,
    )
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        summary = save_roc_artifacts(run_dir, y_true, probabilities, labels)

        assert summary["curve_type"] == "one-vs-rest"
        assert summary["video_count"] == 6
        assert summary["macro_ovr_auc"] == 1.0
        assert summary["weighted_ovr_auc"] == 1.0
        assert summary["micro_ovr_auc"] == 1.0
        assert summary["per_class"]["B"] == {
            "auc_ovr": 1.0,
            "support": 2,
            "negative_support": 4,
        }
        assert (run_dir / "roc_curves.png").stat().st_size > 0

        persisted = json.loads((run_dir / "roc_metrics.json").read_text(encoding="utf-8"))
        assert persisted == summary
        with (run_dir / "roc_auc_per_class.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["fruit"] for row in rows] == labels
        assert [row["support"] for row in rows] == ["2", "2", "2"]
