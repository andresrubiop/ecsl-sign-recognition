import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import pfrutas.jetson as jetson
from pfrutas.features import FeatureStandardizer
from pfrutas.jetson import (
    JETSON_BUNDLE_MANIFEST,
    build_orin_bundle,
    export_slim_checkpoint,
    load_bundle_members,
    summarize_latencies,
    tensor_state_sha256,
    validate_bundle,
)
from pfrutas.model import ModelConfig, SignLSTM


def _checkpoint_payload(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    config = ModelConfig(
        input_dim=4,
        num_classes=2,
        projection_dim=3,
        hidden_dim=2,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    )
    model = SignLSTM(config)
    return {
        "model_state": model.state_dict(),
        "optimizer_state": {
            "state": {0: {"exp_avg": torch.arange(64, dtype=torch.float32)}},
            "param_groups": [],
        },
        "model_config": asdict(config),
        "feature_scaler": FeatureStandardizer(
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ).to_dict(),
        "labels": ["MANGO", "UVILLA"],
        "sequence_length": 6,
        "fold": seed,
        "epoch": 1,
        "metrics": {},
        "preprocessing": {
            "runtime": {"torch_version": torch.__version__},
            "pretraining": {
                "checkpoint_path": "/servidor/privado/ASL_citizen.pt"
            },
        },
    }


def _create_source_run(root: Path) -> Path:
    run_dir = root / "source_run"
    paths = [f"fold_{index}/best.pt" for index in range(5)] + ["final_model.pt"]
    config = _checkpoint_payload()["model_config"]
    for index, relative in enumerate(paths):
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint_payload(index), destination)
    ensemble = {
        "labels": ["MANGO", "UVILLA"],
        "checkpoints": paths,
        "fold_checkpoints": paths[:5],
        "final_checkpoint": paths[-1],
        "sequence_length": 6,
        "model_config": config,
        "cv_metrics": "cv_metrics.json",
        "recommended_confidence_threshold": 0.42,
        "run_fingerprint": "run-hash",
        "dataset_content_fingerprint": "dataset-hash",
    }
    (run_dir / "ensemble.json").write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "cv_metrics.json").write_text(
        json.dumps({"accuracy": 0.8}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _fake_mediapipe_runtime() -> dict:
    return {
        "package": "mediapipe",
        "version": "0.10.21",
        "contract": jetson.MEDIAPIPE_CONTRACT,
        "assets": [
            {
                "path": "modules/holistic_landmark/holistic_landmark_cpu.binarypb",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
        "asset_hash_algorithm": "sha256",
    }


def test_export_slim_checkpoint_is_safe_and_bit_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "slim.pt"
    original = _checkpoint_payload()
    torch.save(original, source)

    info = export_slim_checkpoint(source, destination)
    exported = torch.load(destination, map_location="cpu", weights_only=True)

    assert "optimizer_state" not in exported
    assert type(exported["preprocessing"]["runtime"]["torch_version"]) is str
    assert (
        exported["preprocessing"]["pretraining"]["checkpoint_path"]
        == "ASL_citizen.pt"
    )
    assert info["weights_only_verified"] is True
    assert info["optimizer_state_removed"] is True
    assert info["size_bytes"] < info["source_size_bytes"]
    assert info["tensor_state_sha256"] == tensor_state_sha256(
        original["model_state"]
    )
    for name, tensor in original["model_state"].items():
        assert torch.equal(tensor, exported["model_state"][name])


def test_build_and_validate_six_member_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = _create_source_run(tmp_path)
    bundle_dir = tmp_path / "orin_bundle"
    monkeypatch.setattr(
        jetson,
        "validate_mediapipe_runtime",
        _fake_mediapipe_runtime,
    )

    manifest = build_orin_bundle(run_dir, bundle_dir)
    validated = validate_bundle(bundle_dir)

    assert manifest["schema"] == jetson.JETSON_BUNDLE_SCHEMA
    assert validated["model"]["member_count"] == 6
    assert (bundle_dir / JETSON_BUNDLE_MANIFEST).is_file()
    assert (bundle_dir / "ensemble.json").read_bytes() == (
        run_dir / "ensemble.json"
    ).read_bytes()
    assert (bundle_dir / "cv_metrics.json").read_bytes() == (
        run_dir / "cv_metrics.json"
    ).read_bytes()
    checkpoint_assets = [
        item for item in manifest["assets"] if item["kind"] == "checkpoint"
    ]
    assert len(checkpoint_assets) == 6
    assert all(not Path(item["path"]).is_absolute() for item in manifest["assets"])
    assert all(item["weights_only_verified"] for item in checkpoint_assets)

    full, ensemble, _ = load_bundle_members(bundle_dir, device="cpu", mode="full")
    final, _, _ = load_bundle_members(bundle_dir, device="cpu", mode="final")
    folds, _, _ = load_bundle_members(bundle_dir, device="cpu", mode="folds")
    assert len(full) == 6
    assert len(folds) == 5
    assert len(final) == 1
    assert final[0].relative_path == ensemble["final_checkpoint"]
    assert all("model_state" not in member.payload for member in full)


def test_build_rejects_checkpoint_path_traversal(tmp_path: Path) -> None:
    run_dir = _create_source_run(tmp_path)
    ensemble_path = run_dir / "ensemble.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    ensemble["checkpoints"][0] = "../escape.pt"
    ensemble_path.write_text(json.dumps(ensemble), encoding="utf-8")

    try:
        build_orin_bundle(
            run_dir,
            tmp_path / "bundle",
            include_mediapipe_fingerprint=False,
        )
    except ValueError as exc:
        assert "inseguros" in str(exc)
    else:
        raise AssertionError("Una ruta con .. debe rechazarse.")


def test_validate_bundle_detects_tampered_asset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = _create_source_run(tmp_path)
    bundle_dir = tmp_path / "bundle"
    monkeypatch.setattr(
        jetson,
        "validate_mediapipe_runtime",
        _fake_mediapipe_runtime,
    )
    build_orin_bundle(run_dir, bundle_dir)
    with (bundle_dir / "cv_metrics.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    try:
        validate_bundle(bundle_dir, verify_checkpoints=False)
    except RuntimeError as exc:
        assert "Tamano incorrecto" in str(exc) or "SHA-256 incorrecto" in str(exc)
    else:
        raise AssertionError("Un activo modificado debe invalidar el bundle.")


def test_tensor_state_hash_changes_with_one_value() -> None:
    first = {"weight": torch.tensor([1.0, 2.0])}
    second = {"weight": torch.tensor([1.0, 2.5])}
    assert tensor_state_sha256(first) == tensor_state_sha256(first)
    assert tensor_state_sha256(first) != tensor_state_sha256(second)


def test_summarize_latencies_reports_percentiles_and_fps() -> None:
    result = summarize_latencies([0.01, 0.02, 0.03])
    assert np.isclose(result["mean_ms"], 20.0)
    assert np.isclose(result["p50_ms"], 20.0)
    assert result["p95_ms"] > result["p50_ms"]
    assert np.isclose(result["throughput_fps"], 50.0)


def test_jp5_mediapipe_contract_only_changes_solution_base() -> None:
    jp5_assets = jetson.MEDIAPIPE_ASSET_SHA256_BY_VERSION["0.10.5"]
    jp6_assets = jetson.MEDIAPIPE_ASSET_SHA256_BY_VERSION["0.10.18"]

    assert set(jp5_assets) == set(jp6_assets)
    differing = {
        path
        for path in jp5_assets
        if jp5_assets[path] != jp6_assets[path]
    }
    assert differing == {"python/solution_base.py"}
    assert jp5_assets["python/solution_base.py"] == (
        "0025e315bec76c2b76e1f3daf47eb7cfde07685a2a9c43499d51306e857dfa31"
    )


def test_orin_scripts_share_the_fusion_bundle_default() -> None:
    assert jetson.DEFAULT_ORIN_BUNDLE_DIR == Path("artifacts/orin_nx16_fusion")
