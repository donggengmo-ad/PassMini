import json

import numpy as np
import pytest

from app.package_artifacts import (
    _atomic_copy,
    _atomic_copy_coverage,
    package_artifacts,
)


def test_package_artifacts_uses_tier_scoped_evaluation(tmp_path):
    output_root = tmp_path / "output"
    model_source = output_root / "medium" / "gru"
    evaluation_source = output_root / "evaluation" / "medium" / "gru"
    model_source.mkdir(parents=True)
    evaluation_source.mkdir(parents=True)
    for filename in (
        "model.pt",
        "tokenizer.json",
        "inference.json",
        "history.json",
        "config.json",
    ):
        (model_source / filename).write_text(filename, encoding="utf-8")
    np.savez(evaluation_source / "surprisal.npz", value=np.asarray([1]))
    attempts = np.arange(100, 1_000_001, 100)
    coverage = attempts / attempts[-1]
    for filename in ("random_coverage.npz", "best_first_coverage.npz"):
        np.savez(
            evaluation_source / filename,
            test_size=np.asarray(100),
            attempts=attempts,
            coverage=coverage,
        )
    (evaluation_source / "summary.json").write_text("summary.json", encoding="utf-8")

    catalog_path = tmp_path / "app" / "artifacts" / "catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "medium-gru",
                        "tier": "medium",
                        "model_type": "gru",
                        "display_name": "GRU · Medium",
                        "artifact_dir": "artifacts/medium/gru",
                        "parameter_count": 10,
                        "flops": 20,
                        "training_sample_count": 30,
                        "color": "#000000",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = package_artifacts(output_root, catalog_path)
    destination = tmp_path / "app" / "artifacts" / "medium" / "gru"

    assert report["medium-gru"] == [
        "model.pt",
        "tokenizer.json",
        "inference.json",
        "history.json",
        "evaluation/surprisal.npz",
        "evaluation/random_coverage.npz",
        "evaluation/best_first_coverage.npz",
        "evaluation/summary.json",
    ]
    assert (destination / "model.pt").read_text(encoding="utf-8") == "model.pt"
    assert (destination / "evaluation" / "summary.json").read_text(
        encoding="utf-8"
    ) == "summary.json"
    assert not (destination / "config.json").exists()


def test_atomic_copy_rejects_corrupt_npz_without_replacing_destination(tmp_path):
    source = tmp_path / "source.npz"
    destination = tmp_path / "destination.npz"
    source.write_bytes(b"incomplete archive")
    destination.write_bytes(b"previous valid deployment")

    with pytest.raises(ValueError, match="NPZ 文件无效"):
        _atomic_copy(source, destination)

    assert destination.read_bytes() == b"previous valid deployment"
    assert not (tmp_path / "destination.npz.tmp").exists()


def test_atomic_copy_coverage_limits_points_and_preserves_budget(tmp_path):
    source = tmp_path / "best_first_coverage.npz"
    destination = tmp_path / "deployed.npz"
    attempts = np.arange(100, 1_000_001, 100)
    np.savez(
        source,
        test_size=np.asarray(100),
        attempts=attempts,
        coverage=attempts / attempts[-1],
    )

    _atomic_copy_coverage(source, destination, max_points=100)

    with np.load(destination, allow_pickle=False) as payload:
        deployed_attempts = payload["attempts"]
        assert deployed_attempts.size <= 100
        assert deployed_attempts[0] == 100
        assert deployed_attempts[-1] == 1_000_000
        assert 500_000 in deployed_attempts


def test_package_artifacts_prunes_obsolete_deployment_files(tmp_path):
    output_root = tmp_path / "output"
    model_source = output_root / "bigram"
    evaluation_source = output_root / "evaluation" / "baseline" / "bigram"
    model_source.mkdir(parents=True)
    evaluation_source.mkdir(parents=True)
    for filename in ("model.pt", "tokenizer.json", "inference.json"):
        (model_source / filename).write_text(filename, encoding="utf-8")

    catalog_path = tmp_path / "app" / "artifacts" / "catalog.json"
    destination = catalog_path.parent / "baseline" / "bigram"
    (destination / "evaluation").mkdir(parents=True)
    (destination / "config.json").write_text("stale", encoding="utf-8")
    (destination / "history.json").write_text("stale", encoding="utf-8")
    (destination / "evaluation" / "best_first.json").write_text(
        "stale", encoding="utf-8"
    )
    (catalog_path.parent / ".DS_Store").write_text("stale", encoding="utf-8")
    catalog_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "baseline-bigram",
                        "tier": "baseline",
                        "model_type": "bigram",
                        "display_name": "Bigram · Baseline",
                        "artifact_dir": "artifacts/baseline/bigram",
                        "parameter_count": 0,
                        "flops": 1,
                        "training_sample_count": 1,
                        "color": "#000000",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    package_artifacts(output_root, catalog_path, include_evaluation=False)

    assert not (destination / "config.json").exists()
    assert not (destination / "history.json").exists()
    assert not (destination / "evaluation" / "best_first.json").exists()
    assert not (catalog_path.parent / ".DS_Store").exists()


def test_package_artifacts_packages_high_model_without_evaluation(tmp_path):
    output_root = tmp_path / "output"
    model_source = output_root / "high" / "transformer"
    model_source.mkdir(parents=True)
    model_filenames = (
        "model.pt",
        "tokenizer.json",
        "inference.json",
        "history.json",
    )
    for filename in model_filenames:
        (model_source / filename).write_text(filename, encoding="utf-8")

    catalog_path = tmp_path / "app" / "artifacts" / "catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "high-transformer",
                        "tier": "high",
                        "model_type": "transformer",
                        "display_name": "Transformer · High",
                        "artifact_dir": "artifacts/high/transformer",
                        "parameter_count": 10,
                        "flops": 20,
                        "training_sample_count": 30,
                        "color": "#000000",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = package_artifacts(output_root, catalog_path)
    destination = tmp_path / "app" / "artifacts" / "high" / "transformer"

    assert report["high-transformer"] == list(model_filenames)
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        model_filenames
    )
