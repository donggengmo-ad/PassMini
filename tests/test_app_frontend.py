import json
from pathlib import Path

import numpy as np
import pytest

from app.frontend.catalog import load_catalog
from app.frontend.probability_input import colored_password_html, probability_color
from app.frontend.library import (
    CoverageData,
    SurprisalData,
    TrainingHistory,
    load_coverage_data,
    load_generation_quality,
    load_surprisal_data,
    load_training_history,
    mean_coverage,
    mean_surprisal,
    plot_coverage,
    plot_generation_quality,
    plot_learning_rate,
    plot_surprisal_boxplot,
    plot_surprisal_histogram,
    plot_training_history,
)
from app.frontend.playground import (
    aggregate_scores,
    complete_with_models,
    generate_with_models,
    score_models,
    trace_character_probabilities,
)
from app.frontend.warehouse import get_runtime_model, read_model_metadata
from scripts.models import AutoregressiveBigram
from scripts.tokenizer import CharTokenizer


def test_default_catalog_and_artifacts_are_consistent():
    catalog = load_catalog()

    assert [record.id for record in catalog.enabled_models] == [
        "baseline-bigram",
        "low-gru",
        "low-tcn",
        "low-transformer",
    ]
    assert len({record.id for record in catalog.models}) == len(catalog.models)
    assert all(record.inference_ready for record in catalog.enabled_models)


def test_catalog_rejects_artifact_path_outside_app(tmp_path):
    app_dir = tmp_path / "app"
    catalog_dir = app_dir / "artifacts"
    catalog_dir.mkdir(parents=True)
    path = catalog_dir / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "escape",
                        "tier": "low",
                        "model_type": "gru",
                        "display_name": "Escape",
                        "artifact_dir": "../../outside",
                        "parameter_count": 1,
                        "color": "#000000",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="app 目录内"):
        load_catalog(path)


def test_enabled_models_load_on_cpu_with_expected_parameter_counts():
    catalog = load_catalog()

    for record in catalog.enabled_models:
        model, tokenizer = get_runtime_model(record)
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
        assert model.device.type == "cpu"
        assert model.tokenizer == tokenizer
        assert trainable_parameters == record.parameter_count


def test_warehouse_metadata_uses_history_without_loading_model():
    record = load_catalog().by_id("low-gru")

    metadata = read_model_metadata(record)

    assert metadata.epochs == 50
    assert metadata.best_validation_loss == pytest.approx(min(metadata.history["valid_loss"]))
    assert metadata.files["model"] is True


def test_library_loaders_and_aggregates(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "train_loss": [2.0, 1.0],
                "valid_loss": [2.2, 1.2],
                "learning_rate": [0.1, 0.05],
                "test_loss": 1.3,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        tmp_path / "surprisal.npz",
        evaluation_size=np.asarray(100),
        surprisal_bits=np.asarray([8.0, 10.0]),
        bits_per_token=np.asarray([2.0, 2.5]),
    )
    np.savez_compressed(
        tmp_path / "coverage.npz",
        test_size=np.asarray(10),
        attempts=np.asarray([10, 20]),
        coverage=np.asarray([0.1, 0.2]),
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "generation_quality": {
                    "total_samples": 100,
                    "legal_rate": 0.9,
                    "legal_unique_rate": 0.75,
                }
            }
        ),
        encoding="utf-8",
    )

    history = load_training_history(history_path)
    surprisal = load_surprisal_data(tmp_path / "surprisal.npz")
    coverage = load_coverage_data(tmp_path / "coverage.npz")
    quality = load_generation_quality(tmp_path / "summary.json")

    assert history.test_loss == pytest.approx(1.3)
    assert surprisal.evaluation_size == 100
    assert coverage.efficiency.tolist() == pytest.approx([0.01, 0.01])
    assert quality.legal_unique_rate == pytest.approx(0.75)

    averaged_surprisal = mean_surprisal(
        {
            "a": surprisal,
            "b": SurprisalData(100, np.asarray([10.0, 12.0]), np.asarray([3.0, 3.5])),
        }
    )
    averaged_coverage = mean_coverage(
        {
            "a": coverage,
            "b": CoverageData(10, np.asarray([10, 20]), np.asarray([0.2, 0.4])),
        }
    )
    assert averaged_surprisal.surprisal_bits.tolist() == pytest.approx([9.0, 11.0])
    assert averaged_coverage.coverage.tolist() == pytest.approx([0.15, 0.3])


def test_library_rejects_incompatible_aggregates():
    with pytest.raises(ValueError, match="评测规模"):
        mean_surprisal(
            {
                "a": SurprisalData(10, np.ones(2), np.ones(2)),
                "b": SurprisalData(20, np.ones(2), np.ones(2)),
            }
        )
    with pytest.raises(ValueError, match="attempts"):
        mean_coverage(
            {
                "a": CoverageData(10, np.asarray([10, 20]), np.asarray([0.1, 0.2])),
                "b": CoverageData(10, np.asarray([10, 30]), np.asarray([0.1, 0.2])),
            }
        )


def test_library_charts_use_english_titles(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps({"train_loss": [2.0], "valid_loss": [2.1]}), encoding="utf-8"
    )
    history = load_training_history(history_path)
    surprisal = SurprisalData(2, np.asarray([8.0, 9.0]), np.asarray([2.0, 2.2]))
    coverage = CoverageData(10, np.asarray([10, 20]), np.asarray([0.1, 0.2]))
    quality_path = tmp_path / "summary.json"
    quality_path.write_text(
        json.dumps(
            {
                "total_samples": 10,
                "legal_rate": 0.9,
                "legal_unique_rate": 0.8,
            }
        ),
        encoding="utf-8",
    )
    quality = load_generation_quality(quality_path)
    charts = [
        plot_training_history({"GRU": history}, {"GRU": "#000000"}),
        plot_learning_rate(
            {
                "GRU": type(history)(
                    history.train_loss,
                    history.valid_loss,
                    np.asarray([0.1]),
                    history.test_loss,
                )
            },
            {"GRU": "#000000"},
        ),
        plot_surprisal_histogram({"GRU": surprisal}, {"GRU": "#000000"}),
        plot_surprisal_boxplot({"GRU": surprisal}, {"GRU": "#000000"}),
        plot_coverage({"GRU": coverage}, {"GRU": "#000000"}),
        plot_generation_quality({"GRU": quality}, {"GRU": "#000000"}),
    ]

    assert [chart.to_dict()["title"] for chart in charts] == [
        "Training History",
        "Learning Rate Schedule",
        "Surprisal Distribution",
        "Surprisal Comparison",
        "Coverage by Search Attempts",
        "Random Generation Quality",
    ]


def test_coverage_chart_uniformly_limits_points_and_keeps_endpoints():
    attempts = np.arange(1, 10_002)
    coverage = attempts / attempts[-1]
    chart = plot_coverage(
        {"GRU": CoverageData(20_000, attempts, coverage)},
        {"GRU": "#000000"},
        max_points_per_model=100,
    )

    spec = chart.to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert len(rows) == 100
    assert rows[0]["Attempts"] == 1
    assert rows[-1]["Attempts"] == 10_001


def test_chart_positive_axes_start_at_zero_and_coverage_uses_percentages():
    history = TrainingHistory(
        train_loss=np.asarray([2.0, 1.5]),
        valid_loss=np.asarray([2.1, 1.6]),
        learning_rate=np.asarray([0.01, 0.005]),
        test_loss=1.7,
    )
    surprisal = SurprisalData(2, np.asarray([8.0, 9.0]), np.asarray([2.0, 2.2]))
    coverage = CoverageData(10, np.asarray([10, 20]), np.asarray([0.1, 0.2]))
    colors = {"GRU": "#000000"}

    training_spec = plot_training_history({"GRU": history}, colors).to_dict()
    histogram_spec = plot_surprisal_histogram({"GRU": surprisal}, colors).to_dict()
    coverage_spec = plot_coverage({"GRU": coverage}, colors).to_dict()
    efficiency_spec = plot_coverage(
        {"GRU": coverage}, colors, efficiency=True
    ).to_dict()

    assert training_spec["encoding"]["x"]["scale"]["domainMin"] == 0
    assert training_spec["encoding"]["y"]["scale"] == {"zero": False}
    for spec in (histogram_spec, coverage_spec, efficiency_spec):
        assert spec["encoding"]["x"]["scale"]["domainMin"] == 0
        assert spec["encoding"]["y"]["scale"]["domainMin"] == 0
    for spec in (training_spec, histogram_spec, coverage_spec, efficiency_spec):
        assert "params" not in spec
    assert coverage_spec["encoding"]["y"]["axis"]["format"] == ".1%"
    assert coverage_spec["encoding"]["tooltip"][2]["format"] == ".1%"
    assert "axis" not in efficiency_spec["encoding"]["y"]
    assert efficiency_spec["encoding"]["tooltip"][2]["format"] == ".6g"


def test_playground_services_with_bigram_artifact():
    record = load_catalog().by_id("baseline-bigram")
    runtime = {record.id: get_runtime_model(record)}

    scores = score_models("passmini", runtime)
    aggregate = aggregate_scores(scores)
    generated = generate_with_models(runtime, 3, 8, 1.0, 2026)
    completed = complete_with_models(runtime, "pass", 3, 8, 1.0)

    assert len(scores) == 1
    assert np.isfinite(aggregate.mean_surprisal_bits)
    assert len(generated[record.id]) == 3
    assert 1 <= len(completed[record.id]) <= 3
    assert all(candidate.text.startswith("pass") for candidate in completed[record.id])


def test_character_probability_trace_uses_incremental_context():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveBigram(tokenizer, alpha=1.0)
    a_id = tokenizer.token_to_id["a"]
    b_id = tokenizer.token_to_id["b"]
    c_id = tokenizer.token_to_id["c"]
    model.count[tokenizer.bos_id, a_id] = 9
    model.count[a_id, b_id] = 4
    model.count[b_id, c_id] = 7
    runtime = {"bigram": (model, tokenizer)}

    short_trace = trace_character_probabilities("a", runtime, top_k=3)[0]
    trace = trace_character_probabilities("ab", runtime, top_k=3)[0]

    assert trace.characters[0].probability == pytest.approx(10 / 13)
    assert trace.characters[1].probability == pytest.approx(5 / 8)
    assert short_trace.characters[0] == trace.characters[0]
    assert trace.next_tokens[0].token == "c"
    assert trace.next_tokens[0].probability == pytest.approx(8 / 11)


def test_probability_color_and_html_are_continuous_and_escaped():
    assert probability_color(0.0) == "#ef4444"
    assert probability_color(0.5) == "#f59e0b"
    assert probability_color(1.0) == "#22c55e"
    assert probability_color(0.25) not in {"#ef4444", "#f59e0b"}

    markup = colored_password_html([("<", 0.5), (" ", 0.1)])
    assert ">&lt;</span>" in markup
    assert "&nbsp;" in markup
    assert "50.00%" in markup


@pytest.mark.parametrize("password", ["", "密码", "a" * 13])
def test_playground_rejects_inputs_outside_dataset_contract(password):
    with pytest.raises(ValueError, match="可打印 ASCII"):
        score_models(password, {})
