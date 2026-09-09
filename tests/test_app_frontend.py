import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.frontend.catalog import load_catalog
from app.frontend.components import preset_model_ids
from app.frontend.model_icons import model_icon_html
from app.frontend.probability_input import colored_password_html, probability_color
from app.frontend.library import (
    CoverageData,
    EvaluationSummary,
    MetricSummary,
    SurprisalData,
    TrainingHistory,
    build_research_overview,
    coverage_at_budget,
    load_coverage_data,
    load_evaluation_summary,
    load_generation_quality,
    load_surprisal_data,
    load_training_history,
    mean_coverage,
    mean_surprisal,
    pairwise_difference,
    pairwise_win_rates,
    plot_coverage,
    plot_generation_quality,
    plot_generalization_gap,
    plot_learning_rate,
    plot_pairwise_difference,
    plot_pairwise_win_rates,
    plot_model_zoo,
    plot_surprisal_boxplot,
    plot_surprisal_histogram,
    plot_training_history,
    plot_validation_loss_by_time,
)
from app.frontend.playground import (
    PasswordScore,
    aggregate_scores,
    complete_with_models,
    completion_consensus_frame,
    completion_consensus_summary,
    distribution_entropy,
    generate_challenge_password,
    generate_with_models,
    model_disagreement_matrix,
    plot_completion_consensus,
    plot_model_disagreement,
    plot_probability_keyboard,
    plot_score_comparison,
    plot_surprisal_journey,
    plot_temperature_laboratory,
    score_models,
    trace_character_probabilities,
)
from app.frontend.warehouse import get_runtime_model, read_model_metadata
from scripts.models import AutoregressiveBigram
from scripts.inference import GenerationCandidate
from scripts.tokenizer import CharTokenizer


@pytest.mark.parametrize(
    "model_type", ["bigram", "mlp", "tcn", "gru", "transformer"]
)
def test_model_family_icons_are_accessible_css_only_svg(model_type):
    html = model_icon_html(model_type)

    assert '<svg viewBox="0 0 ' in html
    assert 'role="img"' in html
    assert "aria-label=" in html
    assert ".pm-model-icon:hover" in html
    assert "prefers-reduced-motion: reduce" in html
    assert "<script" not in html


def test_gru_icon_uses_one_cell_and_transformer_traverses_five_queries():
    gru = model_icon_html("gru")
    transformer = model_icon_html("transformer")

    assert gru.count('<rect class="pm-cell"') == 1
    assert gru.count('<circle class="pm-state-shell"') == 1
    assert 'class="pm-state-liquid"' in gru
    assert 'class="pm-flow pm-gru-drain"' in gru
    assert 'class="pm-flow pm-gru-state-return"' in gru
    assert "hₜ₋₁" not in gru
    assert ">hₜ<" not in gru
    assert transformer.count('<circle class="pm-node pm-q ') == 5
    assert transformer.count('<circle class="pm-node pm-k pm-weight-') == 5
    assert transformer.count('<circle class="pm-node pm-v pm-weight-') == 5
    assert transformer.count('<g class="pm-attention-fan pm-fan-') == 5


def test_mlp_icon_keeps_static_wires_under_seamless_flow_animation():
    html = model_icon_html("mlp")

    assert "pm-token-text" not in html
    assert html.count('<rect class="pm-token"') == 8
    assert '<g clip-path="url(#pm-mlp-window)">' in html
    assert '<g class="pm-mlp-tokens">' in html
    assert "to { transform:translateX(260px) }" in html
    assert 'class="pm-wire"' in html
    assert 'class="pm-mlp-flow pm-mlp-flow-a"' in html
    assert 'class="pm-mlp-flow pm-mlp-flow-b"' in html


def test_tcn_icon_uses_compact_unlabelled_kernel():
    html = model_icon_html("tcn")

    assert "causal k = 3" not in html
    assert 'class="pm-kernel-box" x="24" y="39" width="116" height="32"' in html


def test_model_family_icon_rejects_unknown_type():
    with pytest.raises(ValueError, match="不支持的模型图标"):
        model_icon_html("cnn")


def test_default_catalog_and_artifacts_are_consistent():
    catalog = load_catalog()

    assert [record.id for record in catalog.enabled_models] == [
        "baseline-bigram",
        "low-mlp",
        "low-tcn",
        "low-gru",
        "low-transformer",
        "medium-mlp",
        "medium-tcn",
        "medium-gru",
        "medium-transformer",
        "high-mlp",
        "high-tcn",
        "high-gru",
        "high-transformer",
    ]
    assert len({record.id for record in catalog.models}) == len(catalog.models)
    assert all(record.inference_ready for record in catalog.enabled_models)
    assert catalog.by_id("low-gru").flops == 4_882_553
    assert catalog.by_id("low-gru").training_sample_count == 7_745_971
    assert catalog.by_id("medium-gru").training_sample_count == 7_745_971
    assert catalog.by_id("high-gru").training_sample_count == 7_745_971
    assert catalog.by_id("baseline-bigram").flops == 2_574
    assert [record.id for record in catalog.models if record.model_type == "mlp"] == [
        "low-mlp",
        "medium-mlp",
        "high-mlp",
    ]
    assert all(record.enabled for record in catalog.models if record.model_type == "mlp")


def test_model_selection_presets_cover_tiers_families_and_optional_baseline():
    catalog = load_catalog()

    assert preset_model_ids(catalog, "Low tier") == [
        "low-mlp",
        "low-tcn",
        "low-gru",
        "low-transformer",
    ]
    assert preset_model_ids(catalog, "TCN family") == [
        "low-tcn",
        "medium-tcn",
        "high-tcn",
    ]
    assert preset_model_ids(catalog, "MLP family") == [
        "low-mlp",
        "medium-mlp",
        "high-mlp",
    ]
    assert preset_model_ids(catalog, "GRU family", include_baseline=True) == [
        "baseline-bigram",
        "low-gru",
        "medium-gru",
        "high-gru",
    ]
    assert len(preset_model_ids(catalog, "All neural models")) == 12
    with pytest.raises(ValueError, match="未知模型预设"):
        preset_model_ids(catalog, "Unknown")


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
                        "flops": 1,
                        "training_sample_count": 1,
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
                "epoch_seconds": [60.0, 90.0],
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
    assert history.epoch_seconds.tolist() == pytest.approx([60.0, 90.0])
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


@pytest.mark.parametrize("loader", [load_surprisal_data, load_coverage_data])
def test_npz_loaders_report_corrupt_files_as_validation_errors(tmp_path, loader):
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(ValueError, match="尚未写完或已经损坏"):
        loader(path)


def test_evaluation_summary_and_research_overview_use_full_statistics(tmp_path):
    metric = {
        "count": 100,
        "mean": 2.0,
        "median": 1.8,
        "p90": 3.0,
        "p95": 3.5,
        "p99": 4.0,
        "minimum": 0.5,
        "maximum": 6.0,
    }
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "surprisal": {
                    "count": 100,
                    "raw_surprisal_bits": {**metric, "mean": 10.0},
                    "bits_per_token": metric,
                },
                "generation_quality": {
                    "total_samples": 1000,
                    "legal_rate": 0.99,
                    "legal_unique_rate": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    summary = load_evaluation_summary(path)
    history = TrainingHistory(
        np.asarray([2.2, 1.8]),
        np.asarray([2.3, 1.9]),
        np.asarray([0.1, 0.05]),
        2.0,
        np.asarray([1800.0, 1800.0]),
    )
    coverage = CoverageData(100, np.asarray([100, 500]), np.asarray([0.1, 0.4]))
    overview = build_research_overview(
        "Model", 123, summary, history, coverage, coverage,
        best_first_budget=400, random_budget=500,
        flops=456, training_sample_count=789,
    )

    assert summary.bits_per_token.mean == pytest.approx(2.0)
    assert overview.perplexity == pytest.approx(4.0)
    assert overview.flops == 456
    assert overview.training_sample_count == 789
    assert overview.best_validation_loss == pytest.approx(1.9)
    assert overview.best_epoch == 2
    assert overview.training_hours == pytest.approx(1.0)
    assert overview.best_first_coverage == pytest.approx(0.1)
    assert overview.random_coverage == pytest.approx(0.4)
    assert coverage_at_budget(coverage, 99) is None


def test_evaluation_summary_allows_pending_generation_quality(tmp_path):
    metric = {
        "count": 2,
        "mean": 2.0,
        "median": 2.0,
        "p90": 2.8,
        "p95": 2.9,
        "p99": 2.98,
        "minimum": 1.0,
        "maximum": 3.0,
    }
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "surprisal": {
                    "count": 2,
                    "raw_surprisal_bits": metric,
                    "bits_per_token": metric,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = load_evaluation_summary(path)

    assert summary.evaluation_size == 2
    assert summary.generation_quality is None


def test_pairwise_surprisal_comparison_is_paired_and_counts_ties():
    left = SurprisalData(3, np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0]))
    right = SurprisalData(3, np.asarray([2.0, 2.0, 4.0]), np.asarray([2.0, 2.0, 4.0]))

    matrix = pairwise_win_rates({"Left": left, "Right": right})
    differences = pairwise_difference(left, right)

    assert matrix.loc["Left", "Right"] == pytest.approx(5 / 6)
    assert matrix.loc["Right", "Left"] == pytest.approx(1 / 6)
    assert matrix.loc["Left", "Left"] == pytest.approx(0.5)
    assert differences.tolist() == pytest.approx([-1.0, 0.0, -1.0])


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


def test_new_library_comparison_charts_have_expected_semantics():
    history = TrainingHistory(
        np.asarray([2.0, 1.8]),
        np.asarray([2.1, 1.9]),
        None,
        None,
        np.asarray([1800.0, 3600.0]),
    )
    summary_metric = MetricSummary(10, 2.0, 1.9, 2.8, 3.0, 3.5, 1.0, 4.0)
    summary = EvaluationSummary(
        10,
        MetricSummary(10, 10.0, 9.0, 14.0, 15.0, 18.0, 4.0, 22.0),
        summary_metric,
        load_generation_quality(
            load_catalog().by_id("low-gru").evaluation_dir / "summary.json"
        ),
    )
    overview = build_research_overview(
        "GRU", 100, summary, history, flops=1_000, training_sample_count=10
    )
    bigram_overview = build_research_overview(
        "Bigram", 0, summary, flops=10, training_sample_count=10
    )
    pairwise = pairwise_win_rates(
        {
            "A": SurprisalData(2, np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])),
            "B": SurprisalData(2, np.asarray([2.0, 1.0]), np.asarray([2.0, 1.0])),
        }
    )

    time_spec = plot_validation_loss_by_time(
        {"GRU": history}, {"GRU": "#000000"}
    ).to_dict()
    gap_spec = plot_generalization_gap(
        {"GRU": history}, {"GRU": "#000000"}
    ).to_dict()
    zoo_spec = plot_model_zoo(
        [bigram_overview, overview], {"Bigram": "#ffffff", "GRU": "#000000"}
    ).to_dict()
    win_spec = plot_pairwise_win_rates(pairwise).to_dict()
    difference_spec = plot_pairwise_difference(
        "A", "B", np.asarray([-1.0, 1.0])
    ).to_dict()

    assert time_spec["title"] == "Validation Loss by Cumulative Training Time"
    assert time_spec["encoding"]["x"]["scale"]["domainMin"] == 0
    assert time_spec["encoding"]["y"]["scale"] == {"zero": False}
    assert gap_spec["title"] == "Generalization Gap by Epoch"
    assert gap_spec["layer"][0]["encoding"]["y"]["field"] == "Generalization Gap"
    assert gap_spec["layer"][1]["mark"]["type"] == "rule"
    assert zoo_spec["title"] == "Model Zoo"
    assert zoo_spec["layer"][0]["encoding"]["x"]["field"] == "Estimated FLOPs"
    assert zoo_spec["layer"][0]["encoding"]["size"]["scale"]["type"] == "symlog"
    assert zoo_spec["layer"][0]["encoding"]["size"]["scale"]["range"] == [120, 3200]
    assert zoo_spec["layer"][0]["encoding"]["size"]["legend"] is None
    assert zoo_spec["layer"][0]["mark"]["opacity"] == 1.0
    assert "stroke" not in zoo_spec["layer"][0]["mark"]
    assert "strokeWidth" not in zoo_spec["layer"][0]["mark"]
    assert zoo_spec["layer"][1]["mark"]["dy"] == {
        "expr": "-datum['Label Offset']"
    }
    zoo_rows = next(iter(zoo_spec["datasets"].values()))
    offsets = {row["Model"]: row["Label Offset"] for row in zoo_rows}
    assert offsets["Bigram"] < offsets["GRU"]
    assert win_spec["title"] == "Pairwise Lower-Surprisal Win Rate"
    assert win_spec["layer"][0]["encoding"]["x"]["title"] == "Opponent Model"
    assert win_spec["layer"][0]["encoding"]["y"]["title"] == "Focal Model"
    assert win_spec["layer"][0]["encoding"]["y"]["axis"]["labelOverlap"] is False
    assert difference_spec["title"] == "Paired Surprisal Difference"


def test_pairwise_chart_keeps_every_model_label_visible():
    labels = [f"Model {index}" for index in range(13)]
    matrix = np.full((13, 13), 0.5)
    chart = plot_pairwise_win_rates(
        pd.DataFrame(matrix, index=labels, columns=labels)
    ).to_dict()

    x_encoding = chart["layer"][0]["encoding"]["x"]
    y_encoding = chart["layer"][0]["encoding"]["y"]
    assert x_encoding["sort"] == labels
    assert y_encoding["sort"] == labels
    assert x_encoding["axis"]["labelOverlap"] is False
    assert y_encoding["axis"]["labelOverlap"] is False
    assert chart["height"] == 38 * len(labels)


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


def test_training_history_only_lists_test_split_when_data_exists():
    without_test = TrainingHistory(
        train_loss=np.asarray([2.0, 1.8]),
        valid_loss=np.asarray([2.1, 1.9]),
        learning_rate=None,
        test_loss=None,
    )
    with_test = TrainingHistory(
        train_loss=np.asarray([2.0, 1.8]),
        valid_loss=np.asarray([2.1, 1.9]),
        learning_rate=None,
        test_loss=1.85,
    )
    colors = {"GRU": "#000000"}

    without_spec = plot_training_history({"GRU": without_test}, colors).to_dict()
    with_spec = plot_training_history({"GRU": with_test}, colors).to_dict()

    assert without_spec["encoding"]["strokeDash"]["scale"]["domain"] == [
        "Train",
        "Validation",
    ]
    assert with_spec["encoding"]["strokeDash"]["scale"]["domain"] == [
        "Train",
        "Validation",
        "Test",
    ]


def test_surprisal_boxplot_uses_tukey_whiskers_without_scaling_to_extremes():
    values = np.asarray([1.0, 2.0, 2.0, 3.0, 100.0])
    chart = plot_surprisal_boxplot(
        {"GRU": SurprisalData(5, values, values)},
        {"GRU": "#000000"},
    )

    spec = chart.to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert rows[0]["Lower Whisker"] == pytest.approx(1.0)
    assert rows[0]["Upper Whisker"] == pytest.approx(3.0)
    assert rows[0]["Maximum"] == pytest.approx(100.0)
    assert rows[0]["Outliers"] == 1
    assert spec["layer"][0]["encoding"]["y"]["field"] == "Lower Whisker"
    assert spec["layer"][0]["encoding"]["y2"]["field"] == "Upper Whisker"


def test_playground_services_with_bigram_artifact():
    record = load_catalog().by_id("baseline-bigram")
    runtime = {record.id: get_runtime_model(record)}

    scores = score_models("passmini", runtime)
    aggregate = aggregate_scores(scores)
    generated = generate_with_models(runtime, 3, 8, 1.0, 2026)
    challenge = generate_challenge_password(
        record.id, runtime[record.id], 8, 1.0, 2026
    )
    completed = complete_with_models(runtime, "pass", 3, 8, 1.0)

    assert len(scores) == 1
    assert np.isfinite(aggregate.mean_surprisal_bits)
    assert len(generated[record.id]) == 3
    assert 1 <= len(challenge) <= 8
    assert 1 <= len(completed[record.id]) <= 3
    assert all(candidate.text.startswith("pass") for candidate in completed[record.id])


def test_score_chart_uses_stable_hundred_bit_domain_and_expands_by_steps():
    labels = {"a": "A", "b": "B"}
    colors = {"a": "#000000", "b": "#ffffff"}
    compact = plot_score_comparison(
        [PasswordScore("a", 12.0, 6.0)],
        labels,
        colors,
    ).to_dict()
    expanded = plot_score_comparison(
        [PasswordScore("a", 101.0, 50.5)],
        labels,
        colors,
    ).to_dict()

    assert compact["encoding"]["x"]["scale"]["domain"] == [0, 100.0]
    assert expanded["encoding"]["x"]["scale"]["domain"] == [0, 125.0]


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
    assert sum(item.probability for item in trace.distribution) == pytest.approx(1.0)
    assert trace.characters[0].surprisal_bits == pytest.approx(-np.log2(10 / 13))


def test_character_lab_visualizations_and_temperature_entropy():
    tokenizer = CharTokenizer.from_text(["abc"])
    model = AutoregressiveBigram(tokenizer, alpha=1.0)
    model.count[tokenizer.bos_id, tokenizer.token_to_id["a"]] = 30
    model.count[tokenizer.token_to_id["a"], tokenizer.token_to_id["b"]] = 30
    runtime = {"bigram": (model, tokenizer)}
    cold = trace_character_probabilities("a", runtime, temperature=0.5)[0]
    neutral = trace_character_probabilities("a", runtime, temperature=1.0)[0]
    hot = trace_character_probabilities("a", runtime, temperature=2.0)[0]

    journey = plot_surprisal_journey(
        [neutral], {"bigram": "Bigram"}, {"bigram": "#000000"}
    ).to_dict()
    keyboard = plot_probability_keyboard(neutral, "Bigram").to_dict()
    laboratory = plot_temperature_laboratory(
        {0.5: cold, 1.0: neutral, 2.0: hot}
    ).to_dict()

    assert distribution_entropy(cold) < distribution_entropy(hot)
    assert journey["vconcat"][0]["title"] == "Character Surprisal Journey"
    assert journey["vconcat"][2]["title"] == "Surprisal per Token"
    assert keyboard["title"] == "Probability Keyboard · Bigram"
    assert laboratory["vconcat"][0]["title"] == "Temperature Response of Top Tokens"


def test_model_disagreement_and_completion_consensus_visualizations():
    tokenizer = CharTokenizer.from_text(["abc"])
    first = AutoregressiveBigram(tokenizer, alpha=1.0)
    second = AutoregressiveBigram(tokenizer, alpha=1.0)
    first.count[tokenizer.bos_id, tokenizer.token_to_id["a"]] = 20
    second.count[tokenizer.bos_id, tokenizer.token_to_id["b"]] = 20
    traces = trace_character_probabilities(
        "", {"first": (first, tokenizer), "second": (second, tokenizer)}
    )
    labels = {"first": "First", "second": "Second"}

    matrix = model_disagreement_matrix(traces)
    disagreement = plot_model_disagreement(traces, labels).to_dict()
    completions = {
        "first": [GenerationCandidate("abc", [1], -1.0)],
        "second": [
            GenerationCandidate("abd", [2], -0.8),
            GenerationCandidate("abc", [1], -1.2),
        ],
    }
    consensus_frame = completion_consensus_frame(completions, labels)
    consensus_summary = completion_consensus_summary(completions, labels)
    consensus = plot_completion_consensus(completions, labels).to_dict()

    assert matrix.loc["first", "first"] == pytest.approx(0.0)
    assert matrix.loc["first", "second"] == pytest.approx(
        matrix.loc["second", "first"]
    )
    assert matrix.loc["first", "second"] > 0
    assert disagreement["vconcat"][1]["title"] == "Pairwise Jensen–Shannon Divergence"
    assert len(consensus_frame) == 3
    assert consensus_summary.iloc[0]["Candidate"] == "abc"
    assert consensus_summary.iloc[0]["Support Rate"] == pytest.approx(1.0)
    assert consensus["title"] == "Completion Consensus"
    assert consensus["encoding"]["x"]["field"] == "Consensus Score"


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
