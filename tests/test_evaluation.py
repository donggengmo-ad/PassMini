import numpy as np
import pytest

from scripts.evaluation import (
    CoveragePoint,
    GenerationQualitySummary,
    coverage_curve,
    evaluate_random_generation,
    evaluate_random_generation_with_coverage,
    generate_random_passwords,
    plot_coverage_curve,
    plot_efficiency_curve,
    plot_generation_quality,
    plot_random_coverage_curve,
    plot_random_efficiency_curve,
    random_generation_coverage_curve,
    plot_surprisal_boxplot,
    plot_surprisal_histogram,
    read_test_passwords,
    save_coverage_npz,
    save_surprisal_npz,
    summarize_surprisal,
    summarize_values,
)
from scripts.inference import GenerationCandidate


def test_save_surprisal_npz_keeps_all_values_below_limit(tmp_path):
    path = tmp_path / "surprisal.npz"
    returned = save_surprisal_npz(
        path,
        surprisal_bits=[2.0, 4.0, 8.0],
        bits_per_token=[1.0, 2.0, 4.0],
        max_points=4,
    )

    assert returned == path
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "evaluation_size",
            "surprisal_bits",
            "bits_per_token",
        }
        assert data["evaluation_size"].item() == 3
        np.testing.assert_array_equal(data["surprisal_bits"], [2.0, 4.0, 8.0])
        np.testing.assert_array_equal(data["bits_per_token"], [1.0, 2.0, 4.0])


def test_save_surprisal_npz_uses_aligned_uniform_sampling(tmp_path):
    path = tmp_path / "nested" / "surprisal.npz"
    raw = np.arange(10, dtype=np.float32)
    normalized = raw + 100

    save_surprisal_npz(path, raw, normalized, max_points=4)

    with np.load(path, allow_pickle=False) as data:
        assert data["evaluation_size"].item() == 10
        np.testing.assert_array_equal(data["surprisal_bits"], [0.0, 3.0, 6.0, 9.0])
        np.testing.assert_array_equal(
            data["bits_per_token"],
            [100.0, 103.0, 106.0, 109.0],
        )


@pytest.mark.parametrize(
    ("raw", "normalized", "max_points"),
    [
        ([], [], 10),
        ([1.0], [], 10),
        ([float("nan")], [1.0], 10),
        ([1.0], [1.0], 0),
    ],
)
def test_save_surprisal_npz_rejects_invalid_values(
    tmp_path,
    raw,
    normalized,
    max_points,
):
    with pytest.raises(ValueError):
        save_surprisal_npz(
            tmp_path / "surprisal.npz",
            raw,
            normalized,
            max_points=max_points,
        )


def test_save_coverage_npz_uses_minimal_fixed_fields(tmp_path):
    path = tmp_path / "coverage.npz"
    points = [
        CoveragePoint(attempts=100, hits=2, coverage=0.02, efficiency=0.0002),
        CoveragePoint(attempts=200, hits=5, coverage=0.05, efficiency=0.00025),
    ]

    save_coverage_npz(path, points, test_size=100)

    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"test_size", "attempts", "coverage"}
        assert data["test_size"].item() == 100
        np.testing.assert_array_equal(data["attempts"], [100, 200])
        np.testing.assert_allclose(data["coverage"], [0.02, 0.05])


@pytest.mark.parametrize(
    ("points", "test_size"),
    [
        ([], 100),
        ([CoveragePoint(100, 1, 0.01, 0.0001)], 0),
        (
            [
                CoveragePoint(200, 1, 0.01, 0.00005),
                CoveragePoint(100, 2, 0.02, 0.0002),
            ],
            100,
        ),
        ([CoveragePoint(100, 1, float("nan"), 0.0)], 100),
    ],
)
def test_save_coverage_npz_rejects_invalid_values(tmp_path, points, test_size):
    with pytest.raises(ValueError):
        save_coverage_npz(tmp_path / "coverage.npz", points, test_size)


class SequenceModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.batch_calls = []

    def generate(self, max_length, temperature, generator=None):
        self.calls += 1
        return self.outputs[(self.calls - 1) % len(self.outputs)]

    def generate_batch(self, batch_size, max_length, temperature, generator=None):
        self.batch_calls.append(batch_size)
        return [
            self.generate(max_length, temperature, generator)
            for _ in range(batch_size)
        ]


def make_candidate(text: str, score: float = 0.0) -> GenerationCandidate:
    return GenerationCandidate(text=text, token_ids=[], log_probability=score)


def test_read_test_passwords_samples_deterministically(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    first = read_test_passwords(path, sample_size=2, seed=42)
    second = read_test_passwords(path, sample_size=2, seed=42)

    assert first == second
    assert len(first) == 2
    assert len(set(first)) == 2


def test_read_test_passwords_rejects_duplicate_lines(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("a\na\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复"):
        read_test_passwords(path)


def test_coverage_curve_tracks_cumulative_hits_and_efficiency():
    candidates = [
        make_candidate("x"),
        make_candidate("a"),
        make_candidate("b"),
        make_candidate("y"),
    ]

    points = coverage_curve(candidates, ["a", "b", "c", "d"], checkpoint_step=2)

    assert [point.attempts for point in points] == [2, 4]
    assert [point.hits for point in points] == [1, 2]
    assert [point.coverage for point in points] == [pytest.approx(0.25), pytest.approx(0.5)]
    assert [point.efficiency for point in points] == [pytest.approx(0.125), pytest.approx(0.125)]


def test_coverage_curve_rejects_duplicate_candidates():
    candidates = [make_candidate("a"), make_candidate("a")]

    with pytest.raises(ValueError, match="重复"):
        coverage_curve(candidates, ["a"], checkpoint_step=1)


def test_plot_coverage_curve_uses_attempts_and_coverage():
    import matplotlib.pyplot as plt

    results = {
        "Bigram": [
            make_candidate("a"),
            make_candidate("b"),
        ]
    }
    points = coverage_curve(results["Bigram"], ["a", "b"], checkpoint_step=1)
    figure, axis = plt.subplots()

    returned_axis = plot_coverage_curve({"Bigram": points}, ax=axis)

    assert returned_axis is axis
    assert list(axis.lines[0].get_xdata()) == [1, 2]
    assert list(axis.lines[0].get_ydata()) == [0.5, 1.0]
    assert axis.get_title() == "Cumulative Coverage vs. Search Attempts"
    assert axis.get_xlabel() == "Search Attempts"
    assert axis.get_ylabel() == "Cumulative Coverage"
    plt.close(figure)


def test_plot_efficiency_curve_uses_efficiency_values():
    import matplotlib.pyplot as plt

    points = [
        CoveragePoint(attempts=1, hits=1, coverage=0.5, efficiency=0.5),
        CoveragePoint(attempts=2, hits=1, coverage=0.5, efficiency=0.25),
    ]
    figure, axis = plt.subplots()

    returned_axis = plot_efficiency_curve({"GRU": points}, ax=axis)

    assert returned_axis is axis
    assert list(axis.lines[0].get_xdata()) == [1, 2]
    assert list(axis.lines[0].get_ydata()) == [0.5, 0.25]
    assert axis.get_title() == "Search Efficiency vs. Search Attempts"
    assert axis.get_xlabel() == "Search Attempts"
    assert axis.get_ylabel() == "Cumulative Coverage / Search Attempts"
    plt.close(figure)


def test_evaluate_random_generation_reports_legal_and_legal_unique_rates():
    model = SequenceModel(["abc", "abc", "", "a\n", "A"])

    summary = evaluate_random_generation(model, num_samples=5, max_length=12)

    assert summary == GenerationQualitySummary(
        total_samples=5,
        legal_samples=3,
        distinct_legal_samples=2,
        legal_rate=pytest.approx(0.6),
        legal_unique_rate=pytest.approx(2 / 3),
    )
    assert model.calls == 5


def test_generate_random_passwords_returns_requested_samples():
    model = SequenceModel(["a", "b"])

    samples = generate_random_passwords(model, num_samples=3, max_length=12)

    assert samples == ["a", "b", "a"]
    assert model.calls == 3


def test_generate_random_passwords_uses_batches_and_handles_tail_batch():
    model = SequenceModel(["a", "b"])

    samples = generate_random_passwords(
        model,
        num_samples=5,
        batch_size=2,
        max_length=12,
    )

    assert samples == ["a", "b", "a", "b", "a"]
    assert model.calls == 5
    assert model.batch_calls == [2, 2, 1]


def test_generate_random_passwords_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        generate_random_passwords(SequenceModel(["a"]), num_samples=2, batch_size=0)


def test_random_generation_coverage_curve_allows_repeated_samples():
    samples = ["a", "a", "x", "b"]

    points = random_generation_coverage_curve(
        samples,
        ["a", "b"],
        checkpoint_step=2,
    )

    assert [point.attempts for point in points] == [2, 4]
    assert [point.hits for point in points] == [1, 2]
    assert [point.coverage for point in points] == [0.5, 1.0]
    assert [point.efficiency for point in points] == [0.25, 0.25]


def test_random_generation_coverage_curve_rejects_duplicate_test_passwords():
    with pytest.raises(ValueError, match="重复"):
        random_generation_coverage_curve(["a"], ["a", "a"])


def test_evaluate_random_generation_with_coverage_reuses_one_sample_batch():
    model = SequenceModel(["a", "a", "b", "x"])

    result = evaluate_random_generation_with_coverage(
        model,
        ["a", "b"],
        num_samples=4,
        max_length=12,
        checkpoint_step=2,
    )

    assert result.quality.total_samples == 4
    assert result.quality.distinct_legal_samples == 3
    assert [point.attempts for point in result.coverage] == [2, 4]
    assert [point.hits for point in result.coverage] == [1, 2]
    assert model.calls == 4


def test_plot_random_coverage_curve_labels_attempts_as_sampling():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    points = random_generation_coverage_curve(["a", "a"], ["a"], checkpoint_step=1)

    returned_axis = plot_random_coverage_curve({"GRU": points}, ax=axis)

    assert returned_axis is axis
    assert axis.get_title() == "Cumulative Coverage vs. Sampling Attempts"
    assert axis.get_xlabel() == "Sampling Attempts"
    assert axis.get_ylabel() == "Cumulative Coverage"
    plt.close(figure)


def test_plot_random_efficiency_curve_labels_attempts_as_sampling():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    points = random_generation_coverage_curve(["a", "a"], ["a"], checkpoint_step=1)

    returned_axis = plot_random_efficiency_curve({"GRU": points}, ax=axis)

    assert returned_axis is axis
    assert axis.get_title() == "Sampling Efficiency vs. Sampling Attempts"
    assert axis.get_xlabel() == "Sampling Attempts"
    assert axis.get_ylabel() == "Cumulative Coverage / Sampling Attempts"
    plt.close(figure)


def test_evaluate_random_generation_rejects_invalid_sample_count():
    with pytest.raises(ValueError, match="num_samples"):
        evaluate_random_generation(SequenceModel(["a"]), num_samples=0)


def test_plot_generation_quality_draws_two_rate_bars_per_model():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    results = {
        "Bigram": GenerationQualitySummary(10, 8, 6, 0.8, 0.75),
        "GRU": GenerationQualitySummary(10, 9, 8, 0.9, 8 / 9),
    }

    returned_axis = plot_generation_quality(results, ax=axis)

    assert returned_axis is axis
    assert len(axis.containers) == 2
    assert [label.get_text() for label in axis.get_xticklabels()] == ["Bigram", "GRU"]
    assert axis.get_ylabel() == "Rate"
    assert {text.get_text() for text in axis.get_legend().get_texts()} == {
        "Legal Rate",
        "Legal Unique Rate",
    }
    plt.close(figure)


def test_summarize_values_calculates_distribution_statistics():
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])

    assert summary.count == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.median == pytest.approx(2.5)
    assert summary.p90 == pytest.approx(3.7)
    assert summary.p95 == pytest.approx(3.85)
    assert summary.p99 == pytest.approx(3.97)
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(4.0)


def test_summarize_values_rejects_empty_and_non_finite_values():
    with pytest.raises(ValueError, match="不能为空"):
        summarize_values([])
    with pytest.raises(ValueError, match="有限"):
        summarize_values([1.0, float("nan")])


def test_summarize_surprisal_normalizes_by_characters_plus_eos():
    summary = summarize_surprisal(["a", "bc", ""], [2.0, 6.0, 1.0])

    assert summary.count == 3
    assert summary.raw_surprisal_bits.mean == pytest.approx(3.0)
    assert summary.bits_per_token.mean == pytest.approx((1.0 + 2.0 + 1.0) / 3)


def test_summarize_surprisal_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="数量"):
        summarize_surprisal(["a"], [1.0, 2.0])


def test_plot_surprisal_histogram_draws_one_line_per_model():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    returned_axis = plot_surprisal_histogram(
        {"Bigram": [1.0, 2.0, 2.0], "GRU": [0.5, 1.5, 2.5]},
        ax=axis,
        bins=3,
    )

    assert returned_axis is axis
    assert len(axis.patches) > 0
    assert axis.get_xlabel() == "Surprisal (bits)"
    assert {line.get_text() for line in axis.get_legend().get_texts()} == {
        "Bigram",
        "GRU",
    }
    plt.close(figure)


def test_plot_surprisal_boxplot_labels_models():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    returned_axis = plot_surprisal_boxplot(
        {"Bigram": [1.0, 2.0, 3.0], "GRU": [0.5, 1.5, 2.5]},
        ax=axis,
    )

    assert returned_axis is axis
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "Bigram",
        "GRU",
    ]
    assert axis.get_ylabel() == "Surprisal (bits)"
    plt.close(figure)


def test_surprisal_plots_allow_bits_per_token_label():
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2)
    plot_surprisal_histogram(
        {"GRU": [0.5, 1.0]},
        ax=axes[0],
        value_label="Surprisal (bits/token)",
    )
    plot_surprisal_boxplot(
        {"GRU": [0.5, 1.0]},
        ax=axes[1],
        value_label="Surprisal (bits/token)",
    )

    assert axes[0].get_xlabel() == "Surprisal (bits/token)"
    assert axes[1].get_ylabel() == "Surprisal (bits/token)"
    plt.close(figure)
