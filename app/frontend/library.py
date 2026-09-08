"""Library 的静态评测数据读取、聚合与交互绘图。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import log1p, pi, sqrt
from pathlib import Path
from typing import Mapping
from zipfile import BadZipFile, is_zipfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: np.ndarray
    valid_loss: np.ndarray
    learning_rate: np.ndarray | None
    test_loss: float | None
    epoch_seconds: np.ndarray | None = None


@dataclass(frozen=True)
class SurprisalData:
    evaluation_size: int
    surprisal_bits: np.ndarray
    bits_per_token: np.ndarray


@dataclass(frozen=True)
class CoverageData:
    test_size: int
    attempts: np.ndarray
    coverage: np.ndarray

    @property
    def efficiency(self) -> np.ndarray:
        return self.coverage / self.attempts


@dataclass(frozen=True)
class GenerationQuality:
    total_samples: int
    legal_rate: float
    legal_unique_rate: float


@dataclass(frozen=True)
class MetricSummary:
    """保存完整评测集上的一组分布统计量。"""

    count: int
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class EvaluationSummary:
    """保存 summary.json 中不会因前端抽样而改变的完整评测摘要。"""

    evaluation_size: int
    raw_surprisal_bits: MetricSummary
    bits_per_token: MetricSummary
    generation_quality: GenerationQuality | None


@dataclass(frozen=True)
class ResearchOverview:
    """汇总一个模型在模型规模、训练和评测三方面的核心结果。"""

    model: str
    parameter_count: int
    flops: int
    training_sample_count: int | None
    evaluation_size: int
    mean_surprisal_bits: float
    mean_bits_per_token: float
    median_bits_per_token: float
    p95_bits_per_token: float
    perplexity: float
    best_validation_loss: float | None
    best_epoch: int | None
    generalization_gap: float | None
    training_hours: float | None
    best_first_coverage: float | None
    random_coverage: float | None


def _finite_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须全部是有限值")
    return array


def _load_npz_fields(path: str | Path, required: set[str]) -> dict[str, np.ndarray]:
    """读取并复制指定 NPZ 字段，将损坏压缩包统一转换为数据校验错误。"""

    source = Path(path)
    if not is_zipfile(source):
        raise ValueError(f"无法读取 {source.name}，文件可能尚未写完或已经损坏")
    try:
        with np.load(source, allow_pickle=False) as payload:
            if not required.issubset(payload.files):
                raise ValueError(f"{source.name} 缺少必需字段")
            return {name: np.asarray(payload[name]).copy() for name in required}
    except (BadZipFile, EOFError, OSError) as error:
        raise ValueError(f"无法读取 {source.name}，文件可能尚未写完或已经损坏") from error


@st.cache_data(show_spinner=False, max_entries=32)
def load_training_history(path: str | Path) -> TrainingHistory:
    """读取神经模型的训练、验证和学习率历史。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("history.json 顶层必须是对象")
    train_loss = _finite_vector(payload.get("train_loss"), "train_loss")
    valid_loss = _finite_vector(payload.get("valid_loss"), "valid_loss")
    if train_loss.size != valid_loss.size:
        raise ValueError("train_loss 与 valid_loss 的 epoch 数必须一致")

    raw_learning_rate = payload.get("learning_rate")
    learning_rate = (
        None
        if raw_learning_rate is None
        else _finite_vector(raw_learning_rate, "learning_rate")
    )
    if learning_rate is not None and learning_rate.size != train_loss.size:
        raise ValueError("learning_rate 与 loss 的 epoch 数必须一致")
    raw_epoch_seconds = payload.get("epoch_seconds")
    epoch_seconds = (
        None
        if raw_epoch_seconds is None
        else _finite_vector(raw_epoch_seconds, "epoch_seconds")
    )
    if epoch_seconds is not None:
        if epoch_seconds.size != train_loss.size:
            raise ValueError("epoch_seconds 与 loss 的 epoch 数必须一致")
        if np.any(epoch_seconds <= 0):
            raise ValueError("epoch_seconds 必须全部大于 0")
    raw_test_loss = payload.get("test_loss")
    test_loss = None if raw_test_loss is None else float(raw_test_loss)
    if test_loss is not None and not np.isfinite(test_loss):
        raise ValueError("test_loss 必须是有限值")
    return TrainingHistory(train_loss, valid_loss, learning_rate, test_loss, epoch_seconds)


@st.cache_data(show_spinner=False, max_entries=32)
def load_surprisal_data(path: str | Path) -> SurprisalData:
    """读取单模型 surprisal NPZ，并复制数组以关闭文件句柄。"""

    values = _load_npz_fields(
        path,
        {"evaluation_size", "surprisal_bits", "bits_per_token"},
    )
    evaluation_size = int(values["evaluation_size"].item())
    surprisal = _finite_vector(values["surprisal_bits"], "surprisal_bits")
    per_token = _finite_vector(values["bits_per_token"], "bits_per_token")
    if evaluation_size <= 0 or surprisal.size != per_token.size:
        raise ValueError("surprisal NPZ 的数量信息不一致")
    return SurprisalData(evaluation_size, surprisal, per_token)


@st.cache_data(show_spinner=False, max_entries=32)
def load_coverage_data(path: str | Path) -> CoverageData:
    """读取单模型覆盖率 NPZ。"""

    values = _load_npz_fields(path, {"test_size", "attempts", "coverage"})
    test_size = int(values["test_size"].item())
    attempts = np.asarray(values["attempts"], dtype=np.int64)
    coverage = _finite_vector(values["coverage"], "coverage")
    if test_size <= 0 or attempts.ndim != 1 or attempts.size != coverage.size:
        raise ValueError("coverage NPZ 的数量信息不一致")
    if attempts.size == 0 or np.any(attempts <= 0) or np.any(attempts[1:] <= attempts[:-1]):
        raise ValueError("attempts 必须是严格递增的正整数")
    if np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("coverage 必须位于 [0, 1]")
    return CoverageData(test_size, attempts, coverage)


@st.cache_data(show_spinner=False, max_entries=32)
def load_generation_quality(path: str | Path) -> GenerationQuality:
    """读取模型独立 summary.json 中的随机生成质量摘要。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json 顶层必须是对象")
    quality = payload.get("generation_quality", payload)
    if not isinstance(quality, dict):
        raise ValueError("generation_quality 必须是对象")
    total_samples = int(quality["total_samples"])
    legal_rate = float(quality["legal_rate"])
    legal_unique_rate = float(quality["legal_unique_rate"])
    if total_samples <= 0:
        raise ValueError("total_samples 必须大于 0")
    if not 0 <= legal_rate <= 1 or not 0 <= legal_unique_rate <= 1:
        raise ValueError("生成质量比例必须位于 [0, 1]")
    return GenerationQuality(total_samples, legal_rate, legal_unique_rate)


def _metric_summary(payload: object, name: str) -> MetricSummary:
    """校验并构造完整评测分布的摘要统计。"""

    if not isinstance(payload, dict):
        raise ValueError(f"{name} 必须是对象")
    summary = MetricSummary(
        count=int(payload["count"]),
        mean=float(payload["mean"]),
        median=float(payload["median"]),
        p90=float(payload["p90"]),
        p95=float(payload["p95"]),
        p99=float(payload["p99"]),
        minimum=float(payload["minimum"]),
        maximum=float(payload["maximum"]),
    )
    values = (
        summary.mean,
        summary.median,
        summary.p90,
        summary.p95,
        summary.p99,
        summary.minimum,
        summary.maximum,
    )
    if summary.count <= 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} 的数量和统计量无效")
    return summary


@st.cache_data(show_spinner=False, max_entries=32)
def load_evaluation_summary(path: str | Path) -> EvaluationSummary:
    """读取完整 surprisal 摘要，并按需附带随机生成质量结果。

    Surprisal 评分和随机生成评测可能分阶段完成，因此缺少
    `generation_quality` 不影响已经完成的全量 surprisal 统计。
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("surprisal"), dict):
        raise ValueError("summary.json 缺少 surprisal 对象")
    surprisal = payload["surprisal"]
    raw = _metric_summary(surprisal.get("raw_surprisal_bits"), "raw_surprisal_bits")
    per_token = _metric_summary(surprisal.get("bits_per_token"), "bits_per_token")
    evaluation_size = int(surprisal["count"])
    if evaluation_size <= 0 or raw.count != evaluation_size or per_token.count != evaluation_size:
        raise ValueError("surprisal 摘要的评测数量不一致")
    quality = (
        load_generation_quality(path)
        if "generation_quality" in payload
        else None
    )
    return EvaluationSummary(evaluation_size, raw, per_token, quality)


def coverage_at_budget(data: CoverageData, budget: int) -> float | None:
    """返回不超过给定预算的最后一个实测覆盖率检查点，不进行插值。"""

    if budget <= 0:
        raise ValueError("budget 必须大于 0")
    index = int(np.searchsorted(data.attempts, budget, side="right") - 1)
    return None if index < 0 else float(data.coverage[index])


def build_research_overview(
    model: str,
    parameter_count: int,
    summary: EvaluationSummary,
    history: TrainingHistory | None = None,
    best_first: CoverageData | None = None,
    random_sampling: CoverageData | None = None,
    best_first_budget: int = 500_000,
    random_budget: int = 100_000_000,
    flops: int = 0,
    training_sample_count: int | None = None,
) -> ResearchOverview:
    """从已有 artifact 组装研究总览，不重新执行训练或评测。"""

    if parameter_count < 0 or flops < 0:
        raise ValueError("parameter_count 和 flops 不能为负数")
    if training_sample_count is not None and training_sample_count <= 0:
        raise ValueError("training_sample_count 必须为 null 或正整数")
    best_loss = best_epoch = gap = training_hours = None
    if history is not None:
        best_index = int(np.argmin(history.valid_loss))
        best_loss = float(history.valid_loss[best_index])
        best_epoch = best_index + 1
        gap = float(history.valid_loss[-1] - history.train_loss[-1])
        if history.epoch_seconds is not None:
            training_hours = float(history.epoch_seconds.sum() / 3600)
    return ResearchOverview(
        model=model,
        parameter_count=parameter_count,
        flops=flops,
        training_sample_count=training_sample_count,
        evaluation_size=summary.evaluation_size,
        mean_surprisal_bits=summary.raw_surprisal_bits.mean,
        mean_bits_per_token=summary.bits_per_token.mean,
        median_bits_per_token=summary.bits_per_token.median,
        p95_bits_per_token=summary.bits_per_token.p95,
        perplexity=float(2 ** summary.bits_per_token.mean),
        best_validation_loss=best_loss,
        best_epoch=best_epoch,
        generalization_gap=gap,
        training_hours=training_hours,
        best_first_coverage=(
            None if best_first is None else coverage_at_budget(best_first, best_first_budget)
        ),
        random_coverage=(
            None if random_sampling is None else coverage_at_budget(random_sampling, random_budget)
        ),
    )


def mean_surprisal(data: Mapping[str, SurprisalData]) -> SurprisalData:
    """在完整评测大小和导出点位一致时逐项平均模型 surprisal。"""

    values = list(data.values())
    if not values:
        raise ValueError("至少需要一个模型")
    first = values[0]
    if any(
        item.evaluation_size != first.evaluation_size
        or item.surprisal_bits.shape != first.surprisal_bits.shape
        or item.bits_per_token.shape != first.bits_per_token.shape
        for item in values[1:]
    ):
        raise ValueError("模型的评测规模或导出点位数量不一致，不能逐项平均")
    return SurprisalData(
        evaluation_size=first.evaluation_size,
        surprisal_bits=np.mean([item.surprisal_bits for item in values], axis=0),
        bits_per_token=np.mean([item.bits_per_token for item in values], axis=0),
    )


def mean_coverage(data: Mapping[str, CoverageData]) -> CoverageData:
    """在测试集和 attempts 完全一致时逐检查点平均覆盖率。"""

    values = list(data.values())
    if not values:
        raise ValueError("至少需要一个模型")
    first = values[0]
    if any(
        item.test_size != first.test_size
        or not np.array_equal(item.attempts, first.attempts)
        for item in values[1:]
    ):
        raise ValueError("模型的测试集大小或 attempts 不一致，不能平均覆盖率")
    return CoverageData(
        test_size=first.test_size,
        attempts=first.attempts.copy(),
        coverage=np.mean([item.coverage for item in values], axis=0),
    )


def _color_scale(labels: list[str], colors: Mapping[str, str]) -> alt.Scale:
    """构造固定模型颜色，保证不同视图中的颜色语义一致。"""

    return alt.Scale(domain=labels, range=[colors[label] for label in labels])


def _uniform_indices(size: int, max_points: int) -> np.ndarray:
    """等距保留曲线点，并始终包含首尾点。"""

    if size <= max_points:
        return np.arange(size)
    return np.linspace(0, size - 1, num=max_points, dtype=np.int64)


def plot_training_history(
    histories: Mapping[str, TrainingHistory],
    colors: Mapping[str, str],
) -> alt.Chart:
    """绘制多模型训练/验证/测试损失，线型表示数据 split。"""

    if not histories:
        raise ValueError("没有可绘制的训练历史")
    rows: list[dict[str, float | int | str]] = []
    for label, history in histories.items():
        for epoch, (train_loss, valid_loss) in enumerate(
            zip(history.train_loss, history.valid_loss), start=1
        ):
            rows.append(
                {"Epoch": epoch, "Loss": train_loss, "Model": label, "Split": "Train"}
            )
            rows.append(
                {"Epoch": epoch, "Loss": valid_loss, "Model": label, "Split": "Validation"}
            )
            if history.test_loss is not None:
                rows.append(
                    {
                        "Epoch": epoch,
                        "Loss": history.test_loss,
                        "Model": label,
                        "Split": "Test",
                    }
                )
    labels = list(histories)
    splits = ["Train", "Validation"]
    dash_patterns = [[1, 0], [7, 4]]
    # 只有 artifact 确实记录了 Test Loss 时才加入曲线和图例，避免出现空图例项。
    if any(history.test_loss is not None for history in histories.values()):
        splits.append("Test")
        dash_patterns.append([2, 3])
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line()
        .encode(
            x=alt.X(
                "Epoch:Q",
                title="Epoch",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            y=alt.Y(
                "Loss:Q",
                title="Cross-Entropy Loss",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            strokeDash=alt.StrokeDash(
                "Split:N",
                sort=splits,
                scale=alt.Scale(
                    domain=splits,
                    range=dash_patterns,
                ),
            ),
            tooltip=["Model:N", "Split:N", "Epoch:Q", alt.Tooltip("Loss:Q", format=".4f")],
        )
        .properties(title="Training History", height=420)
    )


def plot_validation_loss_by_time(
    histories: Mapping[str, TrainingHistory],
    colors: Mapping[str, str],
) -> alt.Chart:
    """按累计训练小时绘制验证损失，使不同单轮速度的模型可以公平比较。"""

    available = {
        label: history
        for label, history in histories.items()
        if history.epoch_seconds is not None
    }
    if not available:
        raise ValueError("没有带 epoch_seconds 的训练历史")
    rows: list[dict[str, float | int | str]] = []
    for label, history in available.items():
        assert history.epoch_seconds is not None
        cumulative_hours = np.cumsum(history.epoch_seconds) / 3600
        rows.extend(
            {
                "Training Time (hours)": elapsed,
                "Validation Loss": loss,
                "Epoch": epoch,
                "Model": label,
            }
            for epoch, (elapsed, loss) in enumerate(
                zip(cumulative_hours, history.valid_loss), start=1
            )
        )
    labels = list(available)
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Training Time (hours):Q",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            y=alt.Y("Validation Loss:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=[
                "Model:N",
                "Epoch:Q",
                alt.Tooltip("Training Time (hours):Q", format=".3f"),
                alt.Tooltip("Validation Loss:Q", format=".4f"),
            ],
        )
        .properties(title="Validation Loss by Cumulative Training Time", height=420)
    )


def plot_generalization_gap(
    histories: Mapping[str, TrainingHistory],
    colors: Mapping[str, str],
) -> alt.LayerChart:
    """逐 epoch 绘制验证损失减训练损失，并用零线标出无差距位置。"""

    if not histories:
        raise ValueError("没有可绘制的训练历史")
    rows: list[dict[str, float | int | str]] = []
    for label, history in histories.items():
        rows.extend(
            {
                "Epoch": epoch,
                "Generalization Gap": valid_loss - train_loss,
                "Model": label,
            }
            for epoch, (train_loss, valid_loss) in enumerate(
                zip(history.train_loss, history.valid_loss), start=1
            )
        )

    labels = list(histories)
    line = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Epoch:Q",
                title="Epoch",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            y=alt.Y(
                "Generalization Gap:Q",
                title="Validation Loss − Training Loss",
            ),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=[
                "Model:N",
                "Epoch:Q",
                alt.Tooltip("Generalization Gap:Q", format=".4f"),
            ],
        )
    )
    zero_line = (
        alt.Chart(pd.DataFrame({"Generalization Gap": [0.0]}))
        .mark_rule(color="#8a96a3", strokeDash=[5, 4])
        .encode(y="Generalization Gap:Q")
    )
    return (line + zero_line).properties(title="Generalization Gap by Epoch", height=340)


def plot_learning_rate(
    histories: Mapping[str, TrainingHistory],
    colors: Mapping[str, str],
) -> alt.Chart:
    """绘制具有学习率记录的模型。"""

    available = {key: value for key, value in histories.items() if value.learning_rate is not None}
    if not available:
        raise ValueError("没有可绘制的学习率历史")
    rows: list[dict[str, float | int | str]] = []
    for label, history in available.items():
        assert history.learning_rate is not None
        rows.extend(
            {"Epoch": epoch, "Learning Rate": value, "Model": label}
            for epoch, value in enumerate(history.learning_rate, start=1)
        )
    labels = list(available)
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Epoch:Q",
                title="Epoch",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            y=alt.Y(
                "Learning Rate:Q",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=["Model:N", "Epoch:Q", alt.Tooltip("Learning Rate:Q", format=".6g")],
        )
        .properties(title="Learning Rate Schedule", height=320)
    )


def plot_model_zoo(
    overviews: list[ResearchOverview],
    colors: Mapping[str, str],
    metric: str = "Mean Bits / Token",
) -> alt.LayerChart:
    """按 Estimated FLOPs 排列模型，并用圆点面积的对称对数尺度编码参数量。"""

    if not overviews:
        raise ValueError("没有可绘制的研究总览")
    metrics = {
        "Mean Bits / Token": ("mean_bits_per_token", "Mean Bits per Token", False),
        "Random Coverage": ("random_coverage", "Random Coverage (%)", True),
        "Best-first Coverage": (
            "best_first_coverage",
            "Best-first Coverage (%)",
            True,
        ),
    }
    if metric not in metrics:
        raise ValueError(f"不支持的 Model Zoo 指标: {metric}")
    field, axis_title, percentage = metrics[metric]
    available = [item for item in overviews if getattr(item, field) is not None]
    if not available:
        raise ValueError(f"所选模型没有 {metric} 数据")
    labels = [item.model for item in available]
    minimum_area = 120
    maximum_area = 3_200
    # 较大的线性区间可减少 symlog 对低、中、高参数档位的压缩，同时仍让 Bigram 可见。
    symlog_constant = 1_000_000
    max_parameters = max(item.parameter_count for item in available)

    def label_offset(parameter_count: int) -> float:
        """复现圆点面积缩放，并在圆点半径之外保留 6 px 间距。"""

        if max_parameters == 0:
            area = minimum_area
        else:
            ratio = log1p(parameter_count / symlog_constant) / log1p(
                max_parameters / symlog_constant
            )
            area = minimum_area + ratio * (maximum_area - minimum_area)
        return sqrt(area / pi) + 6

    rows = [
        {
            "Model": item.model,
            "Parameter Count": item.parameter_count,
            "Estimated FLOPs": item.flops,
            "Value": getattr(item, field),
            "Label Offset": label_offset(item.parameter_count),
        }
        for item in available
    ]
    frame = pd.DataFrame(rows)
    y_axis = alt.Y(
        "Value:Q",
        title=axis_title,
        scale=alt.Scale(domainMin=0, zero=True) if percentage else alt.Scale(zero=False),
        axis=alt.Axis(format=".1%") if percentage else alt.Axis(),
    )
    base = alt.Chart(frame).encode(
        x=alt.X(
            "Estimated FLOPs:Q",
            scale=alt.Scale(domainMin=0, zero=True),
        ),
        y=y_axis,
    )
    points = base.mark_circle(
        opacity=1.0, stroke="#f2f5f8", strokeWidth=0.8
    ).encode(
        color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
        size=alt.Size(
            "Parameter Count:Q",
            scale=alt.Scale(
                type="symlog",
                constant=symlog_constant,
                domain=[0, max(max_parameters, 1)],
                range=[minimum_area, maximum_area],
            ),
            legend=None,
        ),
        tooltip=[
            "Model:N",
            alt.Tooltip("Estimated FLOPs:Q", format=","),
            alt.Tooltip("Parameter Count:Q", format=","),
            alt.Tooltip(
                "Value:Q", title=metric, format=".2%" if percentage else ".4f"
            ),
        ],
    )
    # dy 直接读取每条记录的圆点半径，让 Bigram 靠近小圆、较大模型避开大圆。
    labels_layer = base.mark_text(
        dy=alt.ExprRef(expr="-datum['Label Offset']"),
        align="center",
        baseline="bottom",
        fontWeight="bold",
    ).encode(
        text="Model:N",
        color=alt.Color("Model:N", scale=_color_scale(labels, colors), legend=None),
    )
    return (points + labels_layer).properties(title="Model Zoo", height=430)


def pairwise_win_rates(
    data: Mapping[str, SurprisalData],
    normalized: bool = False,
) -> pd.DataFrame:
    """计算逐样本低惊讶度胜率；平局为双方各计半胜，行模型与列模型比较。"""

    values = list(data.values())
    if len(values) < 2:
        raise ValueError("成对比较至少需要两个模型")
    first = values[0]
    if any(
        item.evaluation_size != first.evaluation_size
        or item.surprisal_bits.shape != first.surprisal_bits.shape
        or item.bits_per_token.shape != first.bits_per_token.shape
        for item in values[1:]
    ):
        raise ValueError("模型的评测规模或导出点位数量不一致，不能成对比较")

    labels = list(data)
    arrays = [
        item.bits_per_token if normalized else item.surprisal_bits
        for item in data.values()
    ]
    matrix = np.empty((len(labels), len(labels)), dtype=np.float64)
    for row, left in enumerate(arrays):
        for column, right in enumerate(arrays):
            matrix[row, column] = np.mean(left < right) + 0.5 * np.mean(left == right)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def pairwise_difference(
    left: SurprisalData,
    right: SurprisalData,
    normalized: bool = False,
) -> np.ndarray:
    """返回对齐样本的左模型减右模型惊讶度，负值表示左模型更占优。"""

    if (
        left.evaluation_size != right.evaluation_size
        or left.surprisal_bits.shape != right.surprisal_bits.shape
        or left.bits_per_token.shape != right.bits_per_token.shape
    ):
        raise ValueError("两个模型的评测规模或导出点位数量不一致")
    left_values = left.bits_per_token if normalized else left.surprisal_bits
    right_values = right.bits_per_token if normalized else right.surprisal_bits
    return left_values - right_values


def plot_pairwise_win_rates(matrix: pd.DataFrame) -> alt.LayerChart:
    """将成对低惊讶度胜率绘制为以 50% 为中点的热力图。"""

    labels = list(matrix.index)
    rows = [
        {"Row Model": row, "Column Model": column, "Win Rate": matrix.loc[row, column]}
        for row in labels
        for column in labels
    ]
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(
        x=alt.X("Column Model:N", sort=labels, title="Compared Model"),
        y=alt.Y("Row Model:N", sort=labels, title="Candidate Model"),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "Win Rate:Q",
            scale=alt.Scale(domain=[0, 0.5, 1], range=["#d95f59", "#e8edf2", "#2e9f83"]),
            legend=alt.Legend(format=".0%"),
        ),
        tooltip=[
            "Row Model:N",
            "Column Model:N",
            alt.Tooltip("Win Rate:Q", format=".1%"),
        ],
    )
    labels_layer = base.mark_text().encode(text=alt.Text("Win Rate:Q", format=".0%"))
    return (heatmap + labels_layer).properties(
        title="Pairwise Lower-Surprisal Win Rate", height=420
    )


def plot_pairwise_difference(
    left_label: str,
    right_label: str,
    differences: np.ndarray,
    normalized: bool = False,
    bins: int = 60,
) -> alt.LayerChart:
    """绘制对齐样本的惊讶度差值分布，并用零线表示两模型分界。"""

    values = _finite_vector(differences, "differences")
    density, edges = np.histogram(values, bins=bins, density=True)
    frame = pd.DataFrame({"Difference": edges[:-1], "Density": density})
    metric = "Bits per Token Difference" if normalized else "Surprisal Difference (bits)"
    histogram = (
        alt.Chart(frame)
        .mark_area(opacity=0.65, interpolate="step-after")
        .encode(
            x=alt.X("Difference:Q", title=f"{left_label} − {right_label} {metric}"),
            y=alt.Y("Density:Q", scale=alt.Scale(domainMin=0, zero=True)),
            tooltip=[
                alt.Tooltip("Difference:Q", format=".3f"),
                alt.Tooltip("Density:Q", format=".4f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"Difference": [0.0]})).mark_rule(
        color="#f2f5f8", strokeDash=[5, 4]
    ).encode(x="Difference:Q")
    return (histogram + zero).properties(title="Paired Surprisal Difference", height=340)


def plot_surprisal_histogram(
    data: Mapping[str, SurprisalData],
    colors: Mapping[str, str],
    normalized: bool = False,
    bins: int = 50,
) -> alt.Chart:
    """服务端先计算公共区间直方图，再传递少量密度点给浏览器。"""

    if not data:
        raise ValueError("没有可绘制的 surprisal 数据")
    all_values = [
        item.bits_per_token if normalized else item.surprisal_bits
        for item in data.values()
    ]
    lower = min(float(values.min()) for values in all_values)
    upper = max(float(values.max()) for values in all_values)
    if lower == upper:
        margin = max(abs(lower) * 0.01, 0.01)
        lower, upper = lower - margin, upper + margin
    edges = np.linspace(lower, upper, num=bins + 1)

    # 所有模型共享同一组 bin，曲线的横向位置和密度才可以直接比较。
    rows: list[dict[str, float | str]] = []
    for (label, _), values in zip(data.items(), all_values):
        density, _ = np.histogram(values, bins=edges, density=True)
        rows.extend(
            {"Value": edge, "Density": value, "Model": label}
            for edge, value in zip(edges[:-1], density)
        )
    labels = list(data)
    metric = "Bits per Token" if normalized else "Surprisal (bits)"
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X(
                "Value:Q",
                title=metric,
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            y=alt.Y(
                "Density:Q",
                title="Density",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=["Model:N", alt.Tooltip("Value:Q", format=".3f"), alt.Tooltip("Density:Q", format=".4f")],
        )
        .properties(
            title="Bits per Token Distribution" if normalized else "Surprisal Distribution",
            height=420,
        )
    )


def plot_surprisal_boxplot(
    data: Mapping[str, SurprisalData],
    colors: Mapping[str, str],
    normalized: bool = False,
) -> alt.LayerChart:
    """在服务端计算 Tukey 箱线统计量，避免极端值压缩箱体和传输完整数组。"""

    if not data:
        raise ValueError("没有可绘制的 surprisal 数据")
    rows = []
    for label, item in data.items():
        values = item.bits_per_token if normalized else item.surprisal_bits
        minimum, q1, median, q3, maximum = np.quantile(
            values, [0, 0.25, 0.5, 0.75, 1]
        )
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        inliers = values[(values >= lower_fence) & (values <= upper_fence)]
        rows.append(
            {
                "Model": label,
                "Minimum": minimum,
                "Lower Whisker": float(inliers.min()),
                "Q1": q1,
                "Median": median,
                "Q3": q3,
                "Upper Whisker": float(inliers.max()),
                "Maximum": maximum,
                "Outliers": int(values.size - inliers.size),
            }
        )
    labels = list(data)
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(
        x=alt.X("Model:N", sort=labels, title=None),
        tooltip=[
            "Model:N",
            alt.Tooltip("Minimum:Q", format=".3f"),
            alt.Tooltip("Lower Whisker:Q", format=".3f"),
            alt.Tooltip("Q1:Q", format=".3f"),
            alt.Tooltip("Median:Q", format=".3f"),
            alt.Tooltip("Q3:Q", format=".3f"),
            alt.Tooltip("Upper Whisker:Q", format=".3f"),
            alt.Tooltip("Maximum:Q", format=".3f"),
            "Outliers:Q",
        ],
    )
    color = alt.Color("Model:N", scale=_color_scale(labels, colors), legend=None)
    positive_scale = alt.Scale(domainMin=0, zero=True)
    metric = "Bits per Token" if normalized else "Surprisal (bits)"
    whisker = base.mark_rule().encode(
        y=alt.Y("Lower Whisker:Q", title=metric, scale=positive_scale),
        y2="Upper Whisker:Q",
        color=color,
    )
    box = base.mark_bar(size=34).encode(
        y=alt.Y("Q1:Q", title=metric, scale=positive_scale),
        y2="Q3:Q",
        color=color,
    )
    median = base.mark_tick(color="#ffffff", size=32, thickness=2).encode(
        y=alt.Y("Median:Q", title=metric, scale=positive_scale)
    )
    return (
        (whisker + box + median)
        .properties(
            title="Bits per Token Comparison" if normalized else "Surprisal Comparison",
            height=340,
        )
        .resolve_scale(color="shared")
    )


def plot_coverage(
    data: Mapping[str, CoverageData],
    colors: Mapping[str, str],
    efficiency: bool = False,
    sampling: bool = False,
    max_points_per_model: int = 2_500,
) -> alt.Chart:
    """等距限点后绘制搜索或随机采样的覆盖率/效率曲线。"""

    if not data:
        raise ValueError("没有可绘制的覆盖率数据")
    if max_points_per_model < 2:
        raise ValueError("max_points_per_model 必须至少为 2")
    rows: list[dict[str, float | int | str]] = []
    for label, item in data.items():
        values = item.efficiency if efficiency else item.coverage
        indices = _uniform_indices(item.attempts.size, max_points_per_model)
        rows.extend(
            {"Attempts": attempt, "Value": value, "Model": label}
            for attempt, value in zip(item.attempts[indices], values[indices])
        )
    labels = list(data)
    attempt_name = "Sampling Attempts" if sampling else "Search Attempts"
    metric_name = "Coverage / Attempts" if efficiency else "Coverage"
    if efficiency:
        value_axis = alt.Y(
            "Value:Q",
            title=metric_name,
            scale=alt.Scale(domainMin=0, zero=True),
        )
        value_tooltip = alt.Tooltip("Value:Q", title=metric_name, format=".6g")
    else:
        value_axis = alt.Y(
            "Value:Q",
            title="Coverage (%)",
            scale=alt.Scale(domainMin=0, zero=True),
            axis=alt.Axis(format=".1%"),
        )
        value_tooltip = alt.Tooltip("Value:Q", title=metric_name, format=".1%")
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line()
        .encode(
            x=alt.X(
                "Attempts:Q",
                title=attempt_name,
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            y=value_axis,
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=[
                "Model:N",
                alt.Tooltip("Attempts:Q", format=","),
                value_tooltip,
            ],
        )
        .properties(title=f"{metric_name} by {attempt_name}", height=420)
    )


def plot_generation_quality(
    data: Mapping[str, GenerationQuality],
    colors: Mapping[str, str],
) -> alt.Chart:
    """绘制多模型合法率和合法唯一率。"""

    if not data:
        raise ValueError("没有可绘制的生成质量数据")
    rows = [
        {"Model": label, "Metric": metric, "Rate": value}
        for label, item in data.items()
        for metric, value in (
            ("Legal Rate", item.legal_rate),
            ("Legal Unique Rate", item.legal_unique_rate),
        )
    ]
    labels = list(data)
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=labels, title=None),
            xOffset=alt.XOffset("Metric:N", sort=["Legal Rate", "Legal Unique Rate"]),
            y=alt.Y("Rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors), legend=None),
            opacity=alt.Opacity(
                "Metric:N",
                sort=["Legal Rate", "Legal Unique Rate"],
                scale=alt.Scale(domain=["Legal Rate", "Legal Unique Rate"], range=[1.0, 0.5]),
            ),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Rate:Q", format=".2%")],
        )
        .properties(title="Random Generation Quality", height=380)
    )
