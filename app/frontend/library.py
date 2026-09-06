"""Library 的静态评测数据读取、聚合与交互绘图。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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


def _finite_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须全部是有限值")
    return array


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
    raw_test_loss = payload.get("test_loss")
    test_loss = None if raw_test_loss is None else float(raw_test_loss)
    if test_loss is not None and not np.isfinite(test_loss):
        raise ValueError("test_loss 必须是有限值")
    return TrainingHistory(train_loss, valid_loss, learning_rate, test_loss)


@st.cache_data(show_spinner=False, max_entries=32)
def load_surprisal_data(path: str | Path) -> SurprisalData:
    """读取单模型 surprisal NPZ，并复制数组以关闭文件句柄。"""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"evaluation_size", "surprisal_bits", "bits_per_token"}
        if not required.issubset(payload.files):
            raise ValueError("surprisal NPZ 缺少必需字段")
        evaluation_size = int(np.asarray(payload["evaluation_size"]).item())
        surprisal = _finite_vector(payload["surprisal_bits"], "surprisal_bits").copy()
        per_token = _finite_vector(payload["bits_per_token"], "bits_per_token").copy()
    if evaluation_size <= 0 or surprisal.size != per_token.size:
        raise ValueError("surprisal NPZ 的数量信息不一致")
    return SurprisalData(evaluation_size, surprisal, per_token)


@st.cache_data(show_spinner=False, max_entries=32)
def load_coverage_data(path: str | Path) -> CoverageData:
    """读取单模型覆盖率 NPZ。"""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"test_size", "attempts", "coverage"}
        if not required.issubset(payload.files):
            raise ValueError("coverage NPZ 缺少必需字段")
        test_size = int(np.asarray(payload["test_size"]).item())
        attempts = np.asarray(payload["attempts"], dtype=np.int64).copy()
        coverage = _finite_vector(payload["coverage"], "coverage").copy()
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
            rows.append({"Epoch": epoch, "Loss": train_loss, "Model": label, "Split": "Train"})
            rows.append(
                {"Epoch": epoch, "Loss": valid_loss, "Model": label, "Split": "Validation"}
            )
            if history.test_loss is not None:
                rows.append(
                    {"Epoch": epoch, "Loss": history.test_loss, "Model": label, "Split": "Test"}
                )
    labels = list(histories)
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line()
        .encode(
            x=alt.X("Epoch:Q", title="Epoch"),
            y=alt.Y("Loss:Q", title="Cross-Entropy Loss", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            strokeDash=alt.StrokeDash(
                "Split:N",
                sort=["Train", "Validation", "Test"],
                scale=alt.Scale(
                    domain=["Train", "Validation", "Test"],
                    range=[[1, 0], [7, 4], [2, 3]],
                ),
            ),
            tooltip=["Model:N", "Split:N", "Epoch:Q", alt.Tooltip("Loss:Q", format=".4f")],
        )
        .properties(title="Training History", height=420)
        .interactive(bind_y=False)
    )


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
            x=alt.X("Epoch:Q", title="Epoch"),
            y=alt.Y("Learning Rate:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=["Model:N", "Epoch:Q", alt.Tooltip("Learning Rate:Q", format=".6g")],
        )
        .properties(title="Learning Rate Schedule", height=320)
    )


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
            x=alt.X("Value:Q", title=metric),
            y=alt.Y("Density:Q", title="Density"),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=["Model:N", alt.Tooltip("Value:Q", format=".3f"), alt.Tooltip("Density:Q", format=".4f")],
        )
        .properties(
            title="Bits per Token Distribution" if normalized else "Surprisal Distribution",
            height=420,
        )
        .interactive(bind_y=False)
    )


def plot_surprisal_boxplot(
    data: Mapping[str, SurprisalData],
    colors: Mapping[str, str],
    normalized: bool = False,
) -> alt.LayerChart:
    """在服务端计算五数概括，避免把完整评分数组发送给浏览器。"""

    if not data:
        raise ValueError("没有可绘制的 surprisal 数据")
    rows = []
    for label, item in data.items():
        values = item.bits_per_token if normalized else item.surprisal_bits
        minimum, q1, median, q3, maximum = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
        rows.append(
            {
                "Model": label,
                "Minimum": minimum,
                "Q1": q1,
                "Median": median,
                "Q3": q3,
                "Maximum": maximum,
            }
        )
    labels = list(data)
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(x=alt.X("Model:N", sort=labels, title=None))
    color = alt.Color("Model:N", scale=_color_scale(labels, colors), legend=None)
    whisker = base.mark_rule().encode(y="Minimum:Q", y2="Maximum:Q", color=color)
    box = base.mark_bar(size=34).encode(y="Q1:Q", y2="Q3:Q", color=color)
    median = base.mark_tick(color="#ffffff", size=32, thickness=2).encode(y="Median:Q")
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
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_line()
        .encode(
            x=alt.X("Attempts:Q", title=attempt_name),
            y=alt.Y("Value:Q", title=metric_name),
            color=alt.Color("Model:N", scale=_color_scale(labels, colors)),
            tooltip=["Model:N", alt.Tooltip("Attempts:Q", format=","), alt.Tooltip("Value:Q", format=".6g")],
        )
        .properties(title=f"{metric_name} by {attempt_name}", height=420)
        .interactive(bind_y=False)
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
