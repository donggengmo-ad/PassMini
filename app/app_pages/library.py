"""Library：多模型离线评测页面。"""

import numpy as np
import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import render_current_selection, selected_records
from app.frontend.library import (
    GenerationQuality,
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


def show_chart(chart) -> None:
    """使用 Streamlit 原生 Altair 渲染器展示交互图。"""

    st.altair_chart(chart, width="stretch")


def report_unavailable(names: list[str]) -> None:
    """集中列出当前视图缺少 artifact 的模型。"""

    if names:
        st.caption("Unavailable: " + " · ".join(names))


catalog = load_catalog()
records = selected_records(catalog)
st.title("Library", anchor=False)
render_current_selection(records)
if not records:
    st.stop()

colors = {record.display_name: record.color for record in records}
view = st.segmented_control(
    "View",
    ["Training", "Surprisal", "Coverage", "Generation Quality"],
    default="Training",
    required=True,
    key="library_view",
    width="stretch",
    persist_state="session",
)
show_aggregate = st.toggle(
    "Show aggregate",
    value=False,
    key="library_show_aggregate",
    persist_state="session",
)

# 每次 rerun 只进入当前视图，避免隐藏标签页仍读取数据并构造全部图表。
if view == "Training":
    histories = {}
    missing = []
    for record in records:
        try:
            histories[record.display_name] = load_training_history(record.history_path)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
    if histories:
        show_chart(plot_training_history(histories, colors))
        if any(history.learning_rate is not None for history in histories.values()):
            show_chart(plot_learning_rate(histories, colors))
    else:
        st.info("所选模型暂时没有可绘制的训练历史。")
    report_unavailable(missing)

elif view == "Surprisal":
    normalized = st.radio(
        "Metric",
        ["Surprisal bits", "Bits per token"],
        horizontal=True,
        key="library_surprisal_metric",
        persist_state="session",
    ) == "Bits per token"
    surprisal = {}
    missing = []
    for record in records:
        try:
            surprisal[record.display_name] = load_surprisal_data(
                record.evaluation_dir / "surprisal.npz"
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
    surprisal_colors = dict(colors)
    if show_aggregate and len(surprisal) > 1:
        try:
            surprisal["Selected Mean"] = mean_surprisal(surprisal)
            surprisal_colors["Selected Mean"] = "#f2f5f8"
        except ValueError as error:
            st.warning(str(error))
    if surprisal:
        metric_cards = st.columns(min(4, len(surprisal)))
        for column, (label, item) in zip(metric_cards, surprisal.items()):
            values = item.bits_per_token if normalized else item.surprisal_bits
            column.metric(label, f"{np.mean(values):.3f}", help="Mean of exported points")
        show_chart(plot_surprisal_histogram(surprisal, surprisal_colors, normalized))
        show_chart(plot_surprisal_boxplot(surprisal, surprisal_colors, normalized))
    else:
        st.info("评测仍在运行或 NPZ 尚未打包；完成后本页会直接读取结果。")
    report_unavailable(missing)

elif view == "Coverage":
    mode = st.radio(
        "Evaluation",
        ["Best-first", "Random sampling"],
        horizontal=True,
        key="library_coverage_mode",
        persist_state="session",
    )
    filename = "best_first_coverage.npz" if mode == "Best-first" else "random_coverage.npz"
    coverage = {}
    missing = []
    for record in records:
        try:
            coverage[record.display_name] = load_coverage_data(
                record.evaluation_dir / filename
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
    coverage_colors = dict(colors)
    if show_aggregate and len(coverage) > 1:
        try:
            coverage["Selected Mean"] = mean_coverage(coverage)
            coverage_colors["Selected Mean"] = "#f2f5f8"
        except ValueError as error:
            st.warning(str(error))
    if coverage:
        sampling = mode == "Random sampling"
        show_chart(plot_coverage(coverage, coverage_colors, sampling=sampling))
        show_chart(
            plot_coverage(
                coverage,
                coverage_colors,
                efficiency=True,
                sampling=sampling,
            )
        )
    else:
        st.info("所选模型暂时没有对应的覆盖率 NPZ。")
    report_unavailable(missing)

elif view == "Generation Quality":
    quality = {}
    missing = []
    for record in records:
        try:
            quality[record.display_name] = load_generation_quality(
                record.evaluation_dir / "summary.json"
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
    quality_colors = dict(colors)
    if show_aggregate and len(quality) > 1:
        values = list(quality.values())
        if len({value.total_samples for value in values}) == 1:
            quality["Selected Mean"] = GenerationQuality(
                total_samples=values[0].total_samples,
                legal_rate=float(np.mean([value.legal_rate for value in values])),
                legal_unique_rate=float(
                    np.mean([value.legal_unique_rate for value in values])
                ),
            )
            quality_colors["Selected Mean"] = "#f2f5f8"
        else:
            st.warning("采样数量不一致，不能计算生成质量平均值。")
    if quality:
        show_chart(plot_generation_quality(quality, quality_colors))
    else:
        st.info("随机生成质量摘要尚未打包。")
    report_unavailable(missing)

st.caption(
    "Surprisal and coverage are relative to the recorded model, test set, and budget; "
    "they are not estimates of real-world cracking time."
)
