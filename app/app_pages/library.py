"""Library：多模型离线评测与研究比较页面。"""

import numpy as np
import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import render_current_selection, selected_records
from app.frontend.library import (
    GenerationQuality,
    build_research_overview,
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


def show_chart(chart) -> None:
    """使用 Streamlit 原生 Altair 渲染器展示交互图。"""

    st.altair_chart(chart, width="stretch")


def report_unavailable(names: list[str]) -> None:
    """集中列出当前视图缺少 artifact 的模型。"""

    if names:
        st.caption("暂不可用：" + " · ".join(names))


catalog = load_catalog()
records = selected_records(catalog)
st.title("Library", anchor=False)
render_current_selection(records)
if not records:
    st.stop()

colors = {record.display_name: record.color for record in records}
view = st.segmented_control(
    "View",
    [
        "Research Overview",
        "Training",
        "Surprisal",
        "Pairwise Comparison",
        "Coverage",
        "Generation Quality",
    ],
    default="Research Overview",
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
    disabled=view in {"Research Overview", "Pairwise Comparison"},
)

# 每次 rerun 只进入当前视图，避免隐藏子页仍读取数据并构造全部图表。
if view == "Research Overview":
    overviews = []
    missing = []
    for record in records:
        try:
            summary = load_evaluation_summary(record.evaluation_dir / "summary.json")
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
            continue

        # 训练历史和覆盖率允许缺失；总览只把已有实测结果填入对应列。
        try:
            history = load_training_history(record.history_path)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            history = None
        try:
            best_first = load_coverage_data(
                record.evaluation_dir / "best_first_coverage.npz"
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            best_first = None
        try:
            random_sampling = load_coverage_data(
                record.evaluation_dir / "random_coverage.npz"
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            random_sampling = None
        overviews.append(
            build_research_overview(
                model=record.display_name,
                parameter_count=record.parameter_count,
                flops=record.flops,
                training_sample_count=record.training_sample_count,
                summary=summary,
                history=history,
                best_first=best_first,
                random_sampling=random_sampling,
            )
        )

    if overviews:
        st.caption(
            "完整测试集统计 · Best-first 预算：500,000 · "
            "Random sampling 预算：100,000,000"
        )
        st.dataframe(
            [
                {
                    "Model": item.model,
                    "Parameters": item.parameter_count,
                    "Estimated FLOPs": item.flops,
                    "Training Samples": item.training_sample_count,
                    "Evaluation Size": item.evaluation_size,
                    "Mean Surprisal": item.mean_surprisal_bits,
                    "Mean Bits / Token": item.mean_bits_per_token,
                    "Median Bits / Token": item.median_bits_per_token,
                    "P95 Bits / Token": item.p95_bits_per_token,
                    "Perplexity": item.perplexity,
                    "Best Validation Loss": item.best_validation_loss,
                    "Best Epoch": item.best_epoch,
                    "Final Generalization Gap": item.generalization_gap,
                    "Training Hours": item.training_hours,
                    "Best-first Coverage": item.best_first_coverage,
                    "Random Coverage": item.random_coverage,
                }
                for item in overviews
            ],
            column_config={
                "Model": st.column_config.TextColumn(
                    help="Warehouse 中登记的模型架构与规模档位。",
                    pinned=True,
                ),
                "Parameters": st.column_config.NumberColumn(
                    format="localized",
                    help="模型的可训练参数量。",
                ),
                "Estimated FLOPs": st.column_config.NumberColumn(
                    format="localized",
                    help=(
                        "输入 [1, 13] 时生成完整 logits 的解析式 FLOPs 估计。\n\n"
                        "覆盖矩阵乘法、卷积、门控、因果注意力、归一化和主要逐元素运算。"
                    ),
                ),
                "Training Samples": st.column_config.NumberColumn(
                    format="localized",
                    help="该模型实际训练时使用的去重训练密码数量。",
                ),
                "Evaluation Size": st.column_config.NumberColumn(
                    format="localized",
                    help="完整测试集中参与 surprisal 评测的密码数量。",
                ),
                "Mean Surprisal": st.column_config.NumberColumn(
                    format="%.4f",
                    help="完整测试集上的平均总惊讶度，包含 EOS，单位为 bit；越低越好。",
                ),
                "Mean Bits / Token": st.column_config.NumberColumn(
                    format="%.4f",
                    help="每条密码总惊讶度除以字符数加 EOS 后的全量平均；越低越好。",
                ),
                "Median Bits / Token": st.column_config.NumberColumn(
                    format="%.4f",
                    help="完整测试集 Bits/Token 的中位数，对极端高惊讶度样本更稳健。",
                ),
                "P95 Bits / Token": st.column_config.NumberColumn(
                    format="%.4f",
                    help="95% 测试密码的 Bits/Token 不超过该值，用于观察分布尾部。",
                ),
                "Perplexity": st.column_config.NumberColumn(
                    format="%.3f",
                    help="由 $2^{\\frac{Mean Bits}{Token}}$ 得到的平均分支复杂度；越低表示拟合越集中。",
                ),
                "Best Validation Loss": st.column_config.NumberColumn(
                    format="%.4f",
                    help="训练历史中最低的验证集交叉熵损失。Bigram 不使用该训练流程。",
                ),
                "Best Epoch": st.column_config.NumberColumn(
                    format="%d",
                    help="最低验证损失首次出现的 epoch。",
                ),
                "Final Generalization Gap": st.column_config.NumberColumn(
                    format="%.4f",
                    help="最后一轮验证损失减训练损失；正值越大通常表示泛化差距越明显。",
                ),
                "Training Hours": st.column_config.NumberColumn(
                    format="%.2f",
                    help="history.json 中所有 epoch_seconds 的总和，单位为小时。",
                ),
                "Best-first Coverage": st.column_config.NumberColumn(
                    format="percent",
                    help="Best-first 搜索在不超过 500,000 次尝试的最后检查点覆盖率。",
                ),
                "Random Coverage": st.column_config.NumberColumn(
                    format="percent",
                    help="随机采样在不超过 100,000,000 次尝试的最后检查点覆盖率。",
                ),
            },
            hide_index=True,
            width="stretch",
        )
        zoo_metric = st.segmented_control(
            "Model Zoo metric",
            ["Mean Bits / Token", "Random Coverage", "Best-first Coverage"],
            default="Mean Bits / Token",
            required=True,
            key="library_model_zoo_metric",
            width="stretch",
            persist_state="session",
        )
        show_chart(plot_model_zoo(overviews, colors, zoo_metric))
        st.caption(
            "Model Zoo 横轴使用 batch size 为 1、sequence length 为 13 时的 "
            "Estimated FLOPs；圆点面积通过扩展后的 symlog 尺度编码 Parameters。"
            "架构由色相区分，同架构的 tier 由同色系的明度与饱和度区分"
            "Coverage 为实际检查点的取样。"
        )
    else:
        st.info("所选模型暂时没有完整评测摘要。")
    report_unavailable(missing)

elif view == "Training":
    histories = {}
    missing = []
    for record in records:
        try:
            histories[record.display_name] = load_training_history(record.history_path)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
    if histories:
        show_chart(plot_training_history(histories, colors))
        if not any(history.test_loss is not None for history in histories.values()):
            st.caption(
                "当前选中模型的 history.json 未记录 Test Loss，因此图中只显示 "
                "Train 与 Validation。"
            )
        if any(history.epoch_seconds is not None for history in histories.values()):
            show_chart(plot_validation_loss_by_time(histories, colors))
        show_chart(plot_generalization_gap(histories, colors))
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
    summaries = {}
    missing = []
    for record in records:
        try:
            model_surprisal = load_surprisal_data(
                record.evaluation_dir / "surprisal.npz"
            )
            model_summary = load_evaluation_summary(
                record.evaluation_dir / "summary.json"
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            missing.append(record.display_name)
            continue
        # 两个文件都通过校验后再同步登记，保证后续按同名读取时键集合完全一致。
        surprisal[record.display_name] = model_surprisal
        summaries[record.display_name] = model_summary
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
            if label == "Selected Mean":
                values = item.bits_per_token if normalized else item.surprisal_bits
                value = float(np.mean(values))
                help_text = (
                    "基于对齐导出点位的模型集合平均。\n\n"
                    "该卡片不是单模型的全量 summary 统计。"
                )
            else:
                metric = (
                    summaries[label].bits_per_token
                    if normalized
                    else summaries[label].raw_surprisal_bits
                )
                value = metric.mean
                help_text = (
                    "完整测试集统计：\n\n"
                    f"- Mean: `{metric.mean:.3f}`\n\n"
                    f"- Median: `{metric.median:.3f}`\n\n"
                    f"- P90: `{metric.p90:.3f}`\n\n"
                    f"- P95: `{metric.p95:.3f}`\n\n"
                    f"- P99: `{metric.p99:.3f}`\n\n"
                    f"- Minimum: `{metric.minimum:.3f}`\n\n"
                    f"- Maximum: `{metric.maximum:.3f}`\n\n"
                    f"- Count: `{metric.count:,}`"
                )
            column.metric(label, f"{value:.3f}", help=help_text, border=True)
        show_chart(plot_surprisal_histogram(surprisal, surprisal_colors, normalized))
        show_chart(plot_surprisal_boxplot(surprisal, surprisal_colors, normalized))
        st.caption(
            "指标卡使用完整评测集的 summary 统计；分布图使用确定性等距导出的采样点。"
            "箱线图采用 Tukey 1.5×IQR whisker，完整 Minimum、Maximum 与异常值数量"
            "可在悬停时查看。"
        )
    else:
        st.info("评测仍在运行或 NPZ 尚未打包；完成后本页会直接读取结果。")
    report_unavailable(missing)

elif view == "Pairwise Comparison":
    normalized = st.radio(
        "Metric",
        ["Surprisal bits", "Bits per token"],
        horizontal=True,
        key="library_pairwise_metric",
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
    if len(surprisal) >= 2:
        try:
            matrix = pairwise_win_rates(surprisal, normalized)
            show_chart(plot_pairwise_win_rates(matrix))
            pair_labels = list(surprisal)
            left_column, right_column = st.columns(2)
            left_label = left_column.selectbox("Left model", pair_labels, key="pairwise_left")
            right_options = [label for label in pair_labels if label != left_label]
            right_label = right_column.selectbox(
                "Right model", right_options, key="pairwise_right"
            )
            differences = pairwise_difference(
                surprisal[left_label], surprisal[right_label], normalized
            )
            wins = float(np.mean(differences < 0) + 0.5 * np.mean(differences == 0))
            win_card, mean_card, median_card = st.columns(3)
            win_card.metric(
                f"{left_label} Win Rate",
                f"{wins:.1%}",
                help=(
                    "左模型在对齐密码上获得更低 surprisal 的比例。\n\n"
                    "相等样本为双方各计 0.5 次胜利。"
                ),
                border=True,
            )
            mean_card.metric(
                "Mean Difference",
                f"{np.mean(differences):.3f}",
                help=(
                    "所有对齐样本上 `left − right` surprisal 差值的算术平均。\n\n"
                    "负值表示左模型平均为密码分配了更高概率。"
                ),
                border=True,
            )
            median_card.metric(
                "Median Difference",
                f"{np.median(differences):.3f}",
                help=(
                    "所有对齐样本上 `left − right` 差值的中位数。\n\n"
                    "它比均值更不容易被少量极端密码影响。"
                ),
                border=True,
            )
            show_chart(
                plot_pairwise_difference(left_label, right_label, differences, normalized)
            )
            st.caption(
                "Row model Win Rate 表示该模型在对齐导出密码上取得更低 Surprisal 的"
                "比例，相同分数按 0.5 次胜利计算。Pairwise Difference 为负时更支持"
                "左侧模型。"
            )
        except ValueError as error:
            st.warning(str(error))
    else:
        st.info("成对比较至少需要两个具有 surprisal NPZ 的模型。")
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
    "Surprisal 与 Coverage 受限于测试集和预算，不一定直接反映真实破解时间。"
)
