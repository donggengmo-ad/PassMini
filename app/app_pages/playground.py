"""Playground：评分、逐字符分析与生成实验。"""

import numpy as np
import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import render_current_selection, selected_records
from app.frontend.probability_input import character_input, colored_password_html
from app.frontend.playground import (
    aggregate_scores,
    complete_with_models,
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
from app.frontend.warehouse import get_runtime_model


catalog = load_catalog()
records = selected_records(catalog)
st.title("Playground", anchor=False)
render_current_selection(records)
st.warning(
    "输入不会被记录，但不建议输入正在使用的真实密码。",
    icon=":material/warning:",
)
if not records:
    st.stop()


def load_selected_models():
    """保持 Warehouse 选择顺序加载模型，并让资源缓存复用权重。"""

    return {record.id: get_runtime_model(record) for record in records}


def load_one_model(model_id: str):
    """只加载一个已装备模型，供单模型逐字符实验使用。"""

    record = next(record for record in records if record.id == model_id)
    return get_runtime_model(record)


def show_chart(chart) -> None:
    """使用 Streamlit 原生 Altair 渲染器展示交互图。"""

    st.altair_chart(chart, width="stretch")


def render_scores(password: str, source: str) -> None:
    """对指定演示密码执行多模型评分，并统一渲染指标、图表和明细。"""

    with st.spinner("正在加载模型并评分……"):
        scores = score_models(password, load_selected_models())
        aggregate = aggregate_scores(scores)
    st.code(password, language=None)
    st.caption(source)
    mean_a, mean_b, spread = st.columns(3)
    mean_a.metric("Mean Surprisal", f"{aggregate.mean_surprisal_bits:.3f} bits")
    mean_b.metric("Mean Bits / Token", f"{aggregate.mean_bits_per_token:.3f}")
    spread.metric("Model Std. Dev.", f"{aggregate.standard_deviation:.3f}")
    show_chart(plot_score_comparison(scores, labels, colors))
    st.dataframe(
        [
            {
                "Model": labels[result.model_id],
                "Surprisal (bits)": result.surprisal_bits,
                "Bits / Token": result.bits_per_token,
            }
            for result in scores
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Mean Surprisal 是 bit 空间中的算术平均，等价于对各模型概率取几何平均。"
    )


labels = {record.id: record.display_name for record in records}
colors = {record.id: record.color for record in records}
area = st.segmented_control(
    "Experiment Area",
    ["Score Arena", "Character Lab", "Generation Lab"],
    default="Score Arena",
    required=True,
    key="playground_area",
    width="stretch",
    persist_state="session",
)

# 子页使用条件分支而非 tabs，避免隐藏实验仍执行实时模型推理。
if area == "Score Arena":
    score_mode = st.segmented_control(
        "Score Mode",
        ["Manual Input", "Generator versus Judge"],
        default="Manual Input",
        required=True,
        key="playground_score_mode",
        width="stretch",
        persist_state="session",
    )
    if score_mode == "Manual Input":
        with st.form("password-scoring"):
            password = st.text_input(
                "Demo password",
                type="password",
                max_chars=12,
                help="接受 1–12 个可打印 ASCII 字符。",
            )
            submitted = st.form_submit_button(
                "Score with selected models",
                type="primary",
                icon=":material/query_stats:",
            )
        if submitted:
            try:
                render_scores(password, "手动输入的演示文本")
            except (FileNotFoundError, TypeError, ValueError) as error:
                st.error(str(error))

    elif score_mode == "Generator versus Judge":
        with st.form("generator-versus-judge"):
            generator_id = st.selectbox(
                "Generator model",
                [record.id for record in records],
                format_func=labels.__getitem__,
            )
            length_column, temperature_column, seed_column = st.columns(3)
            max_length = length_column.slider("Maximum length", 1, 12, 12)
            temperature = temperature_column.slider(
                "Temperature", 0.1, 2.0, 1.0, 0.1
            )
            seed = seed_column.number_input("Seed", min_value=0, value=2026, step=1)
            submitted = st.form_submit_button(
                "Generate and judge",
                type="primary",
                icon=":material/swords:",
            )
        if submitted:
            try:
                runtime = load_selected_models()
                with st.spinner("正在生成挑战密码……"):
                    password = generate_challenge_password(
                        generator_id,
                        runtime[generator_id],
                        max_length=max_length,
                        temperature=temperature,
                        seed=int(seed),
                    )
                render_scores(password, f"由 {labels[generator_id]} 生成")
            except (FileNotFoundError, TypeError, ValueError) as error:
                st.error(str(error))

elif area == "Character Lab":
    character_mode = st.segmented_control(
        "Character Experiment",
        [
            "Surprisal Journey",
            "Probability Keyboard",
            "Temperature Laboratory",
            "Model Disagreement",
        ],
        default="Surprisal Journey",
        required=True,
        key="playground_character_mode",
        width="stretch",
        persist_state="session",
    )
    typed_password = character_input(
        "Demo password",
        key="character_lab_input",
        max_length=12,
    )
    st.caption(
        "每个已输入字符保留模型在输入前为它分配的概率。红色表示低概率，绿色表示高概率。"
    )
    try:
        if character_mode == "Surprisal Journey":
            runtime = load_selected_models()
            with st.spinner("正在追踪逐字符 Surprisal……"):
                traces = trace_character_probabilities(typed_password, runtime, top_k=5)
            for trace in traces:
                with st.container(border=True):
                    st.subheader(labels[trace.model_id], anchor=False)
                    st.html(
                        colored_password_html(
                            [(item.character, item.probability) for item in trace.characters]
                        )
                    )
            if typed_password:
                show_chart(plot_surprisal_journey(traces, labels, colors))
                st.caption(
                    "Step Surprisal 为 −log₂ p(character | prefix)；Cumulative Surprisal "
                    "是沿输入前缀逐步累加的结果；Surprisal per Token 再除以当前前缀长度。"
                    "只有进行完整密码评分时才计入 EOS。"
                )
            else:
                st.info("输入字符后即可观察每一步和累计惊讶度。")

        elif character_mode == "Probability Keyboard":
            model_id = st.selectbox(
                "Model",
                [record.id for record in records],
                format_func=labels.__getitem__,
                key="probability_keyboard_model",
            )
            runtime = {model_id: load_one_model(model_id)}
            with st.spinner("正在更新 Next-token 分布……"):
                trace = trace_character_probabilities(
                    typed_password, runtime, top_k=8
                )[0]
            with st.container(border=True):
                st.subheader(labels[trace.model_id], anchor=False)
                st.html(
                    colored_password_html(
                        [(item.character, item.probability) for item in trace.characters]
                    )
                )
                show_chart(plot_probability_keyboard(trace, labels[trace.model_id]))
                st.dataframe(
                    [
                        {"Token": item.token, "Probability": item.probability}
                        for item in trace.next_tokens
                    ],
                    column_config={
                        "Probability": st.column_config.NumberColumn(format="percent")
                    },
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "Probability Keyboard 的颜色按当前模型中概率最高的 token 归一化；"
                "悬停可查看绝对概率。"
            )

        elif character_mode == "Temperature Laboratory":
            model_id = st.selectbox(
                "Model",
                [record.id for record in records],
                format_func=labels.__getitem__,
                key="temperature_model",
            )
            runtime = {model_id: load_one_model(model_id)}
            temperatures = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
            with st.spinner("正在比较不同 Temperature……"):
                traces = {
                    temperature: trace_character_probabilities(
                        typed_password,
                        runtime,
                        top_k=8,
                        temperature=temperature,
                    )[0]
                    for temperature in temperatures
                }
            show_chart(plot_temperature_laboratory(traces))
            st.caption(
                "Temperature 在 softmax 前缩放 logits。较低取值使分布更集中；"
                "较高取值使分布更平坦，通常也会提高 Entropy。"
            )

        elif character_mode == "Model Disagreement":
            runtime = load_selected_models()
            if len(runtime) < 2:
                st.info("至少在 Warehouse 选择两个模型才能比较分歧。")
            else:
                with st.spinner("正在比较模型概率分布……"):
                    traces = trace_character_probabilities(
                        typed_password, runtime, top_k=8
                    )
                matrix = model_disagreement_matrix(traces)
                upper = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
                entropies = np.asarray(
                    [distribution_entropy(trace) for trace in traces], dtype=float
                )
                mean_card, maximum_card, entropy_card = st.columns(3)
                mean_card.metric(
                    "Mean Pairwise JSD",
                    f"{float(np.mean(upper)):.4f} bits",
                    help=(
                        "所有不同模型对的 Jensen–Shannon divergence 算术平均。\n\n"
                        "0 表示下一 token 分布完全一致；值越大表示整体分歧越强。"
                    ),
                    border=True,
                )
                maximum_card.metric(
                    "Maximum Pairwise JSD",
                    f"{float(np.max(upper)):.4f} bits",
                    help=(
                        "当前前缀下分歧最大的一对模型的 JSD。\n\n"
                        "它用于发现平均值可能掩盖的局部架构冲突。"
                    ),
                    border=True,
                )
                entropy_card.metric(
                    "Entropy Range",
                    f"{float(np.ptp(entropies)):.4f} bits",
                    help=(
                        "所选模型下一 token Shannon entropy 的最大值减最小值。\n\n"
                        "反映模型对下一步预测不确定程度的跨度。"
                    ),
                    border=True,
                )
                show_chart(plot_model_disagreement(traces, labels))
                st.caption(
                    "此处 Jensen–Shannon divergence 具有对称性，取值限制在 0–1 bit。"
                    "数值越大，表示模型对 Next token 的分歧越强。"
                )
    except (FileNotFoundError, TypeError, ValueError) as error:
        st.error(str(error))

elif area == "Generation Lab":
    generation_mode = st.segmented_control(
        "Generation Mode",
        ["Random Sampling", "Beam Completion"],
        default="Random Sampling",
        required=True,
        key="playground_generation_mode",
        width="stretch",
        persist_state="session",
    )
    if generation_mode == "Random Sampling":
        with st.form("random-generation"):
            sample_column, length_column, temperature_column, seed_column = st.columns(4)
            num_samples = sample_column.slider("Samples per model", 1, 20, 8)
            max_length = length_column.slider("Maximum length", 1, 12, 12)
            temperature = temperature_column.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
            seed = seed_column.number_input("Seed", min_value=0, value=2026, step=1)
            submitted = st.form_submit_button(
                "Generate", type="primary", icon=":material/casino:"
            )
        if submitted:
            try:
                with st.spinner("正在使用所选模型生成……"):
                    generated = generate_with_models(
                        load_selected_models(),
                        num_samples=num_samples,
                        max_length=max_length,
                        temperature=temperature,
                        seed=int(seed),
                    )
                for model_id, passwords in generated.items():
                    with st.container(border=True):
                        st.subheader(labels[model_id], anchor=False)
                        st.dataframe(
                            [
                                {"Password": password, "Length": len(password)}
                                for password in passwords
                            ],
                            hide_index=True,
                            width="stretch",
                        )
            except (FileNotFoundError, TypeError, ValueError) as error:
                st.error(str(error))

    elif generation_mode == "Beam Completion":
        with st.form("prefix-completion"):
            prefix = st.text_input("Prefix", max_chars=12)
            beam_column, length_column, temperature_column = st.columns(3)
            beam_width = beam_column.slider("Beam width", 1, 20, 5)
            max_length = length_column.slider(
                "Maximum length", 1, 12, 12, key="beam_length"
            )
            temperature = temperature_column.slider(
                "Temperature", 0.1, 2.0, 1.0, 0.1, key="beam_temperature"
            )
            submitted = st.form_submit_button(
                "Complete prefix",
                type="primary",
                icon=":material/auto_awesome:",
            )
        if submitted:
            try:
                with st.spinner("正在搜索补全结果……"):
                    completions = complete_with_models(
                        load_selected_models(),
                        prefix=prefix,
                        beam_width=beam_width,
                        max_length=max_length,
                        temperature=temperature,
                    )
                show_chart(plot_completion_consensus(completions, labels))
                st.dataframe(
                    completion_consensus_summary(completions, labels),
                    column_config={
                        "Support Rate": st.column_config.NumberColumn(
                            format="percent",
                            help="返回该候选的模型数除以全部参与模型数。",
                        ),
                        "Consensus Score": st.column_config.NumberColumn(
                            format="%.3f",
                            help=(
                                "各模型 reciprocal rank 的和除以模型数；未返回候选的模型贡献 0。"
                            ),
                        ),
                        "Mean Log Probability": st.column_config.NumberColumn(
                            format="%.4f",
                            help="仅在返回该候选的模型之间平均；不同模型的绝对值需谨慎比较。",
                        ),
                    },
                    hide_index=True,
                    width="stretch",
                )
                for model_id, candidates in completions.items():
                    with st.container(border=True):
                        st.subheader(labels[model_id], anchor=False)
                        st.dataframe(
                            [
                                {
                                    "Candidate": candidate.text,
                                    "Log Probability": candidate.log_probability,
                                }
                                for candidate in candidates
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                st.caption(
                    "Consensus 在相同 Beam Search 设置下使用 reciprocal rank。候选会同时"
                    "因更多模型支持和更高排名而获得分数；图表只显示前 25 个候选，表格"
                    "保留全部结果。"
                )
            except (FileNotFoundError, TypeError, ValueError) as error:
                st.error(str(error))
