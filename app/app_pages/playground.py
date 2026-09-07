"""Playground：多模型实时评分、生成与前缀补全。"""

import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import render_current_selection, selected_records
from app.frontend.probability_input import (
    character_input,
    colored_password_html,
)
from app.frontend.playground import (
    aggregate_scores,
    complete_with_models,
    generate_with_models,
    score_models,
    trace_character_probabilities,
)
from app.frontend.warehouse import get_runtime_model


catalog = load_catalog()
records = selected_records(catalog)
st.title("Playground", anchor=False)
render_current_selection(records)
st.warning(
    "请只输入演示文本，不要输入正在使用的真实密码。输入不会由应用主动保存。",
    icon=":material/warning:",
)
if not records:
    st.stop()


def load_selected_models():
    """保持 Warehouse 选择顺序加载模型，并让资源缓存复用权重。"""

    return {record.id: get_runtime_model(record) for record in records}


labels = {record.id: record.display_name for record in records}
tool = st.segmented_control(
    "Tool",
    [
        "Password Scoring",
        "Random Generation",
        "Prefix Completion",
        "Character Guidance",
    ],
    default="Password Scoring",
    required=True,
    key="playground_tool",
    width="stretch",
    persist_state="session",
)

# 四种操作都可能加载多个模型；单视图避免无关工具同时执行昂贵推理。
if tool == "Password Scoring":
    with st.form("password-scoring"):
        password = st.text_input(
            "Demo password",
            type="password",
            max_chars=12,
            help="接受 1–12 个可打印 ASCII 字符。",
        )
        score_submitted = st.form_submit_button(
            "Score with selected models",
            type="primary",
            icon=":material/query_stats:",
        )
    if score_submitted:
        try:
            with st.spinner("Loading models and scoring…"):
                scores = score_models(password, load_selected_models())
                aggregate = aggregate_scores(scores)
            mean_a, mean_b, spread = st.columns(3)
            mean_a.metric("Mean Surprisal", f"{aggregate.mean_surprisal_bits:.3f} bits")
            mean_b.metric("Mean Bits / Token", f"{aggregate.mean_bits_per_token:.3f}")
            spread.metric("Model Std. Dev.", f"{aggregate.standard_deviation:.3f}")
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
                "Mean Surprisal is the arithmetic mean in bit space, equivalent to using "
                "the geometric mean of model probabilities."
            )
        except (FileNotFoundError, TypeError, ValueError) as error:
            st.error(str(error))

elif tool == "Random Generation":
    with st.form("random-generation"):
        sample_column, length_column, temperature_column, seed_column = st.columns(4)
        num_samples = sample_column.slider("Samples per model", 1, 20, 8)
        max_length = length_column.slider("Maximum length", 1, 12, 12)
        temperature = temperature_column.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        seed = seed_column.number_input("Seed", min_value=0, value=2026, step=1)
        generation_submitted = st.form_submit_button(
            "Generate", type="primary", icon=":material/casino:"
        )
    if generation_submitted:
        try:
            with st.spinner("Generating with selected models…"):
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

elif tool == "Prefix Completion":
    with st.form("prefix-completion"):
        prefix = st.text_input("Prefix", max_chars=12)
        beam_column, length_column, temperature_column = st.columns(3)
        beam_width = beam_column.slider("Beam width", 1, 20, 5)
        completion_length = length_column.slider(
            "Maximum length", 1, 12, 12, key="beam_length"
        )
        completion_temperature = temperature_column.slider(
            "Temperature", 0.1, 2.0, 1.0, 0.1, key="beam_temperature"
        )
        completion_submitted = st.form_submit_button(
            "Complete prefix",
            type="primary",
            icon=":material/auto_awesome:",
        )
    if completion_submitted:
        try:
            with st.spinner("Searching completions…"):
                completions = complete_with_models(
                    load_selected_models(),
                    prefix=prefix,
                    beam_width=beam_width,
                    max_length=completion_length,
                    temperature=completion_temperature,
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
        except (FileNotFoundError, TypeError, ValueError) as error:
            st.error(str(error))

elif tool == "Character Guidance":
    typed_password = character_input(
        "Demo password",
        key="character_guidance_input",
        max_length=12,
    )
    st.caption(
        "Each character is colored by its conditional probability when entered. "
        "Red = 0%, yellow = 50%, green = 100%; hover a character for its exact value."
    )

    # 输入组件先出现，再加载缓存模型并逐步推进状态，减少键入时的等待感。
    try:
        with st.spinner("Updating character probabilities…"):
            traces = trace_character_probabilities(
                typed_password,
                load_selected_models(),
                top_k=5,
                max_length=12,
            )
        for trace in traces:
            with st.container(border=True):
                st.subheader(labels[trace.model_id], anchor=False)
                history_column, prediction_column = st.columns([1.1, 1])
                with history_column:
                    st.markdown("**Typed character probabilities**")
                    st.html(
                        colored_password_html(
                            [
                                (item.character, item.probability)
                                for item in trace.characters
                            ]
                        )
                    )
                with prediction_column:
                    st.markdown("**Most likely next tokens**")
                    st.dataframe(
                        [
                            {
                                "Token": prediction.token,
                                "Probability": prediction.probability,
                            }
                            for prediction in trace.next_tokens
                        ],
                        column_config={
                            "Probability": st.column_config.NumberColumn(
                                format="percent"
                            )
                        },
                        hide_index=True,
                        width="stretch",
                    )
    except (FileNotFoundError, TypeError, ValueError) as error:
        st.error(str(error))
