"""Warehouse：模型选择与元信息页面。"""

import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import (
    MODEL_SELECTION_PRESETS,
    SELECTION_KEY,
    preset_model_ids,
    render_model_selector,
)
from app.frontend.warehouse import read_model_metadata


catalog = load_catalog()
st.title("Warehouse", anchor=False)
st.write("选择任意多个已完成模型。这个选择会直接传递给 Library 和 Playground。")

with st.expander("Selection presets"):
    st.caption("可按同一档位或同一模型架构快速选择；应用后仍可在 Models 中手动微调。")
    with st.form("warehouse-selection-preset"):
        preset_column, baseline_column = st.columns([2, 1])
        preset = preset_column.selectbox(
            "Preset",
            MODEL_SELECTION_PRESETS,
            key="warehouse_selection_preset",
        )
        include_baseline = baseline_column.checkbox(
            "Include baseline",
            value=False,
            key="warehouse_include_baseline",
            help="勾选后会把 Bigram · Baseline 一并加入当前预设。",
        )
        apply_preset = st.form_submit_button(
            "Apply preset",
            type="primary",
            icon=":material/checklist:",
            key="warehouse_apply_preset",
        )
    if apply_preset:
        # Models 控件尚未在本轮创建，此时写入共享状态可安全更新其初始选择。
        st.session_state[SELECTION_KEY] = preset_model_ids(
            catalog,
            preset,
            include_baseline,
        )

selected = render_model_selector(catalog)
if selected:
    st.success(f"已选择 {len(selected)} 个模型。", icon=":material/check:")
else:
    st.warning("当前模型集合为空；Library 和 Playground 将等待选择。")

st.subheader("Available models", anchor=False)
for record in catalog.enabled_models:
    metadata = read_model_metadata(record)
    with st.container(border=True):
        title, parameters, compute, training, status = st.columns([2.2, 1, 1, 1, 1])
        title.subheader(record.display_name, anchor=False)
        title.caption(f"Model ID: {record.id}")
        parameters.metric(
            "Parameters",
            f"{record.parameter_count:,}",
            help="可训练参数总数；Bigram 是非参数化计数基线，因此记为 0。",
        )
        compute.metric(
            "Estimated FLOPs",
            f"{record.flops:,}",
            help=(
                "输入形状 [1, 13] 时生成完整 logits 的解析式 FLOPs 估计。\n\n"
                "包含矩阵乘法、卷积、门控、因果注意力、归一化和主要逐元素运算；"
                "用于模型间统一比较，不等同于特定硬件的实际指令数。"
            ),
        )
        training.metric("Epochs", metadata.epochs if metadata.epochs is not None else "—")
        status.metric("Runtime", "Ready" if record.inference_ready else "Missing")

        if metadata.best_validation_loss is not None:
            st.caption(f"Best validation loss: {metadata.best_validation_loss:.4f}")
        ready_files = sum(metadata.files.values())
        st.progress(
            ready_files / len(metadata.files),
            text=f"Artifacts: {ready_files}/{len(metadata.files)}",
        )
        with st.expander("Configuration and artifact status"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Inference configuration**")
                st.json(metadata.inference or {"status": "Unavailable"})
            with right:
                st.markdown("**Files**")
                st.dataframe(
                    [
                        {"Artifact": name, "Available": available}
                        for name, available in metadata.files.items()
                    ],
                    hide_index=True,
                    width="stretch",
                )

planned = [record for record in catalog.models if not record.enabled]
if planned:
    st.subheader("Planned models", anchor=False)
    st.caption("这些档位已规划参数，但尚未登记为可推理 artifact。")
    st.dataframe(
        [
            {
                "Model": record.display_name,
                "Model ID": record.id,
                "Parameters": record.parameter_count,
                "Estimated FLOPs": record.flops,
                "Status": "Planned",
            }
            for record in planned
        ],
        column_config={
            "Parameters": st.column_config.NumberColumn(
                format="localized", help="规划模型的可训练参数量。"
            ),
            "Estimated FLOPs": st.column_config.NumberColumn(
                format="localized",
                help="输入 [1, 13] 时生成完整 logits 的解析式 FLOPs 估计。",
            ),
        },
        hide_index=True,
        width="stretch",
    )
