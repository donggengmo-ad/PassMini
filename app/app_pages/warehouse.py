"""Warehouse：模型选择与元信息页面。"""

import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import (
    MODEL_SELECTION_PRESETS,
    SELECTION_KEY,
    preset_model_ids,
    render_model_selector,
)
from app.frontend.model_icons import render_model_icon
from app.frontend.warehouse import read_model_metadata


FAMILY_DESCRIPTIONS = {
    "bigram": "基于相邻 token 转移计数的统计概率基线。",
    "mlp": "用固定长度上下文和全连接层实现有限阶 Markov 建模。",
    "tcn": "用共享的扩张因果卷积并行提取局部序列模式。",
    "gru": "通过门控隐藏状态逐步压缩并传递历史信息。",
    "transformer": "通过 Q、K、V 自注意力组合历史 token 信息。",
}


def model_summary_rows(records, selected_ids: set[str]) -> list[dict]:
    """整理一个模型家族的档位、规模和部署状态。"""

    rows = []
    for record in records:
        metadata = read_model_metadata(record)
        rows.append(
            {
                "Selected": record.id in selected_ids,
                "Tier": record.tier.title(),
                "Parameters": record.parameter_count,
                "Estimated FLOPs": record.flops,
                "Epochs": metadata.epochs,
                "Runtime": "Ready" if record.inference_ready else "Missing",
                "Artifacts": f"{sum(metadata.files.values())}/{len(metadata.files)}",
            }
        )
    return rows


def render_family_summary(records, selected_ids: set[str]) -> None:
    """紧凑展示一个家族的各档位摘要。"""

    st.dataframe(
        model_summary_rows(records, selected_ids),
        column_config={
            "Selected": st.column_config.CheckboxColumn(
                help="该模型是否已在顶部 Models 中装备。"
            ),
            "Parameters": st.column_config.NumberColumn(
                format="localized",
                help="模型的可训练参数量；Bigram 是计数基线，因此记为 0。",
            ),
            "Estimated FLOPs": st.column_config.NumberColumn(
                format="localized",
                help="输入 [1, 13] 时生成完整 logits 的解析式 FLOPs 估计。",
            ),
        },
        hide_index=True,
        width="stretch",
    )


def render_family_details(records) -> None:
    """按档位保留配置、验证损失和 artifact 明细。"""

    for record in records:
        metadata = read_model_metadata(record)
        with st.expander(f"{record.tier.title()} details"):
            st.caption(f"Model ID: {record.id}")
            if metadata.best_validation_loss is not None:
                st.caption(
                    f"Best validation loss: {metadata.best_validation_loss:.4f}"
                )
            left, right = st.columns(2)
            with left:
                st.markdown("**Inference configuration**")
                st.json(metadata.inference or {"status": "Unavailable"})
            with right:
                st.markdown("**Artifacts**")
                st.dataframe(
                    [
                        {"Artifact": name, "Available": available}
                        for name, available in metadata.files.items()
                    ],
                    hide_index=True,
                    width="stretch",
                )


def render_family_card(model_type: str, records, selected_ids: set[str]) -> None:
    """渲染家族标题、结构动画、档位摘要和详细信息。"""

    family_name = model_type.upper() if model_type != "transformer" else "Transformer"
    st.subheader(family_name, anchor=False)
    st.caption(FAMILY_DESCRIPTIONS[model_type])
    render_model_icon(model_type)
    render_family_summary(records, selected_ids)
    render_family_details(records)


catalog = load_catalog()
st.title("Warehouse", anchor=False)
st.write("选择任意多个已完成模型，这个选择会直接传递给 Library 和 Playground。")

with st.expander("Selection presets"):
    st.caption("可按同一档位或同一模型架构快速选择；应用后仍可在 Models 中手动调。")
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

st.subheader("Model families", anchor=False)
selected_ids = {record.id for record in selected}
families = {
    model_type: [
        record for record in catalog.enabled_models if record.model_type == model_type
    ]
    for model_type in FAMILY_DESCRIPTIONS
}

# Bigram 只有一个 baseline 档位，单独使用横向布局避免与三档神经模型强行对齐。
with st.container(border=True):
    icon_column, summary_column = st.columns([1, 1.7], vertical_alignment="center")
    with icon_column:
        st.subheader("Bigram", anchor=False)
        st.caption(FAMILY_DESCRIPTIONS["bigram"])
        render_model_icon("bigram")
    with summary_column:
        render_family_summary(families["bigram"], selected_ids)
        render_family_details(families["bigram"])

# 四个神经模型家族使用 2×2 卡片；窄屏时 Streamlit 会自动纵向堆叠。
for left_type, right_type in (("mlp", "tcn"), ("gru", "transformer")):
    left_column, right_column = st.columns(2)
    for column, model_type in (
        (left_column, left_type),
        (right_column, right_type),
    ):
        with column:
            with st.container(border=True):
                render_family_card(model_type, families[model_type], selected_ids)

planned = [record for record in catalog.models if not record.enabled]
if planned:
    st.subheader("Planned models", anchor=False)
    st.caption("这些档位已规划参数，但尚未登记为可推理模型。")
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
