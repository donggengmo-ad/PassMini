"""Warehouse：模型选择与元信息页面。"""

import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import render_model_selector
from app.frontend.warehouse import read_model_metadata


catalog = load_catalog()
st.title("Warehouse", anchor=False)
st.write("选择任意多个已完成模型。这个选择会直接传递给 Library 和 Playground。")

selected = render_model_selector(catalog)
if selected:
    st.success(f"已选择 {len(selected)} 个模型。", icon=":material/check:")
else:
    st.warning("当前模型集合为空；Library 和 Playground 将等待选择。")

st.subheader("Available models", anchor=False)
for record in catalog.enabled_models:
    metadata = read_model_metadata(record)
    with st.container(border=True):
        title, parameters, training, status = st.columns([2.2, 1, 1, 1])
        title.subheader(record.display_name, anchor=False)
        title.caption(f"Model ID: {record.id}")
        parameters.metric("Parameters", f"{record.parameter_count:,}")
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
                "Status": "Planned",
            }
            for record in planned
        ],
        hide_index=True,
        width="stretch",
    )
