"""多个页面共享的 Streamlit 展示组件。"""

from __future__ import annotations

import streamlit as st

from .catalog import ModelCatalog, ModelRecord


SELECTION_KEY = "selected_model_ids"


def initialize_selection(catalog: ModelCatalog) -> None:
    """首次访问时默认选择全部可用模型，并清除已经失效的 ID。"""

    enabled_ids = [model.id for model in catalog.enabled_models]
    if SELECTION_KEY not in st.session_state:
        st.session_state[SELECTION_KEY] = enabled_ids
        return

    allowed = set(enabled_ids)
    selected_ids = st.session_state[SELECTION_KEY]
    filtered_ids = [
        model_id for model_id in selected_ids if model_id in allowed
    ]
    # Widget 建立后不能改写同名状态；只有 catalog 变化时才需要修正。
    if filtered_ids != selected_ids:
        st.session_state[SELECTION_KEY] = filtered_ids


def selected_records(catalog: ModelCatalog) -> list[ModelRecord]:
    """按选择顺序返回当前模型记录。"""

    initialize_selection(catalog)
    return catalog.select(st.session_state[SELECTION_KEY])


def render_model_selector(catalog: ModelCatalog) -> list[ModelRecord]:
    """渲染跨页面持久化的多模型选择器。"""

    initialize_selection(catalog)
    labels = {model.id: model.display_name for model in catalog.enabled_models}
    st.multiselect(
        "Models",
        options=list(labels),
        format_func=labels.__getitem__,
        key=SELECTION_KEY,
        help="这里的选择会在 Warehouse、Library 和 Playground 之间共享。",
        persist_state="session",
    )
    return selected_records(catalog)


def render_current_selection(records: list[ModelRecord]) -> None:
    """在非 Warehouse 页面显示当前模型集合。"""

    if not records:
        st.warning("尚未选择模型。请先在 Warehouse 中选择至少一个可用模型。")
        return
    labels = " · ".join(record.display_name for record in records)
    st.caption(f"Current models: {labels}")


def model_color_map(records: list[ModelRecord]) -> dict[str, str]:
    """返回模型 ID 到目录颜色的映射。"""

    return {record.id: record.color for record in records}
