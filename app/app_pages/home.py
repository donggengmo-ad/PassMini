"""Home：应用概览与三层入口说明。"""

import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import selected_records


catalog = load_catalog()
records = selected_records(catalog)

st.title("PassMini", anchor=False)
st.write("密码建模实验")

with st.container(horizontal=True):
    st.metric(
        "Available Models",
        len(catalog.enabled_models),
        icon=":material/check_circle:",
        border=True,
    )
    st.metric(
        "Selected Models",
        len(records),
        icon=":material/checklist:",
        border=True,
    )
    st.metric(
        "Model Families",
        len({record.model_type for record in catalog.enabled_models}),
        icon=":material/schema:",
        border=True,
    )

st.subheader("Workspace", anchor=False)
warehouse, library, playground = st.columns(3, border=True)
with warehouse:
    st.subheader("Warehouse", anchor=False)
    st.write("查看并装备模型")
with library:
    st.subheader("Library", anchor=False)
    st.write("评估模型")
with playground:
    st.subheader("Playground", anchor=False)
    st.write("模型交互游戏")

st.info(
    "不建议输入正在使用的真实密码。",
    icon=":material/security:",
)
