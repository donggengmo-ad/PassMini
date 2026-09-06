"""Home：应用概览与三层入口说明。"""

import streamlit as st

from app.frontend.catalog import load_catalog
from app.frontend.components import selected_records


catalog = load_catalog()
records = selected_records(catalog)

st.title("PassMini", anchor=False)
st.write("字符级密码模型的训练结果、概率评测与小规模交互实验。")

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
        "Planned Models",
        len(catalog.models) - len(catalog.enabled_models),
        icon=":material/schedule:",
        border=True,
    )

st.subheader("Workspace", anchor=False)
warehouse, library, playground = st.columns(3, border=True)
with warehouse:
    st.subheader("Warehouse", anchor=False)
    st.write("选择模型，检查配置、训练摘要和部署文件。")
with library:
    st.subheader("Library", anchor=False)
    st.write("按当前模型集合比较训练历史、surprisal 和覆盖率。")
with playground:
    st.subheader("Playground", anchor=False)
    st.write("进行多模型密码评分、随机生成和前缀补全。")

st.info(
    "本应用展示模型分布下的相对结果，不估计真实破解时间。"
    "请勿输入正在使用的真实密码。",
    icon=":material/security:",
)
