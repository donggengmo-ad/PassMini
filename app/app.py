"""PassMini Streamlit 应用入口与页面导航。"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit 以 app 目录中的脚本启动，需要显式加入仓库根目录以复用 scripts。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.frontend.catalog import load_catalog
from app.frontend.components import initialize_selection


st.set_page_config(
    page_title="PassMini",
    page_icon=":material/password:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

initialize_selection(load_catalog())

page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="Home",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "app_pages/warehouse.py",
            title="Warehouse",
            icon=":material/warehouse:",
        ),
        st.Page(
            "app_pages/library.py",
            title="Library",
            icon=":material/library_books:",
        ),
        st.Page(
            "app_pages/playground.py",
            title="Playground",
            icon=":material/experiment:",
        ),
    ],
    position="top",
)
page.run()
