from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ENTRY = PROJECT_ROOT / "app" / "app.py"


@pytest.mark.parametrize(
    "page",
    [
        None,
        "app_pages/warehouse.py",
        "app_pages/library.py",
        "app_pages/playground.py",
    ],
)
def test_streamlit_page_smoke(page):
    app = AppTest.from_file(APP_ENTRY, default_timeout=30).run()
    if page is not None:
        app = app.switch_page(page).run()

    assert not app.exception


def test_model_selection_persists_between_pages():
    app = AppTest.from_file(APP_ENTRY, default_timeout=30).run()
    app = app.switch_page("app_pages/warehouse.py").run()
    app.multiselect[0].set_value(["low-gru", "low-tcn"])
    app = app.run()

    app = app.switch_page("app_pages/library.py").run()

    assert app.session_state["selected_model_ids"] == ["low-gru", "low-tcn"]
    assert not app.exception


def test_character_guidance_page_smoke():
    # CCv2 注册表属于单次 AppTest 上下文；强制在本次上下文重新导入组件。
    sys.modules.pop("app.frontend.probability_input", None)
    app = AppTest.from_file(APP_ENTRY, default_timeout=30).run()
    app = app.switch_page("app_pages/playground.py").run()
    app.get("button_group")[0].set_value("Character Guidance")
    app = app.run()

    assert len(app.get("bidi_component")) == 1
    assert len(app.dataframe) == 4
    assert not app.exception
