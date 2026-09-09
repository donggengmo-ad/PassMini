from pathlib import Path
import json
import sys

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ENTRY = PROJECT_ROOT / "app" / "app.py"


def create_app_test() -> AppTest:
    """为每个 AppTest 上下文重新注册页面使用的 CCv2 组件。"""

    sys.modules.pop("app.frontend.model_icons", None)
    return AppTest.from_file(APP_ENTRY, default_timeout=30).run()


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
    app = create_app_test()
    if page is not None:
        app = app.switch_page(page).run()

    assert not app.exception


def test_model_selection_persists_between_pages():
    app = create_app_test()
    app = app.switch_page("app_pages/warehouse.py").run()
    app.multiselect[0].set_value(["low-gru", "low-tcn"])
    app = app.run()

    app = app.switch_page("app_pages/library.py").run()

    assert app.session_state["selected_model_ids"] == ["low-gru", "low-tcn"]
    assert not app.exception


def test_warehouse_selection_preset_updates_shared_model_selection():
    app = create_app_test()
    app = app.switch_page("app_pages/warehouse.py").run()
    app.selectbox(key="warehouse_selection_preset").select("TCN family")
    app.checkbox(key="warehouse_include_baseline").check()
    app.button(key="warehouse_apply_preset").click()
    app = app.run()

    assert app.session_state["selected_model_ids"] == [
        "baseline-bigram",
        "low-tcn",
        "medium-tcn",
        "high-tcn",
    ]
    assert not app.exception


def test_warehouse_groups_models_into_five_animated_family_cards():
    app = create_app_test()
    app = app.switch_page("app_pages/warehouse.py").run()

    assert [item.value for item in app.subheader] == [
        "Model families",
        "Bigram",
        "MLP",
        "TCN",
        "GRU",
        "Transformer",
    ]
    assert len(app.get("bidi_component")) == 5
    summaries = [
        frame.value
        for frame in app.dataframe
        if "Selected" in frame.value.columns
    ]
    assert [len(frame) for frame in summaries] == [1, 3, 3, 3, 3]
    assert not app.exception


def test_library_surprisal_supports_summary_without_generation_quality():
    app = create_app_test()
    app.session_state["selected_model_ids"] = ["medium-tcn"]
    app = app.switch_page("app_pages/library.py").run()
    app.get("button_group")[0].set_value("Surprisal")
    app = app.run()

    assert not app.exception
    assert len(app.metric) == 1
    assert len(app.get("vega_lite_chart")) == 2


def test_library_random_coverage_page_reads_packaged_npz_files():
    app = create_app_test()
    app = app.switch_page("app_pages/library.py").run()
    app.get("button_group")[0].set_value("Coverage")
    app = app.run()
    app.radio[0].set_value("Random sampling")
    app = app.run()

    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 2


@pytest.mark.parametrize(
    ("view", "expected_charts"),
    [
        ("Training", 4),
        ("Surprisal", 2),
        ("Pairwise Comparison", 2),
        ("Coverage", 2),
        ("Generation Quality", 1),
    ],
)
def test_library_views_render_without_invalid_generic_legend_binding(view, expected_charts):
    app = create_app_test()
    app.session_state["selected_model_ids"] = ["low-gru", "low-tcn"]
    app = app.switch_page("app_pages/library.py").run()
    app.get("button_group")[0].set_value(view)
    app = app.run()

    charts = app.get("vega_lite_chart")
    assert not app.exception
    assert len(charts) == expected_charts
    for chart in charts:
        spec = json.loads(chart.proto.spec)
        assert not any(
            parameter.get("bind") == "legend"
            and parameter.get("select", {}).get("fields") == ["Model"]
            for parameter in spec.get("params", [])
        )


def test_library_surprisal_displays_every_selected_model_metric():
    app = create_app_test().switch_page("app_pages/library.py").run()
    app.get("button_group")[0].set_value("Surprisal")
    app.run()
    assert not app.exception
    assert len(app.metric) == len(app.session_state["selected_model_ids"])


def test_model_zoo_handles_missing_coverage_without_page_exception(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("missing coverage")
    monkeypatch.setattr("app.frontend.library.load_coverage_data", unavailable)
    app = create_app_test().switch_page("app_pages/library.py").run()
    app.get("button_group")[1].set_value("Random Coverage")
    app.run()
    assert not app.exception
    assert any("Random Coverage" in item.value for item in app.info)


def test_character_probability_keyboard_page_smoke():
    # CCv2 注册表属于单次 AppTest 上下文；强制在本次上下文重新导入组件。
    sys.modules.pop("app.frontend.probability_input", None)
    app = create_app_test()
    app = app.switch_page("app_pages/playground.py").run()
    app.get("button_group")[0].set_value("Character Lab")
    app = app.run()
    app.get("button_group")[1].set_value("Probability Keyboard")
    app = app.run()

    assert len(app.get("bidi_component")) == 1
    assert len(app.selectbox) == 1
    assert len(app.dataframe) == 1
    assert len(app.get("vega_lite_chart")) == 1
    assert not app.exception


@pytest.mark.parametrize("mode", ["Surprisal Journey", "Temperature Laboratory", "Model Disagreement"])
def test_character_subpages_render_nonempty_input(mode, monkeypatch):
    import app.frontend.probability_input as inputs
    monkeypatch.setattr(inputs, "character_input", lambda *args, **kwargs: "abc123")
    app = create_app_test()
    app.session_state["selected_model_ids"] = ["low-gru", "low-tcn"]
    # 旧版本会把这份子选择当作实时模型集合；现在应忽略它并使用全部装备模型。
    app.session_state["character_models"] = ["low-gru"]
    app = app.switch_page("app_pages/playground.py").run()
    app.get("button_group")[0].set_value("Character Lab")
    app.run()
    app.get("button_group")[1].set_value(mode)
    app.run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) >= 1
    assert all(widget.label != "Character models" for widget in app.multiselect)
    if mode == "Surprisal Journey":
        # 拆成独立图时，子图必须携带原先继承的数据，否则浏览器只会画空坐标轴。
        import json
        import pyarrow as pa
        charts = app.get("vega_lite_chart")
        assert len(charts) == 4
        for chart in charts:
            spec = json.loads(chart.proto.spec)
            dataset = next(d for d in chart.proto.datasets if d.name == spec["data"]["name"])
            frame = pa.ipc.open_stream(dataset.data.data).read_all().to_pandas()
            assert len(frame) == 12
            assert set(frame["Model"]) == {"GRU · Low", "TCN · Low"}


def test_scoring_form_produces_results():
    app = create_app_test()
    app.session_state["selected_model_ids"] = ["low-gru", "baseline-bigram"]
    app = app.switch_page("app_pages/playground.py").run()
    app.text_input[0].set_value("abc123")
    app.button[0].click()
    app.run()
    assert not app.exception
    assert len(app.metric) == 3
    assert len(app.get("vega_lite_chart")) == 1


def test_scoring_result_survives_navigation_without_rescoring(monkeypatch):
    from unittest.mock import Mock
    import app.frontend.playground as playground
    spy = Mock(wraps=playground.score_models)
    monkeypatch.setattr(playground, "score_models", spy)
    app = create_app_test()
    app.session_state["selected_model_ids"] = ["low-gru"]
    app = app.switch_page("app_pages/playground.py").run()
    app.text_input[0].set_value("abc123")
    app.button[0].click().run()
    assert spy.call_count == 1
    values = [metric.value for metric in app.metric]
    app.switch_page("app_pages/library.py").run()
    app.switch_page("app_pages/playground.py").run()
    assert not app.exception
    assert spy.call_count == 1
    assert [metric.value for metric in app.metric] == values
    app.button(key="clear_playground_results").click().run()
    assert not app.exception
    assert not app.metric
    assert app.text_input[0].value == ""


@pytest.mark.parametrize("mode", ["Random Sampling", "Beam Completion"])
def test_generation_result_survives_subpage_switch(mode, monkeypatch):
    from unittest.mock import Mock
    import app.frontend.playground as playground
    from scripts.inference import GenerationCandidate
    function = "generate_with_models" if mode == "Random Sampling" else "complete_with_models"
    payload = ["ab"] if mode == "Random Sampling" else [GenerationCandidate("ab", [4, 5], -1.0)]
    spy = Mock(return_value={"low-gru": payload})
    monkeypatch.setattr(playground, function, spy)
    app = create_app_test()
    app.session_state["selected_model_ids"] = ["low-gru"]
    app.switch_page("app_pages/playground.py").run()
    app.get("button_group")[0].set_value("Generation Lab").run()
    app.get("button_group")[1].set_value(mode).run()
    if mode == "Beam Completion":
        app.text_input[0].set_value("a")
    app.button[0].click().run()
    assert not app.exception
    assert spy.call_count == 1
    app.get("button_group")[0].set_value("Score Arena").run()
    app.get("button_group")[0].set_value("Generation Lab").run()
    assert not app.exception
    assert app.dataframe
    assert spy.call_count == 1
    app.button(key="clear_playground_results").click().run()
    assert not app.exception
    assert not app.dataframe
