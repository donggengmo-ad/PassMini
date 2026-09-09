"""逐字符概率工具的 Streamlit 输入组件与颜色展示。"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

import streamlit as st


_CHARACTER_INPUT = st.components.v2.component(
    "passmini_character_input",
    html="""
    <div class="character-input">
      <label for="value"></label>
      <input id="value" type="text" autocomplete="off" spellcheck="false" />
    </div>
    """,
    css="""
    .character-input {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      width: 100%;
      font-family: var(--st-font);
    }
    label {
      color: var(--st-text-color);
      font-size: 0.9rem;
      font-weight: 500;
    }
    input {
      box-sizing: border-box;
      width: 100%;
      padding: 0.55rem 0.75rem;
      color: var(--st-text-color);
      background: var(--st-background-color);
      border: 1px solid var(--st-widget-border-color);
      border-radius: var(--st-base-radius);
      font: inherit;
      font-family: var(--st-code-font);
    }
    input:focus {
      border-color: var(--st-primary-color);
      outline: 2px solid color-mix(in srgb, var(--st-primary-color) 25%, transparent);
    }
    """,
    js="""
    export default function (component) {
      const { parentElement, data, setStateValue } = component
      const label = parentElement.querySelector("label")
      const input = parentElement.querySelector("input")
      if (!label || !input) return

      label.textContent = data.label
      input.placeholder = data.placeholder
      input.maxLength = data.maxLength
      const nextValue = data.value ?? ""
      if (input.value !== nextValue) {
        input.value = nextValue
      }

      input.oninput = () => {
        const value = input.value
        setStateValue("value", value)
      }
    }
    """,
)


def _state_value(state: object, fallback: str) -> str:
    """兼容字典和组件状态对象，读取当前文本。"""

    if isinstance(state, Mapping):
        value = state.get("value", fallback)
    else:
        value = getattr(state, "value", fallback)
    return value if isinstance(value, str) else fallback


def character_input(
    label: str,
    *,
    key: str,
    max_length: int = 12,
    placeholder: str = "请输入演示密码",
) -> str:
    """挂载逐键更新的受控输入组件，并返回当前文本。"""

    # 非 widget 的备份键在切换页面后仍保留，回到实验时恢复输入。
    backup_key = key + "_text"
    current_value = _state_value(
        st.session_state.get(key, {}), st.session_state.get(backup_key, "")
    )
    result = _CHARACTER_INPUT(
        key=key,
        data={
            "label": label,
            "value": current_value,
            "maxLength": max_length,
            "placeholder": placeholder,
        },
        default={"value": current_value},
        on_value_change=lambda: None,
        width="stretch",
        height="content",
    )
    value = result.value if isinstance(result.value, str) else current_value
    st.session_state[backup_key] = value
    return value


def probability_color(probability: float) -> str:
    """将 `[0, 1]` 概率连续映射为红—黄—绿十六进制颜色。"""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability 必须位于 [0, 1]")
    red = (239, 68, 68)
    yellow = (245, 158, 11)
    green = (34, 197, 94)
    if probability <= 0.5:
        start, end, ratio = red, yellow, probability * 2
    else:
        start, end, ratio = yellow, green, (probability - 0.5) * 2
    channels = tuple(round(left + (right - left) * ratio) for left, right in zip(start, end))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def colored_password_html(characters: Sequence[tuple[str, float]]) -> str:
    """生成经过转义的逐字符概率展示 HTML。"""

    spans = []
    for character, probability in characters:
        visible_character = "&nbsp;" if character == " " else html.escape(character)
        tooltip = html.escape(f"{character!r}: {probability:.2%}", quote=True)
        color = probability_color(probability)
        spans.append(
            f'<span title="{tooltip}" style="color:{color};padding:0 0.04em">'
            f"{visible_character}</span>"
        )
    content = "".join(spans) or '<span style="color:var(--st-gray-text-color)">—</span>'
    return (
        '<div aria-label="逐字符概率" '
        'style="font-family:var(--st-code-font);font-size:2rem;font-weight:600;'
        f'letter-spacing:0.04em;line-height:1.5">{content}</div>'
    )
