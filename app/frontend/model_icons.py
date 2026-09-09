"""Warehouse 使用的模型家族 SVG 图标。"""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


_MODEL_ICON_COMPONENT = st.components.v2.component(
    "passmini_model_icon",
    html="""
    <div id="pm-model-icon-root"></div>
    """,
    js="""
    export default function (component) {
      const { data, parentElement } = component
      const root = parentElement.querySelector("#pm-model-icon-root")
      if (!root) return

      // SVG 来自 Python 内部固定模板，只在家族变化时写入，避免普通 rerun 重启动画。
      if (root.dataset.modelType !== data.modelType) {
        root.innerHTML = data.html
        root.dataset.modelType = data.modelType
      }
    }
    """,
)


_FAMILY_COLORS = {
    "bigram": "#f4b860",
    "mlp": "#ff8f70",
    "tcn": "#54a7ff",
    "gru": "#49d3b4",
    "transformer": "#a98bff",
}

_FAMILY_LABELS = {
    "bigram": "Bigram conditional probability distribution",
    "mlp": "MLP fixed-context fully connected network",
    "tcn": "TCN causal convolution scan",
    "gru": "GRU gated recurrent cell",
    "transformer": "Transformer query key value attention",
}


def _lines(
    starts: Iterable[float],
    start_y: float,
    ends: Iterable[float],
    end_y: float,
    class_name: str,
) -> str:
    """生成两排节点之间的全连接 SVG 线段。"""

    return "".join(
        f'<line class="{class_name}" x1="{start_x}" y1="{start_y}" '
        f'x2="{end_x}" y2="{end_y}" />'
        for start_x in starts
        for end_x in ends
    )


def _bigram_svg() -> str:
    # 每根柱使用独立但首尾闭合的高度序列，表示条件分布反复更新，而非行进波。
    levels = (
        (.22, .48, .31, .66, .38, .57),
        (.35, .71, .46, .28, .62, .41),
        (.58, .32, .77, .49, .68, .36),
        (.74, .52, .39, .83, .56, .69),
        (.43, .86, .61, .47, .79, .54),
        (.91, .63, .82, .58, .73, .88),
        (.67, .89, .55, .76, .48, .81),
        (.52, .37, .72, .59, .84, .44),
        (.79, .46, .64, .35, .57, .71),
        (.31, .61, .42, .74, .29, .53),
        (.47, .26, .56, .33, .65, .39),
        (.25, .44, .30, .52, .36, .21),
    )
    bars = "".join(
        f'<rect class="pm-bar pm-bar-{index}" x="{35 + index * 24}" y="58" '
        f'width="14" height="72" rx="4" style="'
        + ";".join(f"--pm-h{step}:{height}" for step, height in enumerate(sequence))
        + '" />'
        for index, sequence in enumerate(levels)
    )
    return f"""
    <svg viewBox="0 0 340 175" role="img" aria-label="{_FAMILY_LABELS['bigram']}">
      <text class="pm-symbol" x="24" y="34">xₜ</text>
      <path class="pm-wire" d="M48 29 H94" />
      <path class="pm-arrow" d="M94 29 l-8 -5 v10 z" />
      <text class="pm-formula" x="108" y="34">P(xₜ₊₁ | xₜ)</text>
      <line class="pm-axis" x1="26" y1="132" x2="322" y2="132" />
      {bars}
      <text class="pm-caption" x="170" y="158" text-anchor="middle">count → probability</text>
    </svg>
    """


def _mlp_svg() -> str:
    input_x = (70, 135, 200, 265)
    hidden_x = (45, 105, 170, 235, 295)
    output_x = (70, 135, 200, 265)
    connections = (
        _lines(input_x, 73, hidden_x, 112, "pm-wire")
        + _lines(hidden_x, 120, output_x, 151, "pm-wire")
        + _lines(input_x, 73, hidden_x, 112, "pm-mlp-flow pm-mlp-flow-a")
        + _lines(hidden_x, 120, output_x, 151, "pm-mlp-flow pm-mlp-flow-b")
    )
    # 右侧保留一整组副本；向左平移 260px 后与起始画面像素级一致。
    tokens = "".join(
        f'<rect class="pm-token" x="{56 + index * 65}" y="20" '
        'width="28" height="18" rx="5" />'
        for index in range(-4, 8)
    )
    input_nodes = "".join(
        f'<circle class="pm-node" cx="{x}" cy="69" r="8" />' for x in input_x
    )
    hidden_nodes = "".join(
        f'<circle class="pm-node" cx="{x}" cy="116" r="8" />' for x in hidden_x
    )
    output_nodes = "".join(
        f'<circle class="pm-node" cx="{x}" cy="155" r="8" />' for x in output_x
    )
    return f"""
    <svg viewBox="0 0 380 180" role="img" aria-label="{_FAMILY_LABELS['mlp']}">
      <defs><clipPath id="pm-mlp-window"><rect x="34" y="12" width="260" height="34" rx="8" /></clipPath></defs>
      <rect class="pm-window" x="34" y="12" width="260" height="34" rx="8" />
      <g clip-path="url(#pm-mlp-window)">
        <g class="pm-mlp-tokens">{tokens}</g>
      </g>
      <text class="pm-layer-label pm-layer-label-right" x="326" y="72">input</text>
      <text class="pm-layer-label pm-layer-label-right" x="326" y="119">hidden</text>
      <text class="pm-layer-label pm-layer-label-right" x="326" y="158">output</text>
      {connections}
      {input_nodes}{hidden_nodes}{output_nodes}
    </svg>
    """


def _tcn_svg() -> str:
    input_x = (38, 82, 126, 170, 214, 258, 302)
    tokens = "".join(
        f'<rect class="pm-token pm-tcn-token-{index}" x="{x - 13}" y="122" '
        'width="26" height="26" rx="7" />'
        for index, x in enumerate(input_x)
    )
    return f"""
    <svg viewBox="0 0 340 180" role="img" aria-label="{_FAMILY_LABELS['tcn']}">
      <defs>
        <marker id="pm-tcn-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path class="pm-tcn-marker" d="M0 0 L10 5 L0 10 z" />
        </marker>
      </defs>
      <path class="pm-causal-line" d="M25 111 H315" />
      {tokens}
      <g class="pm-tcn-kernel">
        <rect class="pm-kernel-box" x="24" y="39" width="116" height="32" rx="9" />
        <circle class="pm-kernel-tap pm-tcn-weight-0" cx="38" cy="55" r="4" />
        <circle class="pm-kernel-tap pm-tcn-weight-1" cx="82" cy="55" r="4" />
        <circle class="pm-kernel-tap pm-tcn-weight-2" cx="126" cy="55" r="4" />
        <line class="pm-active-wire pm-tcn-weight-0" x1="38" y1="71" x2="38" y2="122" />
        <line class="pm-active-wire pm-tcn-weight-1" x1="82" y1="71" x2="82" y2="122" />
        <line class="pm-active-wire pm-tcn-weight-2" x1="126" y1="71" x2="126" y2="122" />
        <rect class="pm-tcn-highlight pm-tcn-weight-0" x="25" y="122" width="26" height="26" rx="7" />
        <rect class="pm-tcn-highlight pm-tcn-weight-1" x="69" y="122" width="26" height="26" rx="7" />
        <rect class="pm-tcn-highlight pm-tcn-weight-2" x="113" y="122" width="26" height="26" rx="7" />
        <g class="pm-tcn-output">
          <path class="pm-tcn-output-arrow" marker-end="url(#pm-tcn-arrow)" d="M140 55 H159" />
          <rect class="pm-tcn-output-box" x="162" y="47" width="16" height="16" rx="3" />
        </g>
      </g>
      <text class="pm-caption" x="170" y="169" text-anchor="middle">shared kernel scans the sequence</text>
    </svg>
    """


def _gru_svg() -> str:
    return f"""
    <svg viewBox="0 0 340 180" role="img" aria-label="{_FAMILY_LABELS['gru']}">
      <defs>
        <marker id="pm-gru-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path class="pm-marker" d="M0 0 L10 5 L0 10 z" />
        </marker>
        <clipPath id="pm-gru-state-clip">
          <circle cx="49" cy="86" r="27" />
        </clipPath>
      </defs>
      <circle class="pm-state-shell" cx="49" cy="86" r="27" />
      <g clip-path="url(#pm-gru-state-clip)">
        <path class="pm-state-liquid" d="M15 87 Q25 78 36 87 T57 87 T79 87 V119 H15 Z" />
        <circle class="pm-state-bubble pm-state-bubble-a" cx="36" cy="99" r="3" />
        <circle class="pm-state-bubble pm-state-bubble-b" cx="58" cy="108" r="2" />
      </g>
      <text class="pm-caption" x="49" y="126" text-anchor="middle">hidden state</text>

      <rect class="pm-cell" x="108" y="34" width="130" height="112" rx="22" />
      <text class="pm-formula" x="207" y="55" text-anchor="middle">GRU cell</text>
      <circle class="pm-gate pm-gru-reset" cx="145" cy="70" r="16" />
      <circle class="pm-gate pm-gru-update" cx="145" cy="116" r="16" />
      <circle class="pm-mix-node" cx="205" cy="93" r="7" />
      <text class="pm-gate-text" x="145" y="75" text-anchor="middle">r</text>
      <text class="pm-gate-text" x="145" y="121" text-anchor="middle">z</text>

      <path class="pm-wire" d="M75 78 C95 70 111 70 129 70" />
      <path class="pm-wire" d="M161 70 C181 70 188 82 198 88" />
      <path class="pm-wire" d="M198 99 C184 110 172 116 161 116" />
      <path class="pm-wire" d="M129 116 C105 116 88 105 74 98" />
      <path class="pm-wire" d="M145 54 C145 39 130 35 130 23" />
      <path class="pm-wire" d="M205 169 V101" />
      <path class="pm-wire" marker-end="url(#pm-gru-arrow)" d="M212 93 H316" />
      <circle class="pm-drain-drop" cx="130" cy="18" r="4" />

      <path pathLength="1" class="pm-flow pm-gru-hidden-in" d="M75 78 C95 70 111 70 129 70" />
      <path pathLength="1" class="pm-flow pm-gru-retained" d="M161 70 C181 70 188 82 198 88" />
      <path pathLength="1" class="pm-flow pm-gru-to-update" d="M198 99 C184 110 172 116 161 116" />
      <path pathLength="1" class="pm-flow pm-gru-state-return" d="M129 116 C105 116 88 105 74 98" />
      <path pathLength="1" class="pm-flow pm-gru-drain" d="M145 54 C145 39 130 35 130 23" />
      <path pathLength="1" class="pm-flow pm-gru-input" d="M205 169 V101" />
      <path pathLength="1" class="pm-flow pm-gru-output" d="M212 93 H316" />
      <text class="pm-symbol" x="299" y="82">yₜ</text>
      <text class="pm-symbol" x="214" y="170">xₜ</text>
    </svg>
    """


def _transformer_svg() -> str:
    node_x = (46, 108, 170, 232, 294)
    attention_weights = (
        (.86, .28, .56, .20, .45),
        (.34, .90, .20, .57, .28),
        (.63, .24, .95, .30, .54),
        (.22, .68, .35, .90, .24),
        (.48, .37, .72, .43, .95),
    )
    query_nodes = "".join(
        f'<circle class="pm-node pm-q pm-q-{index}" cx="{x}" cy="31" r="9" />'
        for index, x in enumerate(node_x)
    )
    key_nodes = "".join(
        f'<circle class="pm-node pm-k pm-weight-{index}" cx="{x}" cy="91" r="9" />'
        for index, x in enumerate(node_x)
    )
    value_nodes = "".join(
        f'<circle class="pm-node pm-v pm-weight-{index}" cx="{x}" cy="151" r="9" />'
        for index, x in enumerate(node_x)
    )
    key_value = "".join(
        f'<line class="pm-wire pm-kv pm-kv-{index}" x1="{x}" y1="100" x2="{x}" y2="142" />'
        for index, x in enumerate(node_x)
    )
    fans = "".join(
        '<g class="pm-attention-fan pm-fan-{index}">'.format(index=query_index)
        + "".join(
            f'<line class="pm-attention-edge" x1="{query_x}" y1="40" '
            f'x2="{key_x}" y2="82" style="--pm-edge-opacity:{weight};'
            f'--pm-edge-width:{0.8 + 2.6 * weight:.2f}px" />'
            for key_x, weight in zip(node_x, attention_weights[query_index])
        )
        + '</g>'
        for query_index, query_x in enumerate(node_x)
    )
    return f"""
    <svg viewBox="0 0 340 180" role="img" aria-label="{_FAMILY_LABELS['transformer']}">
      <text class="pm-layer-label" x="19" y="35">Q</text>
      <text class="pm-layer-label" x="19" y="95">K</text>
      <text class="pm-layer-label" x="19" y="155">V</text>
      {key_value}{fans}{query_nodes}{key_nodes}{value_nodes}
    </svg>
    """


_SVG_BUILDERS = {
    "bigram": _bigram_svg,
    "mlp": _mlp_svg,
    "tcn": _tcn_svg,
    "gru": _gru_svg,
    "transformer": _transformer_svg,
}


def model_icon_html(model_type: str) -> str:
    """返回带有家族色和悬停动画的独立 SVG 卡片。"""

    if model_type not in _SVG_BUILDERS:
        raise ValueError(f"不支持的模型图标: {model_type!r}")
    color = _FAMILY_COLORS[model_type]
    svg = _SVG_BUILDERS[model_type]()
    return f"""
    <style>
      .pm-model-icon {{
        --pm-accent: {color};
        box-sizing: border-box;
        width: 100%;
        padding: 4px 8px;
        color: var(--st-text-color);
      }}
      .pm-model-icon svg {{ display: block; width: 100%; height: auto; max-height: 180px; overflow: visible; }}
      .pm-model-icon .pm-wire,
      .pm-model-icon .pm-axis,
      .pm-model-icon .pm-causal-line {{ fill: none; stroke: color-mix(in srgb, var(--st-text-color) 38%, transparent); stroke-width: 1.6; }}
      .pm-model-icon .pm-arrow,
      .pm-model-icon .pm-marker {{ fill: color-mix(in srgb, var(--st-text-color) 48%, transparent); }}
      .pm-model-icon .pm-node,
      .pm-model-icon .pm-gate {{ fill: var(--st-secondary-background-color); stroke: var(--pm-accent); stroke-width: 2; }}
      .pm-model-icon .pm-gate {{ transform-box: fill-box; transform-origin: center; }}
      .pm-model-icon .pm-token {{ fill: color-mix(in srgb, var(--pm-accent) 18%, var(--st-secondary-background-color)); stroke: var(--pm-accent); stroke-width: 1.4; }}
      .pm-model-icon .pm-gate-text,
      .pm-model-icon .pm-symbol,
      .pm-model-icon .pm-formula,
      .pm-model-icon .pm-layer-label,
      .pm-model-icon .pm-caption {{ fill: var(--st-text-color); font-family: var(--st-code-font); }}
      .pm-model-icon .pm-gate-text {{ font-size: 12px; font-weight: 650; }}
      .pm-model-icon .pm-symbol {{ font-size: 13px; font-weight: 650; }}
      .pm-model-icon .pm-formula {{ font-size: 12px; font-weight: 560; }}
      .pm-model-icon .pm-layer-label {{ font-size: 11px; opacity: .72; text-anchor: end; }}
      .pm-model-icon .pm-layer-label-right {{ text-anchor: start; }}
      .pm-model-icon .pm-caption {{ font-size: 10px; opacity: .64; }}

      .pm-model-icon .pm-bar {{ fill: var(--pm-accent); opacity: .82; transform-box: fill-box; transform-origin: center bottom; transform:scaleY(var(--pm-h0)); }}
      .pm-model-icon:hover .pm-bar {{ animation: pm-bigram-fluctuate 5.4s ease-in-out infinite; }}
      @keyframes pm-bigram-fluctuate {{
        0%,100% {{ transform:scaleY(var(--pm-h0)) }}
        17% {{ transform:scaleY(var(--pm-h1)) }}
        34% {{ transform:scaleY(var(--pm-h2)) }}
        52% {{ transform:scaleY(var(--pm-h3)) }}
        70% {{ transform:scaleY(var(--pm-h4)) }}
        86% {{ transform:scaleY(var(--pm-h5)) }}
      }}

      .pm-model-icon .pm-window {{ fill: none; stroke: color-mix(in srgb, var(--pm-accent) 58%, transparent); stroke-width: 1.4; stroke-dasharray: 5 4; }}
      .pm-model-icon .pm-mlp-flow {{ fill:none; stroke:var(--pm-accent); stroke-width:2.8; stroke-linecap:round; stroke-dasharray:7 96; opacity:0; }}
      .pm-model-icon:hover .pm-mlp-tokens {{ animation: pm-context-shift 6.4s linear infinite; }}
      .pm-model-icon:hover .pm-mlp-flow-a {{ animation: pm-flow-a 1.6s linear infinite; }}
      .pm-model-icon:hover .pm-mlp-flow-b {{ animation: pm-flow-b 1.6s linear infinite; }}
      @keyframes pm-context-shift {{
        0%,17% {{ transform:translateX(0) }} 25%,42% {{ transform:translateX(-65px) }}
        50%,67% {{ transform:translateX(-130px) }} 75%,92% {{ transform:translateX(-195px) }}
        100% {{ transform:translateX(-260px) }}
      }}
      @keyframes pm-flow-a {{
        0%,6% {{ opacity:0;stroke-dashoffset:100 }} 10%,30% {{ opacity:.95 }}
        34%,100% {{ opacity:0;stroke-dashoffset:0 }}
      }}
      @keyframes pm-flow-b {{
        0%,34% {{ opacity:0;stroke-dashoffset:100 }} 39%,59% {{ opacity:.95 }}
        64%,100% {{ opacity:0;stroke-dashoffset:0 }}
      }}

      .pm-model-icon .pm-kernel-box {{ fill: color-mix(in srgb, var(--pm-accent) 12%, var(--st-secondary-background-color)); stroke: var(--pm-accent); stroke-width: 1.8; }}
      .pm-model-icon .pm-tcn-weight-0 {{ --pm-tcn-opacity:.92; --pm-tcn-wire-width:2.4; --pm-tcn-projection-opacity:.42; }}
      .pm-model-icon .pm-tcn-weight-1 {{ --pm-tcn-opacity:.38; --pm-tcn-wire-width:1.2; --pm-tcn-projection-opacity:.16; }}
      .pm-model-icon .pm-tcn-weight-2 {{ --pm-tcn-opacity:.68; --pm-tcn-wire-width:1.8; --pm-tcn-projection-opacity:.30; }}
      .pm-model-icon .pm-kernel-tap {{ fill:var(--pm-accent); fill-opacity:var(--pm-tcn-opacity); }}
      .pm-model-icon .pm-active-wire {{ stroke:var(--pm-accent); stroke-opacity:var(--pm-tcn-opacity); stroke-width:var(--pm-tcn-wire-width); stroke-dasharray:5 4; }}
      .pm-model-icon .pm-tcn-highlight {{ fill:var(--pm-accent); opacity:var(--pm-tcn-projection-opacity); }}
      .pm-model-icon .pm-tcn-output {{ opacity:0; }}
      .pm-model-icon .pm-tcn-output-arrow {{ fill:none; stroke:var(--pm-accent); stroke-width:1.7; }}
      .pm-model-icon .pm-tcn-marker,
      .pm-model-icon .pm-tcn-output-box {{ fill:var(--pm-accent); }}
      .pm-model-icon:hover .pm-tcn-kernel {{ animation: pm-kernel-scan 4.8s linear infinite; }}
      .pm-model-icon:hover .pm-tcn-output {{ animation:pm-tcn-output-pulse 1.2s ease-in-out infinite; }}
      @keyframes pm-kernel-scan {{
        0%,15% {{ transform:translateX(0);opacity:1 }} 25%,40% {{ transform:translateX(44px);opacity:1 }}
        50%,65% {{ transform:translateX(88px);opacity:1 }} 75%,90% {{ transform:translateX(132px);opacity:1 }}
        93% {{ transform:translateX(132px);opacity:0 }} 94% {{ transform:translateX(0);opacity:0 }}
        100% {{ transform:translateX(0);opacity:1 }}
      }}
      @keyframes pm-tcn-output-pulse {{
        0%,8%,72%,100% {{ opacity:0 }}
        18%,58% {{ opacity:.9 }}
      }}

      .pm-model-icon .pm-cell {{ fill: color-mix(in srgb, var(--pm-accent) 9%, var(--st-secondary-background-color)); stroke: var(--pm-accent); stroke-width: 2; }}
      .pm-model-icon .pm-state-shell {{ fill:var(--st-secondary-background-color); stroke:var(--pm-accent); stroke-width:2.2; }}
      .pm-model-icon .pm-state-liquid {{ fill:var(--pm-accent); opacity:.62; transform-origin:center; }}
      .pm-model-icon .pm-state-bubble {{ fill:color-mix(in srgb, white 62%, var(--pm-accent)); opacity:.7; }}
      .pm-model-icon .pm-mix-node {{ fill:var(--pm-accent); opacity:.72; }}
      .pm-model-icon .pm-drain-drop {{ fill:var(--pm-accent); opacity:.25; transform-box:fill-box; transform-origin:center; }}
      .pm-model-icon .pm-flow {{ fill:none; stroke:var(--pm-accent); stroke-width:3.2; stroke-linecap:round; stroke-dasharray:.09 .91; opacity:0; }}
      .pm-model-icon:hover .pm-state-liquid {{ animation:pm-liquid-level 8s ease-in-out infinite; }}
      .pm-model-icon:hover .pm-state-bubble-a {{ animation:pm-bubble-a 8s ease-in infinite; }}
      .pm-model-icon:hover .pm-state-bubble-b {{ animation:pm-bubble-b 8s ease-in infinite; }}
      .pm-model-icon:hover .pm-gru-hidden-in {{ animation:pm-gru-hidden-in 8s linear infinite; }}
      .pm-model-icon:hover .pm-gru-reset {{ animation:pm-reset-pulse 8s ease-in-out infinite; }}
      .pm-model-icon:hover .pm-gru-drain {{ animation:pm-gru-drain 8s linear infinite; }}
      .pm-model-icon:hover .pm-drain-drop {{ animation:pm-drop-release 8s ease-in-out infinite; }}
      .pm-model-icon:hover .pm-gru-retained {{ animation:pm-gru-retained 8s linear infinite; }}
      .pm-model-icon:hover .pm-gru-input {{ animation:pm-gru-input 8s linear infinite; }}
      .pm-model-icon:hover .pm-gru-to-update {{ animation:pm-gru-to-update 8s linear infinite; }}
      .pm-model-icon:hover .pm-gru-update {{ animation:pm-update-pulse 8s ease-in-out infinite; }}
      .pm-model-icon:hover .pm-gru-state-return {{ animation:pm-gru-state-return 8s linear infinite; }}
      .pm-model-icon:hover .pm-gru-output {{ animation:pm-gru-output 8s linear infinite; }}
      @keyframes pm-liquid-level {{
        0%,2%,100% {{ transform:translate(-2px,-4px) }} 12%,52% {{ transform:translate(2px,5px) }}
        60%,96% {{ transform:translate(-2px,-4px) }}
      }}
      @keyframes pm-bubble-a {{ 0%,48%,72%,100% {{ transform:translateY(8px);opacity:0 }} 55% {{ opacity:.75 }} 68% {{ transform:translateY(-16px);opacity:0 }} }}
      @keyframes pm-bubble-b {{ 0%,52%,76%,100% {{ transform:translateY(8px);opacity:0 }} 60% {{ opacity:.7 }} 72% {{ transform:translateY(-16px);opacity:0 }} }}
      @keyframes pm-gru-hidden-in {{ 0%,2% {{ opacity:0;stroke-dashoffset:1 }} 3%,9% {{ opacity:1 }} 10%,100% {{ opacity:0;stroke-dashoffset:0 }} }}
      @keyframes pm-reset-pulse {{ 0%,10%,16%,100% {{ fill:var(--st-secondary-background-color);transform:scale(1) }} 12%,14% {{ fill:var(--pm-accent);transform:scale(1.12) }} }}
      @keyframes pm-gru-drain {{ 0%,16% {{ opacity:0;stroke-dashoffset:1 }} 17%,21% {{ opacity:1 }} 22%,100% {{ opacity:0;stroke-dashoffset:0 }} }}
      @keyframes pm-drop-release {{ 0%,16%,23%,100% {{ opacity:.2;transform:translateY(5px) scale(.55) }} 19% {{ opacity:.9;transform:translateY(0) scale(1) }} 22% {{ opacity:0;transform:translateY(-8px) scale(.7) }} }}
      @keyframes pm-gru-retained {{ 0%,22% {{ opacity:0;stroke-dashoffset:1 }} 23%,29% {{ opacity:1 }} 30%,100% {{ opacity:0;stroke-dashoffset:0 }} }}
      @keyframes pm-gru-input {{ 0%,30% {{ opacity:0;stroke-dashoffset:1 }} 31%,37% {{ opacity:1 }} 38%,100% {{ opacity:0;stroke-dashoffset:0 }} }}
      @keyframes pm-gru-to-update {{ 0%,38% {{ opacity:0;stroke-dashoffset:1 }} 39%,45% {{ opacity:1 }} 46%,100% {{ opacity:0;stroke-dashoffset:0 }} }}
      @keyframes pm-update-pulse {{ 0%,46%,52%,100% {{ fill:var(--st-secondary-background-color);transform:scale(1) }} 48%,50% {{ fill:var(--pm-accent);transform:scale(1.12) }} }}
      @keyframes pm-gru-state-return {{ 0%,52% {{ opacity:0;stroke-dashoffset:1 }} 53%,59% {{ opacity:1 }} 60%,100% {{ opacity:0;stroke-dashoffset:0 }} }}
      @keyframes pm-gru-output {{ 0%,60% {{ opacity:0;stroke-dashoffset:1 }} 61%,69% {{ opacity:1 }} 70%,100% {{ opacity:0;stroke-dashoffset:0 }} }}

      .pm-model-icon .pm-attention-fan {{ opacity:0; }}
      .pm-model-icon .pm-attention-edge {{ stroke:var(--pm-accent); stroke-opacity:var(--pm-edge-opacity); stroke-width:var(--pm-edge-width); }}
      .pm-model-icon .pm-fan-2 {{ opacity:.5; }}
      .pm-model-icon .pm-kv {{ stroke-dasharray:4 5; }}
      .pm-model-icon .pm-k,
      .pm-model-icon .pm-v {{ fill:var(--pm-accent); fill-opacity:.24; }}
      .pm-model-icon .pm-weight-0 {{ fill-opacity:.76; }}
      .pm-model-icon .pm-weight-1 {{ fill-opacity:.30; }}
      .pm-model-icon .pm-weight-2 {{ fill-opacity:.58; }}
      .pm-model-icon .pm-weight-3 {{ fill-opacity:.20; }}
      .pm-model-icon .pm-weight-4 {{ fill-opacity:.43; }}
      .pm-model-icon:hover .pm-fan-0 {{ animation:pm-fan-0 5s infinite; }}
      .pm-model-icon:hover .pm-fan-1 {{ animation:pm-fan-1 5s infinite; }}
      .pm-model-icon:hover .pm-fan-2 {{ animation:pm-fan-2 5s infinite; }}
      .pm-model-icon:hover .pm-fan-3 {{ animation:pm-fan-3 5s infinite; }}
      .pm-model-icon:hover .pm-fan-4 {{ animation:pm-fan-4 5s infinite; }}
      .pm-model-icon:hover .pm-q-0 {{ animation:pm-q-0 5s infinite; }}
      .pm-model-icon:hover .pm-q-1 {{ animation:pm-q-1 5s infinite; }}
      .pm-model-icon:hover .pm-q-2 {{ animation:pm-q-2 5s infinite; }}
      .pm-model-icon:hover .pm-q-3 {{ animation:pm-q-3 5s infinite; }}
      .pm-model-icon:hover .pm-q-4 {{ animation:pm-q-4 5s infinite; }}
      .pm-model-icon:hover .pm-weight-0 {{ animation:pm-weight-0 5s infinite; }}
      .pm-model-icon:hover .pm-weight-1 {{ animation:pm-weight-1 5s infinite; }}
      .pm-model-icon:hover .pm-weight-2 {{ animation:pm-weight-2 5s infinite; }}
      .pm-model-icon:hover .pm-weight-3 {{ animation:pm-weight-3 5s infinite; }}
      .pm-model-icon:hover .pm-weight-4 {{ animation:pm-weight-4 5s infinite; }}
      @keyframes pm-fan-0 {{ 0%,4%,18%,100% {{opacity:0}} 7%,15% {{opacity:.85}} }}
      @keyframes pm-fan-1 {{ 0%,20%,38%,100% {{opacity:0}} 24%,35% {{opacity:.85}} }}
      @keyframes pm-fan-2 {{ 0%,40%,58%,100% {{opacity:0}} 44%,55% {{opacity:.85}} }}
      @keyframes pm-fan-3 {{ 0%,60%,78%,100% {{opacity:0}} 64%,75% {{opacity:.85}} }}
      @keyframes pm-fan-4 {{ 0%,80%,98%,100% {{opacity:0}} 84%,95% {{opacity:.85}} }}
      @keyframes pm-q-0 {{ 0%,4%,18%,100% {{fill:var(--st-secondary-background-color)}} 7%,15% {{fill:var(--pm-accent)}} }}
      @keyframes pm-q-1 {{ 0%,20%,38%,100% {{fill:var(--st-secondary-background-color)}} 24%,35% {{fill:var(--pm-accent)}} }}
      @keyframes pm-q-2 {{ 0%,40%,58%,100% {{fill:var(--st-secondary-background-color)}} 44%,55% {{fill:var(--pm-accent)}} }}
      @keyframes pm-q-3 {{ 0%,60%,78%,100% {{fill:var(--st-secondary-background-color)}} 64%,75% {{fill:var(--pm-accent)}} }}
      @keyframes pm-q-4 {{ 0%,80%,98%,100% {{fill:var(--st-secondary-background-color)}} 84%,95% {{fill:var(--pm-accent)}} }}
      @keyframes pm-weight-0 {{ 0%,18%,100% {{fill-opacity:.86}} 20%,38% {{fill-opacity:.34}} 40%,58% {{fill-opacity:.63}} 60%,78% {{fill-opacity:.22}} 80%,98% {{fill-opacity:.48}} }}
      @keyframes pm-weight-1 {{ 0%,18%,100% {{fill-opacity:.28}} 20%,38% {{fill-opacity:.90}} 40%,58% {{fill-opacity:.24}} 60%,78% {{fill-opacity:.68}} 80%,98% {{fill-opacity:.37}} }}
      @keyframes pm-weight-2 {{ 0%,18%,100% {{fill-opacity:.56}} 20%,38% {{fill-opacity:.20}} 40%,58% {{fill-opacity:.95}} 60%,78% {{fill-opacity:.35}} 80%,98% {{fill-opacity:.72}} }}
      @keyframes pm-weight-3 {{ 0%,18%,100% {{fill-opacity:.20}} 20%,38% {{fill-opacity:.57}} 40%,58% {{fill-opacity:.30}} 60%,78% {{fill-opacity:.90}} 80%,98% {{fill-opacity:.43}} }}
      @keyframes pm-weight-4 {{ 0%,18%,100% {{fill-opacity:.45}} 20%,38% {{fill-opacity:.28}} 40%,58% {{fill-opacity:.54}} 60%,78% {{fill-opacity:.24}} 80%,98% {{fill-opacity:.95}} }}

      @media (prefers-reduced-motion: reduce) {{
        .pm-model-icon *, .pm-model-icon:hover * {{ animation: none !important; }}
      }}
    </style>
    <div class="pm-model-icon pm-model-icon-{model_type}">{svg}</div>
    """


def render_model_icon(model_type: str) -> None:
    """在 Streamlit 中渲染不触发 Python rerun 的模型家族动画。"""

    _MODEL_ICON_COMPONENT(
        key=f"model-family-icon-{model_type}",
        data={
            "modelType": model_type,
            "html": model_icon_html(model_type),
        },
        width="stretch",
        height="content",
    )
