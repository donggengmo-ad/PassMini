"""Playground 的多模型评分、生成和补全编排。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import altair as alt
import numpy as np
import pandas as pd
import torch

from scripts.data import normalize_password
from scripts.inference import GenerationCandidate, beam_search, score_password
from scripts.models import AutoregressivePasswordModel, PasswordModel
from scripts.tokenizer import CharTokenizer


RuntimeModel = tuple[PasswordModel, CharTokenizer]


@dataclass(frozen=True)
class PasswordScore:
    model_id: str
    surprisal_bits: float
    bits_per_token: float


@dataclass(frozen=True)
class ScoreAggregate:
    mean_surprisal_bits: float
    mean_bits_per_token: float
    standard_deviation: float


@dataclass(frozen=True)
class CharacterProbability:
    """记录字符在其历史前缀之后出现的条件概率。"""

    character: str
    probability: float
    log_probability: float | None = None

    @property
    def surprisal_bits(self) -> float:
        """将字符条件概率转换为该步贡献的惊讶度。"""

        if self.log_probability is not None:
            return -self.log_probability / math.log(2.0)
        return -math.log2(self.probability) if self.probability > 0 else math.inf


@dataclass(frozen=True)
class NextTokenPrediction:
    """记录当前前缀之后的一个候选 token。"""

    token: str
    probability: float
    is_eos: bool


@dataclass(frozen=True)
class ModelProbabilityTrace:
    """保存一个模型对完整输入历史和下一 token 的分析。"""

    model_id: str
    characters: tuple[CharacterProbability, ...]
    next_tokens: tuple[NextTokenPrediction, ...]
    distribution: tuple[NextTokenPrediction, ...] = ()


def validate_password_input(password: str, max_length: int = 12) -> None:
    """沿用数据集可打印 ASCII 和长度边界校验演示输入。"""

    if normalize_password(password, min_length=1, max_length=max_length) != password:
        raise ValueError(f"请输入 1–{max_length} 个可打印 ASCII 字符")


def score_models(password: str, models: Mapping[str, RuntimeModel]) -> list[PasswordScore]:
    """对同一密码逐模型评分，保留每个模型的独立结果。"""

    validate_password_input(password)
    results: list[PasswordScore] = []
    for model_id, (model, tokenizer) in models.items():
        surprisal = score_password(model, tokenizer, password)
        results.append(
            PasswordScore(
                model_id=model_id,
                surprisal_bits=surprisal,
                bits_per_token=surprisal / (len(password) + 1),
            )
        )
    return results


def aggregate_scores(scores: list[PasswordScore]) -> ScoreAggregate:
    """计算模型集合的均值，并用标准差保留模型分歧信息。"""

    if not scores:
        raise ValueError("至少需要一个模型评分")
    surprisals = np.asarray([score.surprisal_bits for score in scores])
    per_token = np.asarray([score.bits_per_token for score in scores])
    return ScoreAggregate(
        mean_surprisal_bits=float(surprisals.mean()),
        mean_bits_per_token=float(per_token.mean()),
        standard_deviation=float(surprisals.std()),
    )


def plot_score_comparison(
    scores: list[PasswordScore],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
) -> alt.Chart:
    """用条形图展示同一密码在各模型下的惊讶度。"""

    if not scores:
        raise ValueError("没有可绘制的评分")
    rows = [
        {
            "Model": labels[score.model_id],
            "Surprisal (bits)": score.surprisal_bits,
            "Bits per Token": score.bits_per_token,
        }
        for score in scores
    ]
    model_labels = [labels[score.model_id] for score in scores]
    # 低惊讶度样本固定使用 0–100，超过后以 25 bit 为一级平滑扩展上限。
    maximum = max(score.surprisal_bits for score in scores)
    axis_upper = max(100.0, math.ceil(maximum / 25.0) * 25.0)
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            y=alt.Y("Model:N", sort=model_labels, title=None),
            x=alt.X(
                "Surprisal (bits):Q",
                scale=alt.Scale(domain=[0, axis_upper], zero=True),
            ),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=model_labels,
                    range=[colors[score.model_id] for score in scores],
                ),
                legend=None,
            ),
            tooltip=[
                "Model:N",
                alt.Tooltip("Surprisal (bits):Q", format=".3f"),
                alt.Tooltip("Bits per Token:Q", format=".3f"),
            ],
        )
        .properties(title="Password Surprisal by Model", height=max(180, 55 * len(rows)))
    )


def trace_character_probabilities(
    text: str,
    models: Mapping[str, RuntimeModel],
    top_k: int = 5,
    max_length: int = 12,
    temperature: float = 1.0,
) -> list[ModelProbabilityTrace]:
    """每个模型只前向一次，返回输入字符的概率轨迹和下一 token 候选。"""

    return _trace_character_temperatures(
        text, models, (temperature,), top_k, max_length
    )[temperature]


def trace_temperature_probabilities(
    text: str,
    models: Mapping[str, RuntimeModel],
    temperatures: tuple[float, ...],
    top_k: int = 8,
    max_length: int = 12,
) -> dict[float, ModelProbabilityTrace]:
    """对一个模型只前向一次，复用 logits 比较多个温度；不跨会话缓存输入。"""

    if len(models) != 1:
        raise ValueError("Temperature Laboratory 必须选择一个模型")
    return {
        temperature: items[0]
        for temperature, items in _trace_character_temperatures(
            text, models, temperatures, top_k, max_length
        ).items()
    }


def _trace_character_temperatures(
    text: str,
    models: Mapping[str, RuntimeModel],
    temperatures: tuple[float, ...],
    top_k: int,
    max_length: int,
) -> dict[float, list[ModelProbabilityTrace]]:
    """逐字符计算条件概率，并给出当前前缀之后的 Top-K token。

    每个模型用一次因果前向计算 BOS 和完整输入的所有位置。第 i 个字符
    的概率始终来自它出现之前的前缀，所以继续追加字符不会改变已经记录的
    颜色。PAD、BOS、UNK 不属于可生成候选；EOS 保留并显示为结束标记。
    """

    if normalize_password(text, min_length=0, max_length=max_length) != text:
        raise ValueError(f"请输入 0–{max_length} 个可打印 ASCII 字符")
    if not models:
        raise ValueError("至少需要一个模型")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if not temperatures or any(not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError("temperature 必须是有限正数")
    traces = {temperature: [] for temperature in temperatures}
    with torch.inference_mode():
        for model_id, (model, tokenizer) in models.items():
            if not isinstance(model, AutoregressivePasswordModel):
                raise TypeError(f"{model_id} 不支持自回归逐字符预测")
            if model.tokenizer != tokenizer:
                raise ValueError(f"{model_id} 的模型与 tokenizer 不一致")

            model.eval()
            token_ids = []
            for character in text:
                token_id = tokenizer.token_to_id.get(character)
                if token_id is None or tokenizer.is_special_token(token_id):
                    raise ValueError(f"字符 {character!r} 不在 {model_id} 的可用词表中")
                token_ids.append(token_id)
            # 因果模型一次前向就能还原每个历史位置的预测，避免 rerun 内逐字符推进。
            inputs = torch.tensor([[tokenizer.bos_id, *token_ids]], device=model.device)
            # 不同温度只改变概率变换，不改变模型 logits；一次前向后在 CPU 复用。
            logits = model(inputs)[0].detach().double().cpu()
            for temperature in temperatures:
                scores = logits / temperature
                invalid_ids = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]
                scores[:, invalid_ids] = -torch.inf
                if (torch.isnan(scores).any() or torch.isposinf(scores).any()
                        or not torch.isfinite(scores).any(dim=-1).all()):
                    raise ValueError("模型没有可展示的有效概率分布")
                # 保留 log probability：即使 exp 下溢为 0，仍可准确绘制该字符的惊讶度。
                log_probabilities = torch.log_softmax(scores, dim=-1).cpu()
                character_probabilities = tuple(
                    CharacterProbability(
                        character=character,
                        probability=math.exp(float(log_probabilities[index, token_id])),
                        log_probability=float(log_probabilities[index, token_id]),
                    )
                    for index, (character, token_id) in enumerate(zip(text, token_ids))
                )
                # 先排除特殊 token，再按分数稳定排序；概率下溢时也不会混入 PAD/UNK。
                allowed = [i for i in range(tokenizer.vocab_size) if i not in invalid_ids]
                ordered = sorted(allowed, key=lambda i: (-float(log_probabilities[-1, i]), i))
                distribution = tuple(
                    NextTokenPrediction(
                        token="[EOS]" if token_id == tokenizer.eos_id else tokenizer.id_to_token[token_id],
                        probability=math.exp(float(log_probabilities[-1, token_id])),
                        is_eos=token_id == tokenizer.eos_id,
                    )
                    for token_id in ordered
                )
                traces[temperature].append(
                    ModelProbabilityTrace(
                        model_id=model_id,
                        characters=character_probabilities,
                        next_tokens=distribution[:top_k],
                        distribution=distribution,
                    )
                )
    return traces


def generate_with_models(
    models: Mapping[str, RuntimeModel],
    num_samples: int,
    max_length: int,
    temperature: float,
    seed: int,
) -> dict[str, list[str]]:
    """使用相同参数逐模型随机生成，但不对生成字符串求平均。"""

    if not 1 <= num_samples <= 20:
        raise ValueError("num_samples 必须位于 [1, 20]")
    results: dict[str, list[str]] = {}
    for model_id, (model, _) in models.items():
        generator = torch.Generator(device=model.device).manual_seed(seed)
        results[model_id] = model.generate_batch(
            batch_size=num_samples,
            max_length=max_length,
            temperature=temperature,
            generator=generator,
        )
    return results


def generate_challenge_password(
    model_id: str,
    runtime: RuntimeModel,
    max_length: int,
    temperature: float,
    seed: int,
    candidate_count: int = 8,
) -> str:
    """批量生成少量候选并返回首个非空密码，供 Generator versus Judge 使用。"""

    generated = generate_with_models(
        {model_id: runtime},
        num_samples=candidate_count,
        max_length=max_length,
        temperature=temperature,
        seed=seed,
    )[model_id]
    for password in generated:
        if password:
            return password
    raise ValueError("该模型本轮全部立即生成 EOS；请更换 seed 或 temperature")


def complete_with_models(
    models: Mapping[str, RuntimeModel],
    prefix: str,
    beam_width: int,
    max_length: int,
    temperature: float,
) -> dict[str, list[GenerationCandidate]]:
    """使用相同搜索参数逐模型执行前缀 Beam 补全。"""

    if normalize_password(prefix, min_length=1, max_length=max_length) != prefix:
        raise ValueError(f"前缀必须是 1–{max_length} 个可打印 ASCII 字符")
    results: dict[str, list[GenerationCandidate]] = {}
    for model_id, (model, tokenizer) in models.items():
        if not isinstance(model, AutoregressivePasswordModel):
            raise TypeError(f"{model_id} 不支持自回归 Beam 补全")
        results[model_id] = beam_search(
            model,
            tokenizer,
            prefix=prefix,
            beam_width=beam_width,
            max_length=max_length,
            temperature=temperature,
        )
    return results


def distribution_entropy(trace: ModelProbabilityTrace) -> float:
    """计算当前下一 token 分布的 Shannon entropy，单位为 bit。"""

    probabilities = np.asarray(
        [item.probability for item in trace.distribution], dtype=np.float64
    )
    if probabilities.size == 0:
        raise ValueError("概率轨迹不包含完整分布")
    positive = probabilities > 0
    return float(-np.sum(probabilities[positive] * np.log2(probabilities[positive])))


def plot_surprisal_journey(
    traces: list[ModelProbabilityTrace],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
) -> alt.Chart:
    """按模型分面绘制单步、累计和单位 token 惊讶度。"""

    rows: list[dict[str, float | int | str]] = []
    for trace in traces:
        cumulative = 0.0
        for position, item in enumerate(trace.characters, start=1):
            cumulative += item.surprisal_bits
            rows.append(
                {
                    "Model": labels[trace.model_id],
                    "Position": position,
                    "Character": repr(item.character),
                    "Step Surprisal": item.surprisal_bits,
                    "Cumulative Surprisal": cumulative,
                    "Surprisal per Token": cumulative / position,
                }
            )
    if not rows:
        raise ValueError("输入至少一个字符后才能绘制惊讶度旅程")
    model_ids = [trace.model_id for trace in traces]
    model_labels = [labels[model_id] for model_id in model_ids]
    color = alt.Color(
        "Model:N",
        scale=alt.Scale(
            domain=model_labels,
            range=[colors[model_id] for model_id in model_ids],
        ),
    )
    frame = pd.DataFrame(rows)
    # 先按模型分面，再在每个模型内部按字符位置排列，避免位置成为首要视觉分组。
    step = (
        alt.Chart(frame)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("Position:O", title="Character Position"),
            y=alt.Y(
                "Step Surprisal:Q",
                title="Step Surprisal (bits)",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            color=color,
            tooltip=[
                "Model:N",
                "Position:O",
                "Character:N",
                alt.Tooltip("Step Surprisal:Q", format=".3f"),
            ],
        )
        .properties(width="container", height=250)
    )
    # 按模型纵向排列，每幅图跟随容器宽度；窄屏不再强制容纳两幅 300px 图。
    step = alt.vconcat(*[
        step.transform_filter(alt.datum.Model == label).encode(
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=model_labels, range=[colors[i] for i in model_ids]),
                legend=None,
            )
        ).properties(title=label)
        for label in model_labels
    ]).properties(title="Character Surprisal Journey")
    cumulative = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("Position:Q", scale=alt.Scale(domainMin=0, zero=True)),
            y=alt.Y(
                "Cumulative Surprisal:Q",
                title="Cumulative Surprisal (bits)",
                scale=alt.Scale(zero=False),
            ),
            color=color,
            tooltip=[
                "Model:N",
                "Position:Q",
                "Character:N",
                alt.Tooltip("Cumulative Surprisal:Q", format=".3f"),
            ],
        )
        .properties(height=440)
    )
    per_token = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("Position:Q", scale=alt.Scale(domainMin=0, zero=True)),
            y=alt.Y(
                "Surprisal per Token:Q",
                title="Surprisal per Token (bits)",
                scale=alt.Scale(zero=False),
            ),
            color=color,
            tooltip=[
                "Model:N",
                "Position:Q",
                "Character:N",
                alt.Tooltip("Surprisal per Token:Q", format=".3f"),
            ],
        )
        .properties(title="Surprisal per Token", height=440)
    )
    return alt.vconcat(step, cumulative, per_token).resolve_scale(color="shared")


def _keyboard_positions(tokens: set[str]) -> list[dict[str, int | str]]:
    """将可打印 ASCII token 按类别排入稳定网格，便于比较概率色块。"""

    groups = [
        ("Digits", "0123456789"),
        ("Lowercase A–M", "abcdefghijklm"),
        ("Lowercase N–Z", "nopqrstuvwxyz"),
        ("Uppercase A–M", "ABCDEFGHIJKLM"),
        ("Uppercase N–Z", "NOPQRSTUVWXYZ"),
        ("Symbols 1", " !\"#$%&'()*+,-./"),
        ("Symbols 2", ":;<=>?@[\\]^_`{|}~"),
    ]
    positions: list[dict[str, int | str]] = []
    for row, (group, characters) in enumerate(groups):
        positions.extend(
            {
                "Token": character,
                "Display": "Space" if character == " " else character,
                "Row": row,
                "Group": group,
                "Column": column,
            }
            for column, character in enumerate(characters)
            if character in tokens
        )
    if "[EOS]" in tokens:
        positions.append(
            {
                "Token": "[EOS]",
                "Display": "EOS",
                "Row": len(groups),
                "Group": "End token",
                "Column": 0,
            }
        )
    return positions


def plot_probability_keyboard(
    trace: ModelProbabilityTrace,
    label: str,
) -> alt.LayerChart:
    """把完整下一 token 分布映射到字符网格，颜色按当前最大概率归一化。"""

    probability = {item.token: item.probability for item in trace.distribution}
    positions = _keyboard_positions(set(probability))
    if not positions:
        raise ValueError("概率轨迹不包含可显示 token")
    for item in positions:
        item["Probability"] = probability[str(item["Token"])]
    frame = pd.DataFrame(positions)
    maximum = max(float(frame["Probability"].max()), np.finfo(float).eps)
    group_order = [
        "Digits",
        "Lowercase A–M",
        "Lowercase N–Z",
        "Uppercase A–M",
        "Uppercase N–Z",
        "Symbols 1",
        "Symbols 2",
        "End token",
    ]
    base = alt.Chart(frame).encode(
        x=alt.X("Column:O", axis=None),
        y=alt.Y("Group:N", sort=group_order, title=None),
        tooltip=["Display:N", alt.Tooltip("Probability:Q", format=".3%")],
    )
    cells = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "Probability:Q",
            scale=alt.Scale(
                domain=[0, maximum / 2, maximum],
                range=["#ef4444", "#f59e0b", "#22c55e"],
            ),
            legend=alt.Legend(format=".1%"),
        )
    )
    text = base.mark_text(color="#111827", fontSize=12).encode(text="Display:N")
    return (cells + text).properties(title=f"Probability Keyboard · {label}", height=330)


def plot_temperature_laboratory(
    traces: Mapping[float, ModelProbabilityTrace],
    top_tokens: int = 8,
) -> alt.VConcatChart:
    """比较多个温度下的下一 token 概率和分布熵。"""

    if len(traces) < 2:
        raise ValueError("温度实验至少需要两个温度")
    mean_probabilities: dict[str, float] = {}
    for trace in traces.values():
        for item in trace.distribution:
            mean_probabilities[item.token] = mean_probabilities.get(item.token, 0.0) + item.probability
    selected = {
        token
        for token, _ in sorted(
            mean_probabilities.items(), key=lambda item: item[1], reverse=True
        )[:top_tokens]
    }
    probability_rows = [
        {"Temperature": temperature, "Token": item.token, "Probability": item.probability}
        for temperature, trace in traces.items()
        for item in trace.distribution
        if item.token in selected
    ]
    entropy_rows = [
        {"Temperature": temperature, "Entropy (bits)": distribution_entropy(trace)}
        for temperature, trace in traces.items()
    ]
    probability_chart = (
        alt.Chart(pd.DataFrame(probability_rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("Temperature:Q", scale=alt.Scale(domainMin=0, zero=True)),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(domainMin=0, zero=True),
                axis=alt.Axis(format=".0%"),
            ),
            color="Token:N",
            tooltip=[
                alt.Tooltip("Temperature:Q", format=".2f"),
                "Token:N",
                alt.Tooltip("Probability:Q", format=".2%"),
            ],
        )
        .properties(title="Temperature Response of Top Tokens", height=300)
    )
    entropy_chart = (
        alt.Chart(pd.DataFrame(entropy_rows))
        .mark_line(point=True, color="#54a7ff")
        .encode(
            x=alt.X("Temperature:Q", scale=alt.Scale(domainMin=0, zero=True)),
            y=alt.Y(
                "Entropy (bits):Q",
                scale=alt.Scale(domainMin=0, zero=True),
            ),
            tooltip=[
                alt.Tooltip("Temperature:Q", format=".2f"),
                alt.Tooltip("Entropy (bits):Q", format=".3f"),
            ],
        )
        .properties(title="Next-Token Entropy", height=220)
    )
    return alt.vconcat(probability_chart, entropy_chart)


def model_disagreement_matrix(traces: list[ModelProbabilityTrace]) -> pd.DataFrame:
    """计算模型下一 token 分布之间的 Jensen–Shannon divergence，单位为 bit。"""

    if len(traces) < 2:
        raise ValueError("模型分歧实验至少需要两个模型")
    token_order = sorted({item.token for trace in traces for item in trace.distribution})
    distributions = []
    for trace in traces:
        if not trace.distribution:
            raise ValueError(f"{trace.model_id} 的概率轨迹不包含完整分布")
        mapping = {item.token: item.probability for item in trace.distribution}
        vector = np.asarray([mapping.get(token, 0.0) for token in token_order])
        vector /= vector.sum()
        distributions.append(vector)

    def divergence(left: np.ndarray, right: np.ndarray) -> float:
        midpoint = 0.5 * (left + right)
        left_mask = left > 0
        right_mask = right > 0
        left_kl = np.sum(left[left_mask] * np.log2(left[left_mask] / midpoint[left_mask]))
        right_kl = np.sum(right[right_mask] * np.log2(right[right_mask] / midpoint[right_mask]))
        return float(0.5 * (left_kl + right_kl))

    matrix = np.asarray(
        [[divergence(left, right) for right in distributions] for left in distributions]
    )
    model_ids = [trace.model_id for trace in traces]
    return pd.DataFrame(matrix, index=model_ids, columns=model_ids)


def plot_model_disagreement(
    traces: list[ModelProbabilityTrace],
    labels: Mapping[str, str],
    top_tokens: int = 12,
) -> alt.VConcatChart:
    """用 token 概率热力图和 JSD 热力图展示模型分歧来自哪里。"""

    matrix = model_disagreement_matrix(traces)
    averages: dict[str, list[float]] = {}
    for trace in traces:
        for item in trace.distribution:
            averages.setdefault(item.token, []).append(item.probability)
    selected = [
        token
        for token, _ in sorted(
            averages.items(), key=lambda item: np.mean(item[1]), reverse=True
        )[:top_tokens]
    ]
    probability_rows = [
        {
            "Model": labels[trace.model_id],
            "Token": token,
            "Probability": next(
                (item.probability for item in trace.distribution if item.token == token),
                0.0,
            ),
        }
        for trace in traces
        for token in selected
    ]
    probability_chart = (
        alt.Chart(pd.DataFrame(probability_rows))
        .mark_rect()
        .encode(
            x=alt.X("Token:N", sort=selected),
            y=alt.Y("Model:N", sort=[labels[trace.model_id] for trace in traces]),
            color=alt.Color("Probability:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Model:N", "Token:N", alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(title="Next-Token Probability Disagreement", height=170)
    )
    jsd_rows = [
        {
            "Model A": labels[row],
            "Model B": labels[column],
            "JSD (bits)": matrix.loc[row, column],
        }
        for row in matrix.index
        for column in matrix.columns
    ]
    jsd_chart = (
        alt.Chart(pd.DataFrame(jsd_rows))
        .mark_rect()
        .encode(
            x=alt.X("Model B:N", sort=[labels[item] for item in matrix.columns]),
            y=alt.Y("Model A:N", sort=[labels[item] for item in matrix.index]),
            color=alt.Color("JSD (bits):Q", scale=alt.Scale(scheme="magma")),
            tooltip=["Model A:N", "Model B:N", alt.Tooltip("JSD (bits):Q", format=".4f")],
        )
        .properties(title="Pairwise Jensen–Shannon Divergence", height=260)
    )
    return alt.vconcat(probability_chart, jsd_chart)


def completion_consensus_frame(
    completions: Mapping[str, list[GenerationCandidate]],
    labels: Mapping[str, str],
) -> pd.DataFrame:
    """把各模型补全候选的名次与对数概率整理为统一长表。"""

    if not completions:
        raise ValueError("没有补全结果")
    rows = [
        {
            "Candidate": candidate.text,
            "Model": labels[model_id],
            "Rank": rank,
            "Log Probability": candidate.log_probability,
        }
        for model_id, candidates in completions.items()
        for rank, candidate in enumerate(candidates, start=1)
    ]
    if not rows:
        raise ValueError("所有模型都没有返回补全候选")
    return pd.DataFrame(rows)


def completion_consensus_summary(
    completions: Mapping[str, list[GenerationCandidate]],
    labels: Mapping[str, str],
) -> pd.DataFrame:
    """按候选汇总支持模型数和归一化 reciprocal-rank consensus。"""

    frame = completion_consensus_frame(completions, labels)
    model_count = len(completions)
    rows = []
    for candidate, group in frame.groupby("Candidate", sort=False):
        ranks = {
            row.Model: int(row.Rank)
            for row in group.itertuples(index=False)
        }
        row: dict[str, float | int | str | None] = {
            "Candidate": candidate,
            "Support": len(ranks),
            "Support Rate": len(ranks) / model_count,
            "Consensus Score": sum(1 / rank for rank in ranks.values()) / model_count,
            "Best Rank": min(ranks.values()),
            "Mean Log Probability": float(group["Log Probability"].mean()),
            "Rank Details": " · ".join(
                f"{model}: #{rank}" for model, rank in ranks.items()
            ),
        }
        for model_name in labels.values():
            row[f"{model_name} Rank"] = ranks.get(model_name)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["Consensus Score", "Support", "Best Rank"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def plot_completion_consensus(
    completions: Mapping[str, list[GenerationCandidate]],
    labels: Mapping[str, str],
    max_candidates: int = 25,
) -> alt.Chart:
    """按 reciprocal-rank consensus 绘制候选榜，颜色表示模型支持率。"""

    if max_candidates <= 0:
        raise ValueError("max_candidates 必须大于 0")
    frame = completion_consensus_summary(completions, labels).head(max_candidates)
    candidate_order = frame["Candidate"].tolist()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Consensus Score:Q",
                title="Reciprocal-Rank Consensus",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("Candidate:N", sort=candidate_order, title=None),
            color=alt.Color(
                "Support Rate:Q",
                title="Model Support",
                scale=alt.Scale(domain=[0, 1], scheme="blues"),
                legend=alt.Legend(format=".0%"),
            ),
            tooltip=[
                "Candidate:N",
                alt.Tooltip("Consensus Score:Q", format=".3f"),
                alt.Tooltip("Support Rate:Q", format=".0%"),
                "Rank Details:N",
                alt.Tooltip("Mean Log Probability:Q", format=".4f"),
            ],
        )
        .properties(title="Completion Consensus", height=max(260, 30 * len(frame)))
    )
