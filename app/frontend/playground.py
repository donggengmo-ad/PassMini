"""Playground 的多模型评分、生成和补全编排。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
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


def _generation_probabilities(
    model: AutoregressivePasswordModel,
    logits: torch.Tensor,
) -> torch.Tensor:
    """将一步 logits 转为生成概率，并屏蔽不可生成的特殊 token。"""

    if logits.shape != (1, model.vocab_size):
        raise ValueError("next_logits 必须是 [1, vocab_size] 张量")
    scores = logits.detach().clone()
    invalid_ids = [model.pad_id, model.tokenizer.bos_id, model.tokenizer.unk_id]
    scores[:, invalid_ids] = -torch.inf
    if not torch.isfinite(scores).any():
        raise ValueError("模型没有可展示的下一 token")
    return torch.softmax(scores, dim=-1)


def trace_character_probabilities(
    text: str,
    models: Mapping[str, RuntimeModel],
    top_k: int = 5,
    max_length: int = 12,
) -> list[ModelProbabilityTrace]:
    """逐字符计算条件概率，并给出当前前缀之后的 Top-K token。

    每个模型只消费一次 BOS，随后通过统一增量状态逐字符推进。第 i 个字符
    的概率始终来自它出现之前的前缀，所以继续追加字符不会改变已经记录的
    颜色。PAD、BOS、UNK 不属于可生成候选；EOS 保留并显示为结束标记。
    """

    if normalize_password(text, min_length=0, max_length=max_length) != text:
        raise ValueError(f"请输入 0–{max_length} 个可打印 ASCII 字符")
    if not models:
        raise ValueError("至少需要一个模型")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    traces: list[ModelProbabilityTrace] = []
    with torch.inference_mode():
        for model_id, (model, tokenizer) in models.items():
            if not isinstance(model, AutoregressivePasswordModel):
                raise TypeError(f"{model_id} 不支持自回归逐字符预测")
            if model.tokenizer != tokenizer:
                raise ValueError(f"{model_id} 的模型与 tokenizer 不一致")

            model.eval()
            state = model.initial_state(batch_size=1)
            character_probabilities: list[CharacterProbability] = []
            for character in text:
                token_id = tokenizer.token_to_id.get(character)
                if token_id is None or tokenizer.is_special_token(token_id):
                    raise ValueError(f"字符 {character!r} 不在 {model_id} 的可用词表中")
                probabilities = _generation_probabilities(model, state.next_logits)
                character_probabilities.append(
                    CharacterProbability(
                        character=character,
                        probability=float(probabilities[0, token_id].cpu()),
                    )
                )
                token = torch.tensor([token_id], dtype=torch.long, device=model.device)
                state = model.advance_state(state, token)

            next_probabilities = _generation_probabilities(model, state.next_logits)[0]
            candidate_count = min(top_k, tokenizer.vocab_size - 3)
            values, token_ids = torch.topk(next_probabilities, k=candidate_count)
            next_tokens = tuple(
                NextTokenPrediction(
                    token="[EOS]" if token_id == tokenizer.eos_id else tokenizer.id_to_token[token_id],
                    probability=float(probability),
                    is_eos=token_id == tokenizer.eos_id,
                )
                for probability, token_id in zip(
                    values.detach().cpu().tolist(),
                    token_ids.detach().cpu().tolist(),
                )
            )
            traces.append(
                ModelProbabilityTrace(
                    model_id=model_id,
                    characters=tuple(character_probabilities),
                    next_tokens=next_tokens,
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
