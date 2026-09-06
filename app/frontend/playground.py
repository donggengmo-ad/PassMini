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
