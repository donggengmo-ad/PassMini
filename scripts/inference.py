"""字符级密码模型的加载、评分和候选搜索应用接口。"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .experiment import (
    AutoregressiveBigramConfig,
    AutoregressiveGRUConfig,
    AutoregressiveModelConfig,
    AutoregressiveTCNConfig,
    AutoregressiveTransformerConfig,
    model_config_from_dict,
)
from .models import (
    AutoregressiveBigram,
    AutoregressiveGRU,
    AutoregressivePasswordModel,
    AutoregressiveState,
    AutoregressiveTCN,
    AutoregressiveTransformer,
    PasswordModel,
)
from .tokenizer import CharTokenizer


InferenceModelConfigType = AutoregressiveModelConfig


def _read_json(path: str | Path) -> dict:
    """读取 JSON 对象并拒绝非对象顶层数据。"""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    return value


def _model_payload(config: Mapping) -> Mapping:
    """读取嵌套实验配置中的 model 子树或扁平模型配置。"""

    nested = config.get("model")
    return nested if isinstance(nested, Mapping) else config


def _parse_model_config(config: Mapping, tokenizer: CharTokenizer) -> InferenceModelConfigType:
    """按 canonical model_type 解析模型结构配置。"""

    payload = dict(_model_payload(config))
    payload.setdefault("model_type", None)
    parsed = model_config_from_dict(payload)
    configured_vocab = payload.get("vocab_size")
    if configured_vocab is not None and int(configured_vocab) != tokenizer.vocab_size:
        raise ValueError("inference config 的 vocab_size 与 tokenizer.vocab_size 不一致")
    return parsed


def _build_model(config: InferenceModelConfigType, tokenizer: CharTokenizer) -> AutoregressivePasswordModel:
    """按配置构造具体模型，不在调用层判断模型类型。"""

    if isinstance(config, AutoregressiveBigramConfig):
        return AutoregressiveBigram(tokenizer, alpha=config.alpha)
    if isinstance(config, AutoregressiveGRUConfig):
        return AutoregressiveGRU(
            tokenizer,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )
    if isinstance(config, AutoregressiveTCNConfig):
        return AutoregressiveTCN(
            tokenizer,
            embedding_dim=config.embedding_dim,
            channels=config.channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
        )
    if isinstance(config, AutoregressiveTransformerConfig):
        return AutoregressiveTransformer(
            tokenizer,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            max_length=config.max_length,
        )
    raise TypeError(f"不支持的模型配置类型: {type(config).__name__}")


def load_inference_model(
    model_path: str | Path,
    tokenizer_path: str | Path,
    config_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[PasswordModel, CharTokenizer]:
    r"""加载独立推理 artifact。

    配置只接受 canonical 模型标签。Bigram artifact 本身包含 count 和 tokenizer，
    但仍校验外部 tokenizer 与其一致；神经模型使用 `state_dict` 加载参数。
    """

    target_device = torch.device(device)
    tokenizer = CharTokenizer.from_json(Path(tokenizer_path))
    config = _parse_model_config(_read_json(config_path), tokenizer)
    if isinstance(config, AutoregressiveBigramConfig):
        model = AutoregressiveBigram.load(model_path)
        if model.tokenizer != tokenizer:
            raise ValueError("Bigram artifact 与 tokenizer.json 不一致")
        if model.alpha != config.alpha:
            raise ValueError("Bigram artifact 与 inference.json 的 alpha 不一致")
        return model, tokenizer

    model = _build_model(config, tokenizer)
    state_dict = torch.load(model_path, map_location=target_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model, tokenizer


def _prepare_inference(
    model: PasswordModel,
    tokenizer: CharTokenizer,
    device: torch.device | str | None,
) -> torch.device:
    """校验 tokenizer 和设备，返回模型实际运行设备。"""

    if model.tokenizer != tokenizer:
        raise ValueError("model 使用的 tokenizer 与传入 tokenizer 不一致")
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model.vocab_size 与 tokenizer.vocab_size 不一致")
    target = model.device if device is None else torch.device(device)
    if target != model.device:
        raise ValueError("传入的 device 必须与 model 当前所在设备一致")
    if isinstance(model, torch.nn.Module):
        model.eval()
    return target


def score_password(
    model: PasswordModel,
    tokenizer: CharTokenizer,
    password: str,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> float:
    """计算单条密码的 surprisal bits。"""

    return score_passwords(model, tokenizer, [password], 1, device, verbose)[0]


def score_passwords(
    model: PasswordModel,
    tokenizer: CharTokenizer,
    passwords: Sequence[str],
    batch_size: int = 256,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> list[float]:
    """按输入顺序批量计算密码 surprisal bits。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    values = list(passwords)
    if not all(isinstance(value, str) for value in values):
        raise TypeError("passwords 中的每个元素必须是 str")
    _prepare_inference(model, tokenizer, device)
    results: list[float] = []
    with torch.inference_mode():
        total_batches = math.ceil(len(values) / batch_size) if values else 0
        for batch_index, start in enumerate(range(0, len(values), batch_size), start=1):
            batch_scores = model.surprisal_bits_batch(values[start : start + batch_size])
            results.extend(float(score) for score in batch_scores.detach().cpu())
            if verbose:
                print(
                    f"score batch {batch_index}/{total_batches} "
                    f"completed={len(results)}/{len(values)}"
                )
    return results


def inference_config_dict(config: InferenceModelConfigType) -> dict:
    """将模型配置转换为 JSON 字典。"""

    return asdict(config)


def save_inference_config(path: str | Path, config: InferenceModelConfigType) -> None:
    """保存独立推理配置，文件名通常为 `inference.json`。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inference_config_dict(config), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class GenerationCandidate:
    """保存一条搜索候选及其累计 log probability。"""

    text: str
    token_ids: list[int]
    log_probability: float


def save_generation_candidates(path: str | Path, candidates: Sequence[GenerationCandidate]) -> None:
    """原子保存候选 JSON。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"text": c.text, "token_ids": list(c.token_ids), "log_probability": float(c.log_probability)}
        for c in candidates
    ]
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary_path = file.name
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise


def load_generation_candidates(path: str | Path) -> list[GenerationCandidate]:
    """加载并校验候选 JSON。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("候选文件顶层必须是 JSON 数组")
    result = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {"text", "token_ids", "log_probability"}:
            raise ValueError(f"第 {index} 条候选字段不完整")
        if not isinstance(item["text"], str) or not isinstance(item["token_ids"], list):
            raise ValueError(f"第 {index} 条候选字段类型非法")
        if not all(isinstance(token_id, int) for token_id in item["token_ids"]):
            raise ValueError(f"第 {index} 条候选的 token_ids 必须是整数数组")
        score = item["log_probability"]
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"第 {index} 条候选的 log_probability 必须是有限数值")
        result.append(GenerationCandidate(item["text"], item["token_ids"], float(score)))
    return result


@dataclass
class _SearchState:
    """保存不可变 token 前缀、累计分数和模型状态。"""

    token_ids: tuple[int, ...]
    log_probability: float
    cache: object
    next_logits: torch.Tensor | None
    terminal: bool = False


@dataclass
class _DepthFirstFrame:
    """保存 DFS 栈帧和已排序的兄弟分支。"""

    state: _SearchState
    next_log_probs: torch.Tensor | None = None
    next_token_ids: list[int] | None = None
    next_index: int = 0


def _validate_search_args(max_length: int, temperature: float) -> None:
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature 必须是有限正数")


def _masked_log_probs(tokenizer: CharTokenizer, scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """把一步 raw logits 转为温度化 log probability，并屏蔽特殊 token。"""

    if scores.ndim != 1:
        raise ValueError("next logits 必须是一维词表分数")
    values = scores.detach().clone() / temperature
    values[[tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = -torch.inf
    if not torch.isfinite(values).any():
        raise ValueError("模型没有可生成的合法 token")
    return torch.log_softmax(values, dim=-1)


def _ordered_token_ids(log_probs: torch.Tensor, limit: int | None = None) -> list[int]:
    ids = torch.nonzero(torch.isfinite(log_probs), as_tuple=False).flatten().tolist()
    ids.sort(key=lambda token_id: (-float(log_probs[token_id]), token_id))
    return ids if limit is None else ids[:limit]


def _initial_state(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    prefix_ids: list[int],
    device: torch.device,
    temperature: float,
) -> _SearchState:
    """通过公共状态接口消费 prefix，并累计其条件概率。"""

    state = model.initial_state(1)
    score = 0.0
    for token_id in prefix_ids:
        log_probs = _masked_log_probs(tokenizer, state.next_logits[0], temperature)
        score += float(log_probs[token_id])
        token = torch.tensor([token_id], dtype=torch.long, device=device)
        state = model.advance_state(state, token)
    return _SearchState(tuple(prefix_ids), score, state.cache, state.next_logits[0])


def _advance_state(
    model: AutoregressivePasswordModel,
    state: _SearchState,
    token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, object]:
    """调用统一状态接口消费一个 token。"""

    parent = AutoregressiveState(state.next_logits.unsqueeze(0), state.cache)
    token = torch.tensor([token_id], dtype=torch.long, device=device)
    child = model.advance_state(parent, token)
    return child.next_logits[0], child.cache


def _to_candidate(state: _SearchState, tokenizer: CharTokenizer) -> GenerationCandidate:
    return GenerationCandidate(
        text=tokenizer.decode(list(state.token_ids)),
        token_ids=list(state.token_ids),
        log_probability=state.log_probability,
    )


def _expand_search_state(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    state: _SearchState,
    temperature: float,
    device: torch.device,
    node_top_k: int | None,
    max_length: int,
) -> list[_SearchState]:
    """展开一个节点，生成终止或未完成子节点。"""

    if state.next_logits is None:
        raise RuntimeError("未完成搜索节点必须保存 next logits")
    log_probs = _masked_log_probs(tokenizer, state.next_logits, temperature)
    children: list[_SearchState] = []
    for token_id in _ordered_token_ids(log_probs, node_top_k):
        child_ids = state.token_ids + (token_id,)
        child_score = state.log_probability + float(log_probs[token_id])
        if token_id == tokenizer.eos_id or len(child_ids) >= max_length:
            children.append(_SearchState(child_ids, child_score, None, None, True))
            continue
        next_logits, cache = _advance_state(model, state, token_id, device)
        children.append(_SearchState(child_ids, child_score, cache, next_logits))
    return children


def beam_search(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    prefix: str,
    beam_width: int = 5,
    max_length: int = 12,
    temperature: float = 1.0,
    verbose: bool = False,
) -> list[GenerationCandidate]:
    """按逐轮 beam 保留策略补全给定前缀。"""

    if not isinstance(prefix, str):
        raise TypeError("prefix 必须是 str")
    if beam_width <= 0:
        raise ValueError("beam_width 必须大于 0")
    _validate_search_args(max_length, temperature)
    device = _prepare_inference(model, tokenizer, None)
    with torch.inference_mode():
        prefix_ids = tokenizer.encode(prefix)
        if len(prefix_ids) > max_length:
            raise ValueError("prefix 长度不能超过 max_length")
        initial = _initial_state(model, tokenizer, prefix_ids, device, temperature)
        if len(prefix_ids) >= max_length:
            return [_to_candidate(initial, tokenizer)]
        active = [initial]
        completed: list[_SearchState] = []
        depth = len(prefix_ids)
        while active:
            expanded: list[_SearchState] = []
            for state in active:
                if state.next_logits is None:
                    continue
                log_probs = _masked_log_probs(tokenizer, state.next_logits, temperature)
                for token_id in _ordered_token_ids(log_probs, beam_width):
                    child_ids = state.token_ids + (token_id,)
                    child_score = state.log_probability + float(log_probs[token_id])
                    if token_id == tokenizer.eos_id or len(child_ids) >= max_length:
                        completed.append(_SearchState(child_ids, child_score, None, None, True))
                    else:
                        next_logits, cache = _advance_state(model, state, token_id, device)
                        expanded.append(_SearchState(child_ids, child_score, cache, next_logits))
            expanded.sort(key=lambda item: (-item.log_probability, item.token_ids))
            active = expanded[:beam_width]
            depth += 1
            if verbose:
                print(f"beam depth={depth} active={len(active)} completed={len(completed)}")
        completed.sort(key=lambda item: (-item.log_probability, item.token_ids))
        result: list[GenerationCandidate] = []
        seen: set[str] = set()
        for state in completed:
            candidate = _to_candidate(state, tokenizer)
            if candidate.text in seen:
                continue
            seen.add(candidate.text)
            result.append(candidate)
            if len(result) >= beam_width:
                break
        return result


def _length_normalized_score(state: _SearchState, length_penalty: float) -> float:
    return state.log_probability / (max(1, len(state.token_ids)) ** length_penalty)


def best_first_search(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    num_candidates: int = 1000,
    max_length: int = 12,
    temperature: float = 1.0,
    verbose: bool = False,
    save_path: str | Path | None = None,
    length_penalty: float = 0.0,
    node_top_k: int | None = None,
    depth_beam_width: int | None = None,
) -> list[GenerationCandidate]:
    r"""从 BOS 开始执行全局 best-first 搜索。

    节点优先级公式为 `priority(s) = log_probability(s) /
    max(1, len(s)) ** length_penalty`。`length_penalty=0` 时使用累计 log
    probability；正值只改变未完成节点出堆顺序，返回值仍按原始累计分数排序。
    `node_top_k` 限制单节点分支数；`depth_beam_width` 限制同一深度实际扩展的
    未完成节点数，超出后直接丢弃，不改变全局堆调度。
    """

    if num_candidates <= 0:
        raise ValueError("num_candidates 必须大于 0")
    _validate_search_args(max_length, temperature)
    if length_penalty < 0 or not math.isfinite(length_penalty):
        raise ValueError("length_penalty 必须是有限非负数")
    for value, name in ((node_top_k, "node_top_k"), (depth_beam_width, "depth_beam_width")):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} 必须是正整数或 None")
    device = _prepare_inference(model, tokenizer, None)
    with torch.inference_mode():
        initial = _initial_state(model, tokenizer, [], device, temperature)
        queue: list[tuple[float, int, _SearchState]] = []
        counter = itertools.count()
        push = lambda state: heapq.heappush(
            queue, (-_length_normalized_score(state, length_penalty), next(counter), state)
        )
        push(initial)
        candidates: list[GenerationCandidate] = []
        seen: set[str] = set()
        expanded_by_depth: dict[int, int] = {}
        popped = 0
        while queue and len(candidates) < num_candidates:
            _, _, state = heapq.heappop(queue)
            popped += 1
            if state.terminal:
                candidate = _to_candidate(state, tokenizer)
                if candidate.text not in seen:
                    seen.add(candidate.text)
                    candidates.append(candidate)
                continue
            depth = len(state.token_ids)
            if depth_beam_width is not None:
                used = expanded_by_depth.get(depth, 0)
                if used >= depth_beam_width:
                    continue
                expanded_by_depth[depth] = used + 1
            for child in _expand_search_state(
                model, tokenizer, state, temperature, device, node_top_k, max_length
            ):
                push(child)
            if verbose and popped % 1000 == 0:
                print(f"best-first popped={popped} completed={len(candidates)} queue={len(queue)}")
        candidates.sort(key=lambda item: (-item.log_probability, item.token_ids))
        if verbose:
            print(f"best-first popped={popped} completed={len(candidates)} queue={len(queue)}")
        if save_path is not None:
            save_generation_candidates(save_path, candidates)
        return candidates


def depth_first_search(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    num_candidates: int = 1000,
    max_length: int = 12,
    temperature: float = 1.0,
    verbose: bool = False,
    save_path: str | Path | None = None,
) -> list[GenerationCandidate]:
    """按局部概率优先、使用显式栈回溯生成候选。"""

    if num_candidates <= 0:
        raise ValueError("num_candidates 必须大于 0")
    _validate_search_args(max_length, temperature)
    device = _prepare_inference(model, tokenizer, None)
    with torch.inference_mode():
        stack = [_DepthFirstFrame(_initial_state(model, tokenizer, [], device, temperature))]
        candidates: list[GenerationCandidate] = []
        seen: set[str] = set()
        expanded = 0
        while stack and len(candidates) < num_candidates:
            frame = stack[-1]
            state = frame.state
            if frame.next_token_ids is None:
                if state.next_logits is None:
                    stack.pop()
                    continue
                frame.next_log_probs = _masked_log_probs(tokenizer, state.next_logits, temperature)
                frame.next_token_ids = _ordered_token_ids(frame.next_log_probs)
            if frame.next_index >= len(frame.next_token_ids):
                stack.pop()
                continue
            token_id = frame.next_token_ids[frame.next_index]
            frame.next_index += 1
            assert frame.next_log_probs is not None
            child_ids = state.token_ids + (token_id,)
            child_score = state.log_probability + float(frame.next_log_probs[token_id])
            expanded += 1
            if token_id == tokenizer.eos_id or len(child_ids) >= max_length:
                candidate = _to_candidate(_SearchState(child_ids, child_score, None, None, True), tokenizer)
                if candidate.text not in seen:
                    seen.add(candidate.text)
                    candidates.append(candidate)
            else:
                next_logits, cache = _advance_state(model, state, token_id, device)
                stack.append(_DepthFirstFrame(_SearchState(child_ids, child_score, cache, next_logits)))
            if verbose and expanded % 1000 == 0:
                print(f"depth-first expanded={expanded} completed={len(candidates)} depth={len(stack)}")
        if verbose:
            print(f"depth-first expanded={expanded} completed={len(candidates)} depth={len(stack)}")
        if save_path is not None:
            save_generation_candidates(save_path, candidates)
        return candidates


__all__ = [
    "InferenceModelConfigType",
    "GenerationCandidate",
    "load_inference_model",
    "score_password",
    "score_passwords",
    "inference_config_dict",
    "save_inference_config",
    "save_generation_candidates",
    "load_generation_candidates",
    "beam_search",
    "best_first_search",
    "depth_first_search",
]
