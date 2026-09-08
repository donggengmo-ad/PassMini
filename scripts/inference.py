"""字符级密码模型的加载、评分和候选搜索应用接口。"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .experiment import (
    AutoregressiveBigramConfig,
    AutoregressiveGRUConfig,
    AutoregressiveMLPConfig,
    AutoregressiveModelConfig,
    AutoregressiveTCNConfig,
    AutoregressiveTransformerConfig,
    model_config_from_dict,
)
from .models import (
    AutoregressiveBigram,
    AutoregressiveGRU,
    AutoregressiveMLP,
    AutoregressivePasswordModel,
    AutoregressiveState,
    AutoregressiveTCN,
    AutoregressiveTransformer,
    PasswordModel,
)
from .monitoring import ProgressCallback
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
    if isinstance(config, AutoregressiveMLPConfig):
        return AutoregressiveMLP(
            tokenizer,
            tau=config.tau,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
        )
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

    return score_passwords(
        model,
        tokenizer,
        [password],
        batch_size=1,
        device=device,
        verbose=verbose,
    )[0]


def score_passwords(
    model: PasswordModel,
    tokenizer: CharTokenizer,
    passwords: Sequence[str],
    batch_size: int = 256,
    device: torch.device | str | None = None,
    verbose: bool = False,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[float]:
    r"""按输入顺序批量计算密码 surprisal bits，并可上报批次进度。
    :param model: 实现批量惊讶度接口的密码模型
    :param tokenizer: 与模型一致的字符 tokenizer
    :param passwords: 待评分密码序列
    :param batch_size: 单次模型评分的密码数量
    :param device: 可选设备校验，必须与模型当前设备一致
    :param verbose: 是否向终端打印每个 batch 的进度
    :param progress_callback: 可选实时回调，接收已完成密码数和当前指标
    :return: 与输入密码顺序一致的 surprisal bits 列表
    """

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    values = list(passwords)
    if not all(isinstance(value, str) for value in values):
        raise TypeError("passwords 中的每个元素必须是 str")
    _prepare_inference(model, tokenizer, device)
    results: list[float] = []
    score_sum = 0.0
    start_time = time.perf_counter()
    with torch.inference_mode():
        total_batches = math.ceil(len(values) / batch_size) if values else 0
        for batch_index, start in enumerate(range(0, len(values), batch_size), start=1):
            batch_scores = model.surprisal_bits_batch(values[start : start + batch_size])
            cpu_scores = [float(score) for score in batch_scores.detach().cpu()]
            results.extend(cpu_scores)
            score_sum += sum(cpu_scores)
            if verbose:
                print(
                    f"score batch {batch_index}/{total_batches} "
                    f"completed={len(results)}/{len(values)}"
                )
            if progress_callback is not None:
                elapsed = max(time.perf_counter() - start_time, 1e-12)
                progress_callback(
                    len(results),
                    {
                        "progress": len(results) / len(values),
                        "mean_surprisal_bits": score_sum / len(results),
                        "passwords_per_second": len(results) / elapsed,
                    },
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


@dataclass(frozen=True)
class _BeamNode:
    """保存 beam 前沿中与批量模型状态逐行对应的候选元数据。"""

    token_ids: tuple[int, ...]
    log_probability: float


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


def _masked_log_probs_batch(
    tokenizer: CharTokenizer,
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """把 `[B,V]` raw logits 批量转为温度化 log probability。"""

    if scores.ndim != 2:
        raise ValueError("next logits 必须是 [B,V] 张量")
    values = scores.detach().clone() / temperature
    values[:, [tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id]] = -torch.inf
    if not torch.isfinite(values).any(dim=1).all():
        raise ValueError("模型没有可生成的合法 token")
    return torch.log_softmax(values, dim=-1)


def _masked_log_probs(
    tokenizer: CharTokenizer,
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """把一行 raw logits 转为温度化 log probability，并屏蔽特殊 token。"""

    if scores.ndim != 1:
        raise ValueError("next logits 必须是一维词表分数")
    return _masked_log_probs_batch(tokenizer, scores.unsqueeze(0), temperature)[0]


def _ordered_token_ids(log_probs: torch.Tensor, limit: int | None = None) -> list[int]:
    ids = torch.nonzero(torch.isfinite(log_probs), as_tuple=False).flatten().tolist()
    ids.sort(key=lambda token_id: (-float(log_probs[token_id]), token_id))
    return ids if limit is None else ids[:limit]


def _ordered_token_rows(
    log_probs: torch.Tensor,
    limit: int | None = None,
) -> list[list[tuple[int, float]]]:
    """批量排序下一 token，并只执行一次 device 到 CPU 的同步。"""

    if log_probs.ndim != 2:
        raise ValueError("log_probs 必须是 [B,V] 张量")
    ordered_ids = torch.argsort(log_probs, dim=-1, descending=True, stable=True)
    if limit is not None:
        ordered_ids = ordered_ids[:, :limit]
    ordered_scores = log_probs.gather(1, ordered_ids)
    cpu_ids = ordered_ids.cpu().tolist()
    cpu_scores = ordered_scores.cpu().tolist()
    return [
        [
            (int(token_id), float(score))
            for token_id, score in zip(row_ids, row_scores, strict=True)
            if math.isfinite(score)
        ]
        for row_ids, row_scores in zip(cpu_ids, cpu_scores, strict=True)
    ]


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


def _as_model_state(state: _SearchState) -> AutoregressiveState:
    """把单个堆节点还原为 batch size 为 1 的公共模型状态。"""

    if state.next_logits is None or state.cache is None:
        raise RuntimeError("未完成搜索节点必须保存模型状态")
    return AutoregressiveState(state.next_logits.unsqueeze(0), state.cache)


def _to_candidate(state: _SearchState, tokenizer: CharTokenizer) -> GenerationCandidate:
    return GenerationCandidate(
        text=tokenizer.decode(list(state.token_ids)),
        token_ids=list(state.token_ids),
        log_probability=state.log_probability,
    )


def _expand_search_states(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    states: Sequence[_SearchState],
    temperature: float,
    device: torch.device,
    node_top_k: int | None,
    max_length: int,
    use_cache: bool,
) -> list[list[_SearchState]]:
    """按同深度批量计算下一 token，并展开所有未终止子节点。

    开启缓存时合并节点中的增量状态；关闭时根据 token 前缀重建 `[B,L]`
    输入并执行完整前向。后一种路径不会把模型张量保存在搜索堆中。
    """

    if not states:
        return []
    parent_state: AutoregressiveState | None = None
    if use_cache:
        parent_state = model.stack_states([_as_model_state(state) for state in states])
        next_logits = parent_state.next_logits
    else:
        # 调用方已经按深度分组，因此可以把 BOS + 前缀直接组成无 PAD 的稠密 batch。
        input_ids = torch.tensor(
            [[tokenizer.bos_id, *state.token_ids] for state in states],
            dtype=torch.long,
            device=device,
        )
        next_logits = model(input_ids)[:, -1, :]
    log_probs = _masked_log_probs_batch(tokenizer, next_logits, temperature)
    ordered_rows = _ordered_token_rows(log_probs, node_top_k)
    children: list[list[_SearchState | None]] = [[] for _ in states]
    pending: list[tuple[int, int, tuple[int, ...], float]] = []

    # 先生成纯元数据；只有不会立即结束的子节点才需要消耗 token、更新昂贵缓存。
    for parent_index, (state, ordered_tokens) in enumerate(
        zip(states, ordered_rows, strict=True)
    ):
        for token_id, token_log_probability in ordered_tokens:
            child_ids = state.token_ids + (token_id,)
            child_score = state.log_probability + token_log_probability
            if token_id == tokenizer.eos_id or len(child_ids) >= max_length:
                children[parent_index].append(
                    _SearchState(child_ids, child_score, None, None, True)
                )
                continue
            child_index = len(children[parent_index])
            children[parent_index].append(None)
            pending.append((parent_index, child_index, child_ids, child_score))

    if pending and use_cache:
        assert parent_state is not None
        parent_indices = torch.tensor(
            [item[0] for item in pending], dtype=torch.long, device=device
        )
        token_ids = torch.tensor(
            [item[2][-1] for item in pending], dtype=torch.long, device=device
        )
        repeated_parents = model.select_state(parent_state, parent_indices)
        advanced = model.advance_state(repeated_parents, token_ids)
        # 堆节点必须各自持有 batch size 为 1 的缓存，之后才能独立出堆和重排。
        for row_index, (
            parent_index,
            child_index,
            child_ids,
            child_score,
        ) in enumerate(pending):
            child_state = model.select_state(advanced, [row_index])
            children[parent_index][child_index] = _SearchState(
                child_ids,
                child_score,
                child_state.cache,
                child_state.next_logits[0],
            )
    elif pending:
        # 无缓存节点只保存搜索元数据；下次出堆时再由完整前缀重算 logits。
        for parent_index, child_index, child_ids, child_score in pending:
            children[parent_index][child_index] = _SearchState(
                child_ids,
                child_score,
                None,
                None,
            )

    return [[child for child in row if child is not None] for row in children]


def beam_search(
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
    prefix: str,
    beam_width: int = 5,
    max_length: int = 12,
    temperature: float = 1.0,
    verbose: bool = False,
) -> list[GenerationCandidate]:
    r"""按逐轮 beam 保留策略批量补全给定前缀。

    每一深度将整个活跃前沿保存为一个 `[B,...]` 模型状态，并批量计算
    `[B,V]` 下一 token 分数。算法先完成全局 beam 剪枝，再复制被保留的父
    缓存，并用一次 `advance_state()` 前推全部未终止子节点。因此它与逐节点
    beam 的保留规则和确定性排序一致，但避免为最终被剪掉的分支运行模型。

    副作用是中间 logits、父索引和缓存复制量随 `beam_width` 增长，极大束宽
    会提高峰值显存；模型的状态转移还必须满足无隐式副作用的公共接口约定。
    """

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
        active = [_BeamNode(initial.token_ids, initial.log_probability)]
        active_state = _as_model_state(initial)
        completed: list[_SearchState] = []
        depth = len(prefix_ids)
        while active:
            log_probs = _masked_log_probs_batch(
                tokenizer, active_state.next_logits, temperature
            )
            ordered_rows = _ordered_token_rows(log_probs, beam_width)
            expanded: list[tuple[_BeamNode, int, int]] = []
            for parent_index, (state, ordered_tokens) in enumerate(
                zip(active, ordered_rows, strict=True)
            ):
                for token_id, token_log_probability in ordered_tokens:
                    child_ids = state.token_ids + (token_id,)
                    child_score = state.log_probability + token_log_probability
                    if token_id == tokenizer.eos_id or len(child_ids) >= max_length:
                        completed.append(
                            _SearchState(child_ids, child_score, None, None, True)
                        )
                    else:
                        expanded.append(
                            (_BeamNode(child_ids, child_score), parent_index, token_id)
                        )
            expanded.sort(
                key=lambda item: (-item[0].log_probability, item[0].token_ids)
            )
            retained = expanded[:beam_width]
            if retained:
                # 全局剪枝后再批量推进，只为最终进入下一层 beam 的节点计算缓存。
                parent_indices = torch.tensor(
                    [item[1] for item in retained], dtype=torch.long, device=device
                )
                token_ids = torch.tensor(
                    [item[2] for item in retained], dtype=torch.long, device=device
                )
                selected_parents = model.select_state(active_state, parent_indices)
                active_state = model.advance_state(selected_parents, token_ids)
                active = [item[0] for item in retained]
            else:
                active = []
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
    expansion_batch_size: int = 1,
    use_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[GenerationCandidate]:
    r"""从 BOS 开始执行全局 best-first 搜索。

    节点优先级公式为 `priority(s) = log_probability(s) /
    max(1, len(s)) ** length_penalty`。`length_penalty=0` 时使用累计 log
    probability；正值只改变未完成节点出堆顺序，返回值仍按原始累计分数排序。
    `node_top_k` 限制单节点分支数；`depth_beam_width` 限制同一深度实际扩展的
    未完成节点数，超出后直接丢弃。

    `expansion_batch_size` 控制每轮从当前堆快照连续弹出的节点数。弹出节点按
    深度分组，同深度缓存合并后批量计算子节点；`1` 保持严格 best-first 出堆
    顺序，同时仍会批量前推一个父节点的兄弟分支。大于 `1` 可以增加 GPU
    利用率，但在整批子节点重新入堆之前不会重新比较优先级：第一个父节点的
    高分子节点可能本应早于本批后续父节点出堆。因此它是显式的近似加速，可能
    改变候选集合、候选完成顺序及 `depth_beam_width` 的剪枝结果，批量越大峰值
    显存也越高。返回值仍按原始累计 log probability 排序。

    `use_cache=True` 默认让每个未完成堆节点保存增量模型状态：Bigram、MLP 和
    GRU 状态大小固定；TCN 历史缓存随感受野和队列宽度增长，Transformer KV
    cache 还会随前缀长度增长。设为 `False` 后，堆节点只保存 CPU 搜索元数据，
    出堆时再按同深度组成完整前缀 batch 并调用 `forward()`。这能使模型状态显存
    不再随队列增长，适合大预算覆盖率评测；副作用是重复计算历史 token，搜索
    速度通常会下降。两条路径使用相同概率和剪枝规则，但浮点计算顺序不同可能
    带来极小数值差异。

    `progress_callback` 约每 1000 个出堆节点以及任务结束时调用一次，横轴
    step 是累计出堆节点数；指标包含目标候选完成比例、完成数、队列长度和
    生成速度。批量模式下单次可能跨过整千边界，因此回调 step 不保证恰为整千。
    """

    if num_candidates <= 0:
        raise ValueError("num_candidates 必须大于 0")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache 必须是 bool")
    _validate_search_args(max_length, temperature)
    if length_penalty < 0 or not math.isfinite(length_penalty):
        raise ValueError("length_penalty 必须是有限非负数")
    for value, name in (
        (node_top_k, "node_top_k"),
        (depth_beam_width, "depth_beam_width"),
        (expansion_batch_size, "expansion_batch_size"),
    ):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} 必须是正整数")
    device = _prepare_inference(model, tokenizer, None)
    with torch.inference_mode():
        initial = (
            _initial_state(model, tokenizer, [], device, temperature)
            if use_cache
            else _SearchState((), 0.0, None, None)
        )
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
        last_reported = 0
        last_verbose = 0
        start_time = time.perf_counter()

        def report_progress(*, force: bool = False) -> None:
            """按固定出堆间隔上报搜索状态，结束时可强制补写最后一点。"""

            nonlocal last_reported
            if progress_callback is None or (not force and popped - last_reported < 1000):
                return
            elapsed = max(time.perf_counter() - start_time, 1e-12)
            progress_callback(
                popped,
                {
                    "candidate_progress": len(candidates) / num_candidates,
                    "completed_candidates": len(candidates),
                    "queue_size": len(queue),
                    "candidates_per_second": len(candidates) / elapsed,
                },
            )
            last_reported = popped

        while queue and len(candidates) < num_candidates:
            # 连续弹出的是同一堆快照中的最高优先级节点；本轮结束后子节点才回堆。
            popped_states = [heapq.heappop(queue)[2]]
            while queue and len(popped_states) < expansion_batch_size:
                popped_states.append(heapq.heappop(queue)[2])
            popped += len(popped_states)
            expandable: list[tuple[int, _SearchState]] = []
            for pop_index, state in enumerate(popped_states):
                if state.terminal:
                    candidate = _to_candidate(state, tokenizer)
                    if candidate.text not in seen:
                        seen.add(candidate.text)
                        candidates.append(candidate)
                    if len(candidates) >= num_candidates:
                        break
                    continue
                depth = len(state.token_ids)
                if depth_beam_width is not None:
                    used = expanded_by_depth.get(depth, 0)
                    if used >= depth_beam_width:
                        continue
                    expanded_by_depth[depth] = used + 1
                expandable.append((pop_index, state))

            if len(candidates) < num_candidates and expandable:
                # 缓存模式只有同深度 Transformer 状态可合并；无缓存模式也需要同长
                # 前缀组成稠密 batch。计算完成后仍按出堆次序恢复确定性的 tie-break。
                grouped: dict[int, list[tuple[int, _SearchState]]] = {}
                for pop_index, state in expandable:
                    grouped.setdefault(len(state.token_ids), []).append((pop_index, state))
                expanded_by_index: dict[int, list[_SearchState]] = {}
                for group in grouped.values():
                    group_children = _expand_search_states(
                        model,
                        tokenizer,
                        [state for _, state in group],
                        temperature,
                        device,
                        node_top_k,
                        max_length,
                        use_cache,
                    )
                    for (pop_index, _), children in zip(group, group_children, strict=True):
                        expanded_by_index[pop_index] = children
                for pop_index, _ in expandable:
                    for child in expanded_by_index[pop_index]:
                        push(child)
            if verbose and popped - last_verbose >= 1000:
                print(f"best-first popped={popped} completed={len(candidates)} queue={len(queue)}")
                last_verbose = popped
            report_progress()
        candidates.sort(key=lambda item: (-item.log_probability, item.token_ids))
        if verbose:
            print(f"best-first popped={popped} completed={len(candidates)} queue={len(queue)}")
        if progress_callback is not None and popped != last_reported:
            report_progress(force=True)
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
