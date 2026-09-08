"""计算模型单次完整前向的 Estimated FLOPs，并同步项目元数据。

统一计算口径为 batch size 1、输入序列长度 13，输出形状为
``[1, 13, vocab_size]`` 的 logits。一次标量乘法、加法、除法、比较或
非线性函数均记为 1 FLOP；Embedding 查询和纯内存操作不计入。

这里采用与当前四种模型源码对应的解析式，而不是依赖只统计 ``nn.Module``
的通用摘要工具。这样能够覆盖权重共享输出矩阵、GRU 门控、因果注意力、
LayerNorm 和主要逐元素运算。结果仍是与上述口径绑定的 Estimated FLOPs，
不等同于特定 GPU 实际执行的硬件指令数。

运行命令
```bash
conda run --no-capture-output -n passmini \
  python utils/calculate_model_flops.py
```
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "data" / "processed" / "tokenizer.json"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "app" / "artifacts" / "catalog.json"
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQUENCE_LENGTH = 13


def _positive_int(config: Mapping[str, object], key: str) -> int:
    """从模型配置中读取正整数，发现缺失或错误时尽早失败。"""

    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} 必须是正整数")
    return value


def estimate_bigram_flops(vocab_size: int, sequence_length: int) -> int:
    """估算 Bigram 为每个位置构造完整 log-count logits 的 FLOPs。"""

    # 每个词表元素执行 add-alpha 和 log；索引计数行属于内存操作，不计 FLOP。
    return sequence_length * vocab_size * 2


def estimate_mlp_flops(
    vocab_size: int,
    sequence_length: int,
    tau: int,
    embedding_dim: int,
    hidden_size: int,
) -> int:
    """估算固定上下文、权重共享 MLP 的完整序列前向 FLOPs。"""

    # 每个位置同时计算输入投影、GELU、嵌入空间投影和共享词表投影。
    input_projection = 2 * tau * embedding_dim * hidden_size
    activation = hidden_size
    output_projection = 2 * hidden_size * embedding_dim
    tied_vocabulary_projection = vocab_size * (2 * embedding_dim - 1)
    return sequence_length * (
        input_projection
        + activation
        + output_projection
        + tied_vocabulary_projection
    )


def estimate_gru_flops(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int,
    hidden_size: int,
    num_layers: int,
) -> int:
    """估算固定权重共享 GRU 的完整序列前向 FLOPs。

    每层、每时间步包含输入和循环两组三门仿射变换，以及 reset/update/new
    门的逐元素计算；随后统计隐藏投影和与 embedding 权重共享的词表投影。
    """

    recurrent = 0
    for layer_index in range(num_layers):
        input_size = embedding_dim if layer_index == 0 else hidden_size
        # 两组带 bias 的三门仿射变换分别为 6*I*H 和 6*H*H；门控本身为 10*H。
        recurrent += 6 * input_size * hidden_size
        recurrent += 6 * hidden_size * hidden_size
        recurrent += 10 * hidden_size

    output_projection = 2 * hidden_size * embedding_dim
    tied_vocabulary_projection = vocab_size * (2 * embedding_dim - 1)
    return sequence_length * (
        recurrent + output_projection + tied_vocabulary_projection
    )


def estimate_tcn_flops(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int,
    channels: int,
    kernel_size: int,
    num_layers: int,
) -> int:
    """估算权重共享因果 TCN 的完整序列前向 FLOPs。"""

    convolutions = 0
    for layer_index in range(num_layers):
        input_channels = embedding_dim if layer_index == 0 else channels
        # 带 bias 的每个卷积输出执行 Cin*K 次乘法和同等数量的累加。
        convolutions += sequence_length * channels * 2 * input_channels * kernel_size
        convolutions += sequence_length * channels  # ReLU

    output_projection = sequence_length * 2 * channels * embedding_dim
    tied_vocabulary_projection = (
        sequence_length * vocab_size * (2 * embedding_dim - 1)
    )
    return convolutions + output_projection + tied_vocabulary_projection


def estimate_transformer_flops(
    vocab_size: int,
    sequence_length: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
) -> int:
    """估算权重共享 decoder-only 因果 Transformer 的完整前向 FLOPs。

    注意力只统计因果下三角中的有效 Query-Key 对。除 QKV、注意力输出和
    FFN 矩阵乘法外，还统计缩放、softmax、残差、LayerNorm 与 ReLU。
    """

    head_dim = d_model // nhead
    causal_pairs = sequence_length * (sequence_length + 1) // 2

    qkv_projection = 6 * sequence_length * d_model * d_model
    attention_scores = nhead * causal_pairs * (2 * head_dim - 1)
    attention_scaling = nhead * causal_pairs
    # 长度为 t 的 softmax：max、减最大值、exp、求和、除法共 5t-2 次运算。
    attention_softmax = nhead * (5 * causal_pairs - 2 * sequence_length)
    weighted_values = d_model * (2 * causal_pairs - sequence_length)
    attention_output = 2 * sequence_length * d_model * d_model

    feedforward = 4 * sequence_length * d_model * dim_feedforward
    feedforward_relu = sequence_length * dim_feedforward
    residual_additions = 2 * sequence_length * d_model
    # 一次 LayerNorm 约为 7D+2：均值、方差、归一化以及可学习仿射变换。
    layer_norms = 2 * sequence_length * (7 * d_model + 2)

    per_layer = (
        qkv_projection
        + attention_scores
        + attention_scaling
        + attention_softmax
        + weighted_values
        + attention_output
        + feedforward
        + feedforward_relu
        + residual_additions
        + layer_norms
    )
    position_addition = sequence_length * d_model
    tied_vocabulary_projection = sequence_length * vocab_size * (2 * d_model - 1)
    return (
        num_layers * per_layer
        + position_addition
        + tied_vocabulary_projection
    )


def estimate_model_flops(
    config: Mapping[str, object],
    vocab_size: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> int:
    """根据 canonical ``model_type`` 分派到对应架构的解析式。"""

    if vocab_size <= 0 or sequence_length <= 0:
        raise ValueError("vocab_size 和 sequence_length 必须大于 0")
    model_type = config.get("model_type")
    if model_type == "bigram":
        return estimate_bigram_flops(vocab_size, sequence_length)
    if model_type == "mlp":
        return estimate_mlp_flops(
            vocab_size,
            sequence_length,
            _positive_int(config, "tau"),
            _positive_int(config, "embedding_dim"),
            _positive_int(config, "hidden_size"),
        )
    if model_type == "gru":
        return estimate_gru_flops(
            vocab_size,
            sequence_length,
            _positive_int(config, "embedding_dim"),
            _positive_int(config, "hidden_size"),
            _positive_int(config, "num_layers"),
        )
    if model_type == "tcn":
        dilations = config.get("dilations")
        if not isinstance(dilations, list) or not dilations:
            raise ValueError("dilations 必须是非空列表")
        if any(not isinstance(value, int) or value <= 0 for value in dilations):
            raise ValueError("dilations 必须只包含正整数")
        return estimate_tcn_flops(
            vocab_size,
            sequence_length,
            _positive_int(config, "embedding_dim"),
            _positive_int(config, "channels"),
            _positive_int(config, "kernel_size"),
            len(dilations),
        )
    if model_type == "transformer":
        d_model = _positive_int(config, "d_model")
        nhead = _positive_int(config, "nhead")
        max_length = _positive_int(config, "max_length")
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        if sequence_length > max_length:
            raise ValueError("sequence_length 不能超过 Transformer max_length")
        return estimate_transformer_flops(
            vocab_size,
            sequence_length,
            d_model,
            nhead,
            _positive_int(config, "num_layers"),
            _positive_int(config, "dim_feedforward"),
        )
    raise ValueError(f"不支持的 model_type: {model_type!r}")


def _read_json_object(path: Path) -> dict:
    """读取顶层为对象的 JSON。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 的 JSON 顶层必须是对象")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    """在目标目录写临时文件，再原子替换正式 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def discover_model_tier_paths(project_root: Path) -> list[Path]:
    """发现仓库内全部 ``model_tiers.json``，忽略隐藏工具目录。"""

    return sorted(
        path
        for path in project_root.rglob("model_tiers.json")
        if not any(part.startswith(".") for part in path.relative_to(project_root).parts)
    )


def _replace_compute_field(config: Mapping[str, object], flops: int) -> dict:
    """保持原字段顺序，将旧 ``mult_adds`` 迁移为唯一的 ``flops``。"""

    result: dict[str, object] = {}
    inserted = False
    for key, value in config.items():
        if key in {"mult_adds", "flops"}:
            continue
        result[key] = value
        if key == "parameter_count":
            result["flops"] = flops
            inserted = True
    if not inserted:
        result["flops"] = flops
    return result


def update_model_tiers(
    path: Path,
    vocab_size: int,
    sequence_length: int,
) -> dict[tuple[str, str], int]:
    """计算并写回一个档位文件，返回 ``(tier, model_type)`` 到 FLOPs 的映射。"""

    payload = _read_json_object(path)
    calculated: dict[tuple[str, str], int] = {}

    # 保留 tier 和模型的原始顺序，只替换计算量字段，避免扰动人工维护的其余配置。
    for tier, models in payload.items():
        if not isinstance(models, dict):
            raise ValueError(f"{path} 中的 tier {tier!r} 必须是对象")
        for model_name, raw_config in list(models.items()):
            if not isinstance(raw_config, dict):
                raise ValueError(f"{tier}.{model_name} 必须是模型配置对象")
            model_type = raw_config.get("model_type")
            if model_type != model_name:
                raise ValueError(f"{tier}.{model_name} 的 model_type 不一致")
            flops = estimate_model_flops(raw_config, vocab_size, sequence_length)
            models[model_name] = _replace_compute_field(raw_config, flops)
            calculated[(tier, model_name)] = flops

    _write_json_atomic(path, payload)
    return calculated


def update_frontend_catalog(
    path: Path,
    tier_flops: Mapping[tuple[str, str], int],
    vocab_size: int,
    sequence_length: int,
) -> None:
    """将同一 FLOPs 口径同步到前端模型目录。"""

    payload = _read_json_object(path)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("catalog 顶层必须包含 models 列表")

    for index, raw_record in enumerate(models):
        if not isinstance(raw_record, dict):
            raise ValueError("catalog 中的模型记录必须是对象")
        model_type = raw_record.get("model_type")
        tier = raw_record.get("tier")
        if model_type == "bigram":
            flops = estimate_bigram_flops(vocab_size, sequence_length)
        else:
            key = (tier, model_type)
            if key not in tier_flops:
                raise ValueError(f"catalog 模型 {tier}.{model_type} 不在档位配置中")
            flops = tier_flops[key]
        models[index] = _replace_compute_field(raw_record, flops)

    _write_json_atomic(path, payload)


def load_vocab_size(path: Path) -> int:
    """从 CharTokenizer JSON 读取词表大小。"""

    tokens = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokenizer.json 必须是非空 token 列表")
    if len(tokens) != len(set(tokens)):
        raise ValueError("tokenizer.json 不能包含重复 token")
    return len(tokens)


def synchronize_flops(
    project_root: Path = PROJECT_ROOT,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    tier_paths: list[Path] | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> dict[tuple[str, str], int]:
    """一次计算全部 tier/type，并同步所有档位文件和前端 catalog。"""

    if sequence_length <= 0:
        raise ValueError("sequence_length 必须大于 0")
    vocab_size = load_vocab_size(tokenizer_path)
    paths = discover_model_tier_paths(project_root) if tier_paths is None else tier_paths
    if not paths:
        raise FileNotFoundError("项目中没有找到 model_tiers.json")

    merged: dict[tuple[str, str], int] = {}
    for path in paths:
        current = update_model_tiers(path, vocab_size, sequence_length)
        for key, value in current.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"多个 model_tiers.json 对 {key} 给出了冲突配置")
            merged[key] = value
    baseline_key = ("baseline", "bigram")
    baseline_flops = estimate_bigram_flops(vocab_size, sequence_length)
    if baseline_key in merged and merged[baseline_key] != baseline_flops:
        raise ValueError("model_tiers.json 中的 baseline.bigram 计算结果冲突")
    merged[baseline_key] = baseline_flops
    update_frontend_catalog(catalog_path, merged, vocab_size, sequence_length)
    return merged


def main() -> None:
    """命令行入口：计算、同步并打印各档位结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="完整前向输入长度，默认 13",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--model-tiers",
        type=Path,
        action="append",
        help="指定 model_tiers.json；可重复传入，默认自动发现全部文件",
    )
    args = parser.parse_args()

    results = synchronize_flops(
        tokenizer_path=args.tokenizer,
        catalog_path=args.catalog,
        tier_paths=args.model_tiers,
        sequence_length=args.sequence_length,
    )
    print(
        f"已按 batch={DEFAULT_BATCH_SIZE}, length={args.sequence_length} 同步 Estimated FLOPs"
    )
    for (tier, model_type), flops in sorted(results.items()):
        print(f"{tier:>6} / {model_type:<11} {flops:>14,}")


if __name__ == "__main__":
    main()
