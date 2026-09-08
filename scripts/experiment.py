"""实验配置、模型配置和训练调度配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass
class DataConfig:
    """数据文件、tokenizer 和各 split 限制。"""

    dataset_path: Path = Path("data/processed")
    tokenizer_path: Path = Path("data/processed/tokenizer.json")
    output_dir: Path = Path("data/experiments/autorg_gru")
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None


@dataclass
class AutoregressiveBigramConfig:
    """Bigram 模型结构配置。"""

    model_type: Literal["bigram"] = "bigram"
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha 必须大于 0")


@dataclass
class AutoregressiveMLPConfig:
    """固定上下文 MLP 模型结构配置。
    :param tau: 每次预测使用的最近 token 数量
    :param embedding_dim: 输入嵌入和共享输出空间维度
    :param hidden_size: MLP 隐藏层维度
    """

    model_type: Literal["mlp"] = "mlp"
    tau: int = 4
    embedding_dim: int = 64
    hidden_size: int = 128

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.embedding_dim <= 0 or self.hidden_size <= 0:
            raise ValueError("tau、embedding_dim、hidden_size 必须大于 0")


@dataclass
class AutoregressiveGRUConfig:
    """固定权重共享 GRU 模型结构配置。
    :param embedding_dim: 输入嵌入维度
    :param hidden_size: GRU 隐藏状态维度
    :param num_layers: GRU 层数
    """

    model_type: Literal["gru"] = "gru"
    embedding_dim: int = 64
    hidden_size: int = 128
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0 or self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("embedding_dim、hidden_size、num_layers 必须大于 0")


@dataclass
class AutoregressiveTCNConfig:
    """因果卷积模型结构配置。
    :param embedding_dim: 输入嵌入维度
    :param channels: 卷积通道数
    :param kernel_size: 卷积核大小
    :param dilations: 每层卷积的膨胀率序列
    """

    model_type: Literal["tcn"] = "tcn"
    embedding_dim: int = 64
    channels: int = 128
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0 or self.channels <= 0 or self.kernel_size <= 0:
            raise ValueError("embedding_dim、channels、kernel_size 必须大于 0")
        if not self.dilations or any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("dilations 必须是正整数序列")


@dataclass
class AutoregressiveTransformerConfig:
    """Transformer 模型结构配置。
    :param d_model: Transformer 模型维度
    :param nhead: 多头注意力头数
    :param num_layers: Transformer decoder 层数
    :param dim_feedforward: 前馈网络维度
    :param max_length: 最大序列长度
    """

    model_type: Literal["transformer"] = "transformer"
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    max_length: int = 16

    def __post_init__(self) -> None:
        if (
            self.d_model <= 0
            or self.nhead <= 0
            or self.num_layers <= 0
            or self.dim_feedforward <= 0
            or self.max_length <= 0
        ):
            raise ValueError("Transformer 配置必须为正整数")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")


AutoregressiveModelConfig = (
    AutoregressiveBigramConfig
    | AutoregressiveMLPConfig
    | AutoregressiveGRUConfig
    | AutoregressiveTCNConfig
    | AutoregressiveTransformerConfig
)


@dataclass
class SchedulerConfig:
    """训练学习率调度策略及其参数。
    :param name: 调度器类型，可选值为 "none"、"reduce_on_plateau"、"step_lr" 或 "cosine"
    :param factor: 在 "reduce_on_plateau" 策略中，每次降低学习率的乘法因子
    :param patience: 在 "reduce_on_plateau" 策略中，等待多少个 epoch 后才降低学习率
    :param min_lr: 在 "reduce_on_plateau" 策略中，学习率的最小值
    :param step_size: 在 "step_lr" 策略中，每隔多少个 epoch 降低学习率
    :param gamma: 在 "step_lr" 策略中，每次降低学习率的乘法因子
    :param t_max: 在 "cosine" 策略中，半个周期的 epoch 数
    :param eta_min: 在 "cosine" 策略中，学习率的最小值
    """

    name: Literal["none", "reduce_on_plateau", "step_lr", "cosine"] = "none"
    factor: float = 0.5
    patience: int = 1
    min_lr: float = 0.0
    step_size: int = 1
    gamma: float = 0.1
    t_max: int | None = None
    eta_min: float = 0.0

    def __post_init__(self) -> None:
        """在配置创建时拒绝非法策略和参数。"""

        allowed = {"none", "reduce_on_plateau", "step_lr", "cosine"}
        if self.name not in allowed:
            raise ValueError(f"不支持的 scheduler: {self.name!r}")
        if self.factor <= 0 or self.factor >= 1:
            raise ValueError("factor 必须在 (0, 1) 范围内")
        if self.patience < 0:
            raise ValueError("patience 不能为负数")
        if self.min_lr < 0:
            raise ValueError("min_lr 不能为负数")
        if self.step_size <= 0:
            raise ValueError("step_size 必须大于 0")
        if self.gamma <= 0:
            raise ValueError("gamma 必须大于 0")
        if self.t_max is not None and self.t_max <= 0:
            raise ValueError("t_max 必须大于 0 或为 None")
        if self.eta_min < 0:
            raise ValueError("eta_min 不能为负数")


@dataclass
class TrainingConfig:
    """训练超参数和 scheduler 配置。"""

    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-3
    max_norm: float = 1.0
    seed: int = 42
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.num_epochs <= 0:
            raise ValueError("batch_size 和 num_epochs 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.max_norm <= 0:
            raise ValueError("max_norm 必须大于 0")


@dataclass
class DeviceConfig:
    """设备选择配置。"""

    mode: Literal["smoke_cpu", "formal_cuda"] = "formal_cuda"
    cuda_index: int = 0


@dataclass
class ExperimentConfig:
    """组合数据、模型、训练和设备配置。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: AutoregressiveModelConfig = field(default_factory=AutoregressiveGRUConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    def __post_init__(self) -> None:
        """校验模型配置与其 canonical 标签一致。"""

        expected = {
            AutoregressiveBigramConfig: "bigram",
            AutoregressiveMLPConfig: "mlp",
            AutoregressiveGRUConfig: "gru",
            AutoregressiveTCNConfig: "tcn",
            AutoregressiveTransformerConfig: "transformer",
        }
        if type(self.model) not in expected:
            raise TypeError("model 必须是受支持的自回归模型配置")
        if self.model.model_type != expected[type(self.model)]:
            raise ValueError("模型配置的 model_type 与 dataclass 类型不一致")

    def to_dict(self) -> dict:
        """递归转换为 JSON 可序列化字典。"""

        def convert(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            raise TypeError(f"不支持序列化的配置类型: {type(value).__name__}")

        return convert(asdict(self))

    def resolve_device(self) -> torch.device:
        """根据设备模式解析实际 torch.device。"""

        if self.device.mode == "smoke_cpu":
            return torch.device("cpu")
        if self.device.mode == "formal_cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("formal_cuda 模式要求 CUDA 可用")
            if not 0 <= self.device.cuda_index < torch.cuda.device_count():
                raise ValueError("cuda_index 超出可用 GPU 范围")
            return torch.device(f"cuda:{self.device.cuda_index}")
        raise ValueError(f"不支持的设备模式: {self.device.mode}")


def model_config_from_dict(payload: Mapping) -> AutoregressiveModelConfig:
    """从只含 canonical 字段的字典构造模型配置。"""

    model_type = payload.get("model_type")
    allowed_fields = {
        "bigram": {"model_type", "alpha", "vocab_size"},
        "mlp": {"model_type", "tau", "embedding_dim", "hidden_size", "vocab_size"},
        "gru": {"model_type", "embedding_dim", "hidden_size", "num_layers", "vocab_size"},
        "tcn": {"model_type", "embedding_dim", "channels", "kernel_size", "dilations", "vocab_size"},
        "transformer": {
            "model_type",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "max_length",
            "vocab_size",
        },
    }
    if model_type in allowed_fields:
        unknown = set(payload) - allowed_fields[model_type]
        if unknown:
            raise ValueError(f"{model_type} 配置包含不支持的字段: {sorted(unknown)}")
    if model_type == "bigram":
        return AutoregressiveBigramConfig(alpha=float(payload.get("alpha", 1.0)))
    if model_type == "mlp":
        return AutoregressiveMLPConfig(
            tau=int(payload.get("tau", 4)),
            embedding_dim=int(payload.get("embedding_dim", 64)),
            hidden_size=int(payload.get("hidden_size", 128)),
        )
    if model_type == "gru":
        return AutoregressiveGRUConfig(
            embedding_dim=int(payload.get("embedding_dim", 64)),
            hidden_size=int(payload.get("hidden_size", 128)),
            num_layers=int(payload.get("num_layers", 1)),
        )
    if model_type == "tcn":
        return AutoregressiveTCNConfig(
            embedding_dim=int(payload.get("embedding_dim", 64)),
            channels=int(payload.get("channels", 128)),
            kernel_size=int(payload.get("kernel_size", 3)),
            dilations=tuple(int(item) for item in payload.get("dilations", (1, 2, 4))),
        )
    if model_type == "transformer":
        return AutoregressiveTransformerConfig(
            d_model=int(payload.get("d_model", 64)),
            nhead=int(payload.get("nhead", 4)),
            num_layers=int(payload.get("num_layers", 2)),
            dim_feedforward=int(payload.get("dim_feedforward", 128)),
            max_length=int(payload.get("max_length", 16)),
        )
    raise ValueError(f"不支持的 canonical model_type: {model_type!r}")


__all__ = [
    "DataConfig",
    "AutoregressiveBigramConfig",
    "AutoregressiveMLPConfig",
    "AutoregressiveGRUConfig",
    "AutoregressiveTCNConfig",
    "AutoregressiveTransformerConfig",
    "AutoregressiveModelConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "DeviceConfig",
    "ExperimentConfig",
    "model_config_from_dict",
]
