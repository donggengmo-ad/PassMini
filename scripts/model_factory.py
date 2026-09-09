"""训练和推理共用的模型构造入口；不加载权重或创建优化器。"""

from .experiment import (
    AutoregressiveBigramConfig, AutoregressiveGRUConfig, AutoregressiveMLPConfig,
    AutoregressiveModelConfig, AutoregressiveTCNConfig, AutoregressiveTransformerConfig,
)
from .models import (
    AutoregressiveBigram, AutoregressiveGRU, AutoregressiveMLP,
    AutoregressivePasswordModel, AutoregressiveTCN, AutoregressiveTransformer,
)
from .tokenizer import CharTokenizer


def build_model_from_config(config: AutoregressiveModelConfig, tokenizer: CharTokenizer) -> AutoregressivePasswordModel:
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
