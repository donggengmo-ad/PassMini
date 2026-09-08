"""供 notebook 和脚本复用的实验流水线。"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from functools import partial
import matplotlib.pyplot as plt
import json
import torchinfo
import torch
from torch.utils.data import DataLoader

from .data import PasswordDataset, collate_batch, read_dataset
from .experiment import (
    AutoregressiveBigramConfig,
    AutoregressiveGRUConfig,
    AutoregressiveMLPConfig,
    AutoregressiveModelConfig,
    AutoregressiveTCNConfig,
    AutoregressiveTransformerConfig,
    ExperimentConfig,
)
from .inference import save_inference_config
from .models import (
    AutoregressiveBigram,
    AutoregressiveGRU,
    AutoregressiveMLP,
    AutoregressivePasswordModel,
    AutoregressiveTCN,
    AutoregressiveTransformer,
)
from .monitoring import TensorBoardMonitor
from .training import build_scheduler, train, evaluate, load_checkpoint
from .tokenizer import CharTokenizer


@dataclass
class ExperimentArtifacts:
    """保存一次训练流水线的主要运行结果。"""

    config: ExperimentConfig
    tokenizer: CharTokenizer
    model: AutoregressivePasswordModel
    history: dict
    test_loss: float | None = None


def seed_everything(seed: int) -> None:
    """固定 Python 和 PyTorch 随机状态。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(config: ExperimentConfig) -> CharTokenizer:
    """从配置指定的 JSON 加载 tokenizer。"""

    return CharTokenizer.from_json(config.data.tokenizer_path)


def build_dataloaders(
    config: ExperimentConfig,
    tokenizer: CharTokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """读取三个 split 并构造统一 padding 的 DataLoader。"""

    datasets = read_dataset(
        config.data.dataset_path,
        train_limit=config.data.train_limit,
        val_limit=config.data.val_limit,
        test_limit=config.data.test_limit,
    )
    password_datasets = {
        split: PasswordDataset(values, tokenizer) for split, values in datasets.items()
    }
    collate_fn = partial(collate_batch, pad_id=tokenizer.pad_id)
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        password_datasets["train"],
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        password_datasets["val"],
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        password_datasets["test"],
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def build_model(
    config: ExperimentConfig | AutoregressiveModelConfig,
    tokenizer: CharTokenizer,
) -> AutoregressivePasswordModel:
    """根据 canonical 模型配置构造模型。"""

    model_config = config.model if isinstance(config, ExperimentConfig) else config
    if isinstance(model_config, AutoregressiveBigramConfig):
        return AutoregressiveBigram(tokenizer, alpha=model_config.alpha)
    if isinstance(model_config, AutoregressiveMLPConfig):
        return AutoregressiveMLP(
            tokenizer,
            tau=model_config.tau,
            embedding_dim=model_config.embedding_dim,
            hidden_size=model_config.hidden_size,
        )
    if isinstance(model_config, AutoregressiveGRUConfig):
        return AutoregressiveGRU(
            tokenizer,
            embedding_dim=model_config.embedding_dim,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
        )
    if isinstance(model_config, AutoregressiveTCNConfig):
        return AutoregressiveTCN(
            tokenizer,
            embedding_dim=model_config.embedding_dim,
            channels=model_config.channels,
            kernel_size=model_config.kernel_size,
            dilations=model_config.dilations,
        )
    if isinstance(model_config, AutoregressiveTransformerConfig):
        return AutoregressiveTransformer(
            tokenizer,
            d_model=model_config.d_model,
            nhead=model_config.nhead,
            num_layers=model_config.num_layers,
            dim_feedforward=model_config.dim_feedforward,
            max_length=model_config.max_length,
        )
    raise TypeError(f"不支持的模型配置: {type(model_config).__name__}")


def build_optimizer(
    model: AutoregressivePasswordModel,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    """为可训练模型创建 Adam optimizer。"""

    if config.model.model_type == "bigram":
        raise ValueError("Bigram 使用 fit()，不创建 optimizer")
    return torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)


def run_training_experiment(
    config: ExperimentConfig,
    use_best: bool = True,
    resume: bool = False,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_log_interval: int = 100,
) -> ExperimentArtifacts:
    """执行一次完整神经模型训练并保存训练/推理 artifact。
    :param config: 实验配置
    :param use_best: 是否在训练结束后加载验证集上最优的模型
    :param resume: 是否从 output_dir 中的 latest checkpoint 恢复训练
    :param tensorboard_log_dir: 可选的 TensorBoard event 目录
    :param tensorboard_log_interval: batch 级训练和验证指标的记录间隔
    :return: 包含训练结果的 ExperimentArtifacts
    """

    if config.model.model_type == "bigram":
        raise ValueError("Bigram 不使用神经训练流水线")
    seed_everything(config.training.seed)
    tokenizer = load_tokenizer(config)
    train_loader, val_loader, test_loader = build_dataloaders(config, tokenizer)
    device = config.resolve_device()
    model = build_model(config, tokenizer).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config.training.scheduler, config.training.num_epochs)
    model_config = asdict(config.model)
    monitor = (
        None
        if tensorboard_log_dir is None
        else TensorBoardMonitor.create(tensorboard_log_dir)
    )
    try:
        history = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            device=device,
            max_norm=config.training.max_norm,
            num_epochs=config.training.num_epochs,
            save_path=config.data.output_dir,
            scheduler=scheduler,
            scheduler_config=config.training.scheduler,
            model_config=model_config,
            resume=resume,
            monitor=monitor,
            monitor_log_interval=tensorboard_log_interval,
        )
        if use_best:
            model.load_state_dict(
                load_checkpoint(
                    config.data.output_dir/'checkpoint_best.pt'
                )["model_state_dict"]
            )

        test_loss = evaluate(
            model,
            test_loader,
            device=device,
            monitor=monitor,
            monitor_group="Test/Batch",
            monitor_log_interval=tensorboard_log_interval,
        )
        if monitor is not None:
            monitor.log_metrics(
                "Loss",
                {"test": test_loss},
                max(len(history["train_loss"]), 1),
                flush=True,
            )
        save_training_artifact(config, tokenizer, history)
        save_inference_artifact(config, model, tokenizer)
        return ExperimentArtifacts(config, tokenizer, model, history, test_loss)
    finally:
        if monitor is not None:
            monitor.close()


def save_training_artifact(config: ExperimentConfig, tokenizer: CharTokenizer, history: dict) -> None:
    """保存完整实验配置、tokenizer 和训练历史。"""

    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.dump(output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


def save_inference_artifact(
    config: ExperimentConfig,
    model: AutoregressivePasswordModel,
    tokenizer: CharTokenizer,
) -> None:
    """保存部署所需的 model.pt、tokenizer.json 和 inference.json。"""

    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    if config.model.model_type == "bigram":
        assert isinstance(model, AutoregressiveBigram)
        model.save(model_path)
    else:
        torch.save(model.state_dict(), model_path)
    tokenizer.dump(output_dir / "tokenizer.json")
    save_inference_config(output_dir / "inference.json", config.model)

def get_device(index: int=0) -> torch.device:
    """
    获取当前可用的设备（GPU 或 CPU）
    :param index: GPU 的索引
    :return: 当前可用的设备
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def plot_loss(history: dict,
              save_path: str | None=None,
              test_loss: float | None=None) -> None:
    """
    绘制训练和验证损失曲线，可选地标记测试损失。
    :param history: 包含训练和验证损失的字典（'train_loss' 和 'valid_loss' 键）
    :param save_path: 保存图像的路径，如果为 None，则不保存
    :param test_loss: 测试集损失标量；如果为 None，则不绘制测试损失线

    横轴沿用 matplotlib 的 epoch 序列；测试损失使用水平虚线表示。
    """
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Valid Loss')
    if test_loss is not None:
        # 测试损失不随训练 epoch 变化，因此用贯穿整个坐标区域的水平线表示。
        plt.axhline(
            test_loss,
            color='green',
            linestyle='--',
            label='Test Loss',
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss')
    plt.grid()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def model_summary(config: ExperimentConfig, **kwargs) -> None:
    """
    打印模型摘要信息，包括参数数量和模型结构。
    :param config: 实验配置
    :param kwargs: 传递给 torchinfo.summary 的其他参数
    """
    tokenizer = load_tokenizer(config)
    model = build_model(config, tokenizer)
    print(torchinfo.summary(model, input_size=(1, 1), dtypes=[torch.long], depth=2, **kwargs))

def load_model_tier(model_type: str,
                    model_tier: str='low',
                    path: str='output/model_tiers.json') -> dict:
    """
    加载模型等级配置。
    :param model_type: 模型类型
    :param model_tier: 模型等级
    :param path: 模型等级配置文件路径
    :return: 模型等级配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        all_tier = json.load(f)
        if model_tier not in all_tier:
            raise ValueError(f"模型等级 {model_tier} 不在配置中")
        for v in all_tier.values():
            if model_type not in v:
                raise ValueError(f"模型类型 {model_type} 不在配置中")
    return all_tier[model_tier][model_type]

__all__ = [
    "ExperimentArtifacts",
    "seed_everything",
    "load_tokenizer",
    "build_dataloaders",
    "build_model",
    "build_optimizer",
    "run_training_experiment",
    "save_training_artifact",
    "save_inference_artifact",
    "get_device",
    "plot_loss",
    "model_summary",
    "load_model_tier",
]
