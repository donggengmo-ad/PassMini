"""自回归神经模型的训练、验证、scheduler 和 checkpoint。"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch

from .experiment import SchedulerConfig
from .monitoring import TensorBoardMonitor
from .models import AutoregressivePasswordModel


Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    num_epochs: int,
) -> Scheduler | None:
    """根据配置构造学习率调度器。"""

    if config.name == "none":
        return None
    if config.name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
    if config.name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    if config.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.t_max or num_epochs,
            eta_min=config.eta_min,
        )
    raise ValueError(f"不支持的 scheduler: {config.name!r}")


def _step_scheduler(
    scheduler: Scheduler | None,
    valid_loss: float,
) -> None:
    """在 epoch 结束后按 scheduler 类型执行一次更新。"""

    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(valid_loss)
    else:
        scheduler.step()


def save_checkpoint(
    filename: str | Path,
    model: AutoregressivePasswordModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float | None = None,
    valid_loss: float | None = None,
    history: dict | None = None,
    *,
    model_config: Mapping | None = None,
    scheduler: Scheduler | None = None,
    scheduler_config: SchedulerConfig | None = None,
    best_val_loss: float = float("inf"),
    **kwargs,
) -> dict:
    """保存可继续训练的完整 checkpoint。"""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config or {"model_type": model.model_type}),
        "tokenizer_state_dict": model.tokenizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scheduler_config": None if scheduler_config is None else asdict(scheduler_config),
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "best_val_loss": best_val_loss,
        "history": history,
        "extra": kwargs,
    }
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def load_checkpoint(
    filename: str | Path,
    map_location: torch.device | str = torch.device("cpu"),
) -> dict:
    """加载训练 checkpoint，不负责构造模型、优化器或 scheduler。"""

    return torch.load(filename, map_location=map_location, weights_only=True)


def _validate_criterion(
    criterion: torch.nn.CrossEntropyLoss,
    pad_id: int,
) -> None:
    """验证 token 总损失和 PAD 忽略契约。"""

    if criterion.reduction != "sum":
        raise ValueError("损失函数必须使用 'sum' 作为 reduction")
    if criterion.ignore_index != pad_id:
        raise ValueError("损失函数的 ignore_index 必须与 model.pad_id 相同")


def train_one_epoch(
    model: AutoregressivePasswordModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss | None = None,
    device: torch.device = torch.device("cpu"),
    max_norm: float | int = 1.0,
    monitor: TensorBoardMonitor | None = None,
    epoch_index: int = 0,
    monitor_log_interval: int = 100,
) -> float:
    """训练一个 epoch，返回非 PAD token 的平均交叉熵并可实时记录进度。"""

    if epoch_index < 0 or monitor_log_interval <= 0:
        raise ValueError("epoch_index 不能为负数，monitor_log_interval 必须大于 0")
    pad_id = model.pad_id
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    _validate_criterion(criterion, pad_id)
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_batches = len(dataloader)
    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # `[B,L,V]` 和 `[B,L]` 展平后，CrossEntropyLoss 把每个时间步视为一个分类样本。
        flat_outputs = outputs.reshape(-1, outputs.size(-1))
        flat_targets = targets.reshape(-1)
        loss = criterion(flat_outputs, flat_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
        total_tokens += (flat_targets != pad_id).sum().item()
        if monitor is not None and (
            batch_index % monitor_log_interval == 0 or batch_index == total_batches
        ):
            global_step = epoch_index * max(total_batches, 1) + batch_index
            monitor.log_metrics(
                "Train/Batch",
                {
                    "running_loss": total_loss / total_tokens if total_tokens else -1.0,
                    "epoch_progress": batch_index / max(total_batches, 1),
                },
                global_step,
            )
    return total_loss / total_tokens if total_tokens > 0 else -1.0


def evaluate(
    model: AutoregressivePasswordModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.CrossEntropyLoss | None = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
    monitor: TensorBoardMonitor | None = None,
    monitor_group: str = "Evaluation/Batch",
    monitor_step_offset: int = 0,
    monitor_log_interval: int = 100,
) -> float:
    """评估模型并返回非 PAD token 的平均交叉熵，可选记录批次进度。"""

    if monitor_step_offset < 0 or monitor_log_interval <= 0:
        raise ValueError("monitor_step_offset 不能为负数，monitor_log_interval 必须大于 0")
    pad_id = model.pad_id
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    _validate_criterion(criterion, pad_id)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        total_batches = len(dataloader)
        for index, (inputs, targets) in enumerate(dataloader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            flat_outputs = outputs.reshape(-1, outputs.size(-1))
            flat_targets = targets.reshape(-1)
            loss = criterion(flat_outputs, flat_targets)
            total_loss += loss.item()
            total_tokens += (flat_targets != pad_id).sum().item()
            if verbose and index % 100 == 0:
                print(
                    f"Evaluating: {index}/{len(dataloader)}, "
                    f"current loss: {total_loss / total_tokens:.4f}"
                )
            if monitor is not None and (
                index % monitor_log_interval == 0 or index == total_batches
            ):
                monitor.log_metrics(
                    monitor_group,
                    {
                        "running_loss": total_loss / total_tokens if total_tokens else -1.0,
                        "progress": index / max(total_batches, 1),
                    },
                    monitor_step_offset + index,
                )
    return total_loss / total_tokens if total_tokens > 0 else -1.0


def train(
    model: AutoregressivePasswordModel,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss | None = None,
    device: torch.device = torch.device("cpu"),
    max_norm: float | int = 1.0,
    num_epochs: int = 10,
    save_path: Path | None = None,
    resume: bool = False,
    *,
    scheduler: Scheduler | None = None,
    scheduler_config: SchedulerConfig | None = None,
    model_config: Mapping | None = None,
    monitor: TensorBoardMonitor | None = None,
    monitor_log_interval: int = 100,
) -> dict[str, list[float]]:
    """训练模型，并可保存/恢复状态以及实时写入 TensorBoard。"""

    if monitor_log_interval <= 0:
        raise ValueError("monitor_log_interval 必须大于 0")
    if len(train_dataloader) == 0 or len(valid_dataloader) == 0:
        raise ValueError("训练集和验证集必须至少各包含一个 batch")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "valid_loss": [],
        "epoch_seconds": [],
        "learning_rate": [],
    }
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        if save_path is None:
            raise ValueError("resume=True 时必须提供 save_path")
        checkpoint = load_checkpoint(save_path / "checkpoint_latest.pt", device)
        # 先检查语义再写入权重：词表大小相同，不代表每个 ID 对应的字符相同。
        if checkpoint.get("tokenizer_state_dict") != model.tokenizer.state_dict():
            raise ValueError("checkpoint 的 tokenizer 与当前模型不一致")
        if model_config is not None and checkpoint.get("model_config") != dict(model_config):
            raise ValueError("checkpoint 的 model_config 与当前配置不一致")
        if (checkpoint.get("scheduler_state_dict") is None) != (scheduler is None):
            raise ValueError("checkpoint 与当前配置的 scheduler 启用状态不一致")
        if scheduler_config is not None and checkpoint.get("scheduler_config") != asdict(scheduler_config):
            raise ValueError("checkpoint 的 scheduler_config 与当前配置不一致")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            state = checkpoint.get("scheduler_state_dict")
            if state is None:
                raise ValueError("checkpoint 不包含 scheduler_state_dict")
            scheduler.load_state_dict(state)
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint["history"]
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    for epoch in range(start_epoch):
        print(
            f"Epoch {epoch + 1}/{num_epochs} - 训练损失: {history['train_loss'][epoch]:.4f}, "
            f"验证损失: {history['valid_loss'][epoch]:.4f}, "
            f"耗时: {history['epoch_seconds'][epoch]:.2f}s"
        )

    for epoch in range(start_epoch, num_epochs):
        start_time = time.perf_counter()
        # 记录本 epoch 实际用于 optimizer.step() 的学习率。
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion=criterion,
            device=device,
            max_norm=max_norm,
            monitor=monitor,
            epoch_index=epoch,
            monitor_log_interval=monitor_log_interval,
        )
        valid_loss = evaluate(
            model,
            valid_dataloader,
            criterion=criterion,
            device=device,
            monitor=monitor,
            monitor_group="Validation/Batch",
            monitor_step_offset=epoch * len(valid_dataloader),
            monitor_log_interval=monitor_log_interval,
        )
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["epoch_seconds"].append(time.perf_counter() - start_time)
        _step_scheduler(scheduler, valid_loss)

        if monitor is not None:
            # epoch 标量共享横轴，便于在 TensorBoard 中直接比较训练、验证和泛化趋势。
            monitor.log_metrics(
                "Loss",
                {
                    "train": train_loss,
                    "validation": valid_loss,
                    "generalization_gap": valid_loss - train_loss,
                },
                epoch + 1,
            )
            monitor.log_metrics(
                "Optimization",
                {"learning_rate": history["learning_rate"][-1]},
                epoch + 1,
            )
            monitor.log_metrics(
                "Runtime",
                {"epoch_seconds": history["epoch_seconds"][-1]},
                epoch + 1,
                flush=True,
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} - 训练损失: {train_loss:.4f}, "
            f"验证损失: {valid_loss:.4f}, "
            f"耗时: {history['epoch_seconds'][-1]:.2f}s"
        )

        is_best = valid_loss < best_val_loss
        if is_best:
            best_val_loss = valid_loss
        if save_path is not None:
            checkpoint_kwargs = dict(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                valid_loss=valid_loss,
                history=history,
                model_config=model_config,
                scheduler=scheduler,
                scheduler_config=scheduler_config,
                best_val_loss=best_val_loss,
            )
            save_checkpoint(save_path / "checkpoint_latest.pt", **checkpoint_kwargs)
            if is_best:
                save_checkpoint(save_path / "checkpoint_best.pt", **checkpoint_kwargs)
    return history


__all__ = [
    "build_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "train_one_epoch",
    "evaluate",
    "train",
]
