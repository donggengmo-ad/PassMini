"""TensorBoard 实时监控的轻量封装。"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol


class ScalarWriter(Protocol):
    """约束 TensorBoard 标量写入器所需的最小接口。"""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


ProgressCallback = Callable[[int, Mapping[str, float | int]], None]


class TensorBoardMonitor:
    """集中写入层级化标量，并允许长任务复用统一进度回调。"""

    def __init__(self, writer: ScalarWriter):
        self.writer = writer

    @classmethod
    def create(
        cls,
        log_dir: str | Path,
        flush_secs: int = 10,
    ) -> "TensorBoardMonitor":
        r"""创建真实 SummaryWriter。
        :param log_dir: TensorBoard event 文件目录；AutoDL 推荐放在 `/root/tf-logs` 下
        :param flush_secs: 后台最多间隔多少秒把缓存写入磁盘
        :return: 可用于训练和评测的监控对象
        """

        if flush_secs <= 0:
            raise ValueError("flush_secs 必须大于 0")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "当前环境缺少 tensorboard，请先执行 `python -m pip install tensorboard`"
            ) from error
        return cls(SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs))

    def log_metrics(
        self,
        group: str,
        metrics: Mapping[str, float | int],
        step: int,
        *,
        flush: bool = False,
    ) -> None:
        r"""在同一图组写入多个标量。
        :param group: TensorBoard 一级标签，例如 `Loss` 或 `Surprisal/GRU`
        :param metrics: 指标名称到有限数值的映射
        :param step: 非负横轴步数
        :param flush: 写入后是否立即刷新磁盘
        """

        if not group:
            raise ValueError("group 不能为空")
        if step < 0:
            raise ValueError("step 不能为负数")
        for name, value in metrics.items():
            scalar = float(value)
            if not name or not math.isfinite(scalar):
                raise ValueError("TensorBoard 指标名称不能为空，数值必须有限")
            self.writer.add_scalar(f"{group}/{name}", scalar, step)
        if flush:
            self.flush()

    def progress_callback(self, group: str) -> ProgressCallback:
        """返回适配长耗时评测函数的标量进度回调。"""

        def report(step: int, metrics: Mapping[str, float | int]) -> None:
            self.log_metrics(group, metrics, step)

        return report

    def flush(self) -> None:
        """立即把异步缓存写入 event 文件，便于网页及时刷新。"""

        self.writer.flush()

    def close(self) -> None:
        """刷新并关闭底层 writer。"""

        self.writer.close()

    def __enter__(self) -> "TensorBoardMonitor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["ProgressCallback", "ScalarWriter", "TensorBoardMonitor"]
