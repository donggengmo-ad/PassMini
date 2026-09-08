"""测试集搜索和 surprisal 评测辅助函数。"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .inference import GenerationCandidate
from .data import normalize_password
from .monitoring import ProgressCallback


@dataclass(frozen=True)
class CoveragePoint:
    r"""保存一个搜索预算检查点的覆盖率结果。
    :ivar attempts: 截止当前检查点已经搜索的候选数量
    :ivar hits: 前 `attempts` 个候选命中的测试集密码数量
    :ivar coverage: 累计命中数除以测试集大小
    :ivar efficiency: 当前累计覆盖率除以搜索次数
    """

    attempts: int
    hits: int
    coverage: float
    efficiency: float


@dataclass(frozen=True)
class MetricSummary:
    r"""保存一组有限数值的分布统计量。

    百分位数使用线性插值计算，保证相同输入在不同评测运行中得到相同结果。
    """

    count: int
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class SurprisalSummary:
    r"""保存 raw surprisal 和按 token 归一化后的统计结果。"""

    count: int
    raw_surprisal_bits: MetricSummary
    bits_per_token: MetricSummary


@dataclass(frozen=True)
class GenerationQualitySummary:
    r"""保存随机生成样本的合法率和合法唯一率。"""

    total_samples: int
    legal_samples: int
    distinct_legal_samples: int
    legal_rate: float
    legal_unique_rate: float


@dataclass(frozen=True)
class RandomGenerationEvaluation:
    r"""保存同一批随机样本得到的质量摘要和覆盖率检查点。"""

    quality: GenerationQualitySummary
    coverage: list[CoveragePoint]


@dataclass(frozen=True)
class EvaluationArtifactPaths:
    """集中描述一个模型的全部评测产物路径。"""

    directory: Path
    surprisal: Path
    random_coverage: Path
    best_first_candidates: Path
    best_first_coverage: Path
    summary: Path


def evaluation_artifact_paths(
    output_root: str | Path,
    tier: str,
    model_type: str,
) -> EvaluationArtifactPaths:
    r"""按照 `<output_root>/<tier>/<model_type>` 构造标准评测路径。
    :param output_root: 评测结果根目录，通常为 `output/evaluation`
    :param tier: 模型档位，例如 `baseline`、`low`、`medium` 或 `high`
    :param model_type: 模型架构标签，例如 `bigram`、`mlp`、`gru`、`tcn` 或 `transformer`
    :return: 该模型全部标准评测文件的路径集合
    """

    if not tier or not model_type:
        raise ValueError("tier 和 model_type 不能为空")
    directory = Path(output_root) / tier / model_type
    return EvaluationArtifactPaths(
        directory=directory,
        surprisal=directory / "surprisal.npz",
        random_coverage=directory / "random_coverage.npz",
        best_first_candidates=directory / "best_first.json",
        best_first_coverage=directory / "best_first_coverage.npz",
        summary=directory / "summary.json",
    )


def save_evaluation_summary(
    path: str | Path,
    *,
    surprisal: SurprisalSummary | None = None,
    generation_quality: GenerationQualitySummary | None = None,
) -> Path:
    r"""增量保存单模型评测摘要，不覆盖同文件中的另一类已完成评测。
    :param path: 单模型 `summary.json` 路径
    :param surprisal: 可选的完整测试集惊讶度摘要
    :param generation_quality: 可选的随机生成质量摘要
    :return: 实际写入路径
    """

    if surprisal is None and generation_quality is None:
        raise ValueError("至少需要提供一种评测摘要")
    output_path = Path(path)
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("已有 summary.json 顶层必须是对象")
    else:
        payload = {}
    if surprisal is not None:
        payload["surprisal"] = asdict(surprisal)
    if generation_quality is not None:
        payload["generation_quality"] = asdict(generation_quality)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _uniform_sample_indices(size: int, max_points: int) -> np.ndarray:
    """返回覆盖完整下标范围的等距抽样位置。"""

    if max_points <= 0:
        raise ValueError("max_points 必须大于 0")
    if size <= max_points:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, num=max_points, dtype=np.int64)


def save_surprisal_npz(
    path: str | Path,
    surprisal_bits: Sequence[float],
    bits_per_token: Sequence[float],
    max_points: int = 100_000,
) -> Path:
    r"""等距抽取完整评分结果，并保存前端绘图所需的数值数组。
    :param path: 输出 NPZ 文件路径
    :param surprisal_bits: 完整评测集的密码总惊讶度
    :param bits_per_token: 与总惊讶度逐项对应的单位 token 惊讶度
    :param max_points: NPZ 最多保存的点数；数据较少时保存全部
    :return: 实际写入的文件路径

    文件只包含 `evaluation_size`、`surprisal_bits` 和 `bits_per_token`。
    正式统计仍应基于完整评分；这里的等距抽样只用于限制前端绘图数据量。
    """

    raw_values = np.asarray(list(surprisal_bits), dtype=np.float32)
    normalized_values = np.asarray(list(bits_per_token), dtype=np.float32)
    if raw_values.ndim != 1 or normalized_values.ndim != 1:
        raise ValueError("surprisal 数据必须是一维序列")
    if raw_values.size == 0:
        raise ValueError("surprisal 数据不能为空")
    if raw_values.size != normalized_values.size:
        raise ValueError("surprisal_bits 和 bits_per_token 数量必须一致")
    if not np.isfinite(raw_values).all() or not np.isfinite(normalized_values).all():
        raise ValueError("surprisal 数据必须全部是有限数值")

    # 两组分数复用同一组等距下标，保证导出后仍然逐项对应。
    indices = _uniform_sample_indices(raw_values.size, max_points)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        np.savez_compressed(
            file,
            evaluation_size=np.asarray(raw_values.size, dtype=np.int64),
            surprisal_bits=raw_values[indices],
            bits_per_token=normalized_values[indices],
        )
    return output_path


def save_coverage_npz(
    path: str | Path,
    points: Sequence[CoveragePoint],
    test_size: int,
) -> Path:
    r"""保存覆盖率曲线的全部检查点。
    :param path: 输出 NPZ 文件路径
    :param points: 按搜索或采样顺序排列的覆盖率检查点
    :param test_size: 计算覆盖率时使用的测试集大小
    :return: 实际写入的文件路径

    文件只包含 `test_size`、`attempts` 和 `coverage`。累计命中数可以通过
    `round(coverage * test_size)` 恢复，搜索效率可以通过
    `coverage / attempts` 动态计算，因此不重复保存。
    """

    if test_size <= 0:
        raise ValueError("test_size 必须大于 0")
    point_list = list(points)
    if not point_list:
        raise ValueError("覆盖率检查点不能为空")
    attempts = np.asarray([point.attempts for point in point_list], dtype=np.int64)
    coverage = np.asarray([point.coverage for point in point_list], dtype=np.float64)
    if np.any(attempts <= 0) or np.any(attempts[1:] <= attempts[:-1]):
        raise ValueError("attempts 必须为严格递增的正整数")
    if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("coverage 必须是 [0, 1] 内的有限数值")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        np.savez_compressed(
            file,
            test_size=np.asarray(test_size, dtype=np.int64),
            attempts=attempts,
            coverage=coverage,
        )
    return output_path


def generate_random_passwords(
    model,
    num_samples: int = 10_000,
    max_length: int = 12,
    temperature: float = 1.0,
    batch_size: int = 512,
    generator=None,
    verbose: bool = False,
    progress_step: int = 1_000,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    r"""调用模型独立随机采样并返回生成文本。
    :param model: 实现 `generate()` 的密码模型
    :param num_samples: 随机生成的样本数量
    :param max_length: 每条生成文本允许的最大字符数
    :param temperature: 传递给模型的温度参数
    :param batch_size: 每次调用模型批量生成的样本数量
    :param generator: 可选的随机数生成器
    :param verbose: 是否打印采样进度，不打印生成文本
    :param progress_step: verbose 模式下的进度输出间隔
    :param progress_callback: 可选实时回调，接收已完成样本数和当前速度/进度
    :return: 按采样顺序排列的生成文本；允许重复
    """

    if num_samples <= 0:
        raise ValueError("num_samples 必须大于 0")
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if progress_step <= 0:
        raise ValueError("progress_step 必须大于 0")

    samples: list[str] = []
    start_time = time.perf_counter()
    while len(samples) < num_samples:
        current_batch_size = min(batch_size, num_samples - len(samples))
        generated_batch = _generate_password_batch(
            model,
            current_batch_size,
            max_length,
            temperature,
            generator,
        )
        samples.extend(generated_batch)
        if progress_callback is not None:
            elapsed = max(time.perf_counter() - start_time, 1e-12)
            progress_callback(
                len(samples),
                {
                    "progress": len(samples) / num_samples,
                    "samples_per_second": len(samples) / elapsed,
                },
            )
        if verbose:
            completed = len(samples)
            crossed_progress_step = completed % progress_step < current_batch_size
            if crossed_progress_step or completed == num_samples:
                print(f"Generated {completed} / {num_samples} samples")
    return samples


def _generate_password_batch(
    model,
    batch_size: int,
    max_length: int,
    temperature: float,
    generator,
) -> list[str]:
    """调用并校验一次模型批量生成，供普通生成与流式评测复用。"""

    generated_batch = model.generate_batch(
        batch_size=batch_size,
        max_length=max_length,
        temperature=temperature,
        generator=generator,
    )
    if not isinstance(generated_batch, list) or not all(
        isinstance(generated, str) for generated in generated_batch
    ):
        raise TypeError("model.generate_batch() 必须返回字符串列表")
    if len(generated_batch) != batch_size:
        raise ValueError("model.generate_batch() 返回数量与 batch_size 不一致")
    return generated_batch


def _is_legal_password(password: str, max_length: int) -> bool:
    """判断字符串是否满足项目统一的密码规范化约束。"""

    return normalize_password(password, min_length=1, max_length=max_length) == password


def summarize_generation_quality(
    samples: Sequence[str],
    max_length: int = 12,
) -> GenerationQualitySummary:
    r"""统计随机生成文本的合法率和合法唯一率。
    :param samples: 按采样顺序排列的生成文本，可以包含重复值
    :param max_length: 合法密码允许的最大字符数
    :return: 生成总数、合法数、不同合法数及两个比例

    合法性沿用数据预处理约束：长度在 1 到 `max_length` 之间，且所有字符均为可打印
    ASCII。合法唯一率只在合法样本内去重；没有合法样本时该指标定义为 0.0。
    """

    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("samples 不能为空")
    if not all(isinstance(sample, str) for sample in sample_list):
        raise TypeError("samples 中的每个元素必须是 str")

    legal_samples = [sample for sample in sample_list if _is_legal_password(sample, max_length)]
    legal_count = len(legal_samples)
    distinct_legal_count = len(set(legal_samples))
    return GenerationQualitySummary(
        total_samples=len(sample_list),
        legal_samples=legal_count,
        distinct_legal_samples=distinct_legal_count,
        legal_rate=legal_count / len(sample_list),
        legal_unique_rate=distinct_legal_count / legal_count if legal_count else 0.0,
    )


def evaluate_random_generation(
    model,
    num_samples: int = 10_000,
    max_length: int = 12,
    temperature: float = 1.0,
    batch_size: int = 512,
    generator=None,
    verbose: bool = False,
    progress_step: int = 1_000,
    progress_callback: ProgressCallback | None = None,
) -> GenerationQualitySummary:
    r"""随机采样并直接返回生成质量摘要。

    `batch_size` 控制每次模型调用的并行样本数。需要同时绘制随机覆盖率时，应先调用 `generate_random_passwords()`，再将返回的
    样本传给 `summarize_generation_quality()` 和 `random_generation_coverage_curve()`。
    `progress_callback` 会原样传给批量生成过程，可用于 TensorBoard 实时监控。
    """

    samples = generate_random_passwords(
        model,
        num_samples=num_samples,
        max_length=max_length,
        temperature=temperature,
        batch_size=batch_size,
        generator=generator,
        verbose=verbose,
        progress_step=progress_step,
        progress_callback=progress_callback,
    )
    return summarize_generation_quality(samples, max_length=max_length)


def evaluate_random_generation_with_coverage(
    model,
    test_passwords: Sequence[str],
    num_samples: int = 10_000,
    max_length: int = 12,
    temperature: float = 1.0,
    batch_size: int = 512,
    generator=None,
    checkpoint_step: int = 100,
    verbose: bool = False,
    progress_step: int = 1_000,
    progress_callback: ProgressCallback | None = None,
) -> RandomGenerationEvaluation:
    r"""用同一批随机样本同时计算质量和测试集覆盖率。
    :param model: 实现 `generate()` 的密码模型
    :param test_passwords: 无重复的测试集密码序列
    :param num_samples: 随机生成的样本数量
    :param max_length: 每条生成文本允许的最大字符数
    :param temperature: 传递给模型的温度参数
    :param batch_size: 每次调用模型批量生成的样本数量
    :param generator: 可选的随机数生成器
    :param checkpoint_step: 每隔多少次采样记录一个覆盖率检查点
    :param verbose: 是否打印采样进度，不打印生成文本
    :param progress_step: verbose 模式下的进度输出间隔
    :param progress_callback: 可选实时回调，接收已完成样本数和当前速度/进度
    :return: 同一批样本对应的质量摘要和覆盖率曲线点

    函数逐 batch 生成并立即更新合法计数、不同合法密码集合、测试集命中集合
    和覆盖率检查点，不保存完整样本列表。这样质量指标与覆盖率仍严格来自同一批
    随机样本，同时把主要内存规模从 `num_samples` 降到 `batch_size`。

    为精确计算合法唯一率，函数仍需保存所有不同合法密码；覆盖率曲线也会保存
    `num_samples // checkpoint_step` 个检查点。因此极高多样性或过密检查点仍会
    消耗 CPU 内存，但不会再保留每一次重复采样得到的字符串。
    """

    if num_samples <= 0:
        raise ValueError("num_samples 必须大于 0")
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step 必须大于 0")
    if progress_step <= 0:
        raise ValueError("progress_step 必须大于 0")
    test_set = set(test_passwords)
    if not test_set:
        raise ValueError("测试集不能为空")
    if len(test_set) != len(test_passwords):
        raise ValueError("测试集包含重复密码")

    legal_count = 0
    distinct_legal: set[str] = set()
    seen_hits: set[str] = set()
    points: list[CoveragePoint] = []
    completed = 0
    start_time = time.perf_counter()

    while completed < num_samples:
        current_batch_size = min(batch_size, num_samples - completed)
        generated_batch = _generate_password_batch(
            model,
            current_batch_size,
            max_length,
            temperature,
            generator,
        )

        # 按原采样顺序逐项更新，确保跨 batch 的检查点与完整列表算法完全一致。
        for password in generated_batch:
            completed += 1
            if _is_legal_password(password, max_length):
                legal_count += 1
                distinct_legal.add(password)
            if password in test_set:
                seen_hits.add(password)
            if completed % checkpoint_step == 0:
                coverage = len(seen_hits) / len(test_set)
                points.append(
                    CoveragePoint(
                        attempts=completed,
                        hits=len(seen_hits),
                        coverage=coverage,
                        efficiency=coverage / completed,
                    )
                )

        if progress_callback is not None:
            elapsed = max(time.perf_counter() - start_time, 1e-12)
            progress_callback(
                completed,
                {
                    "progress": completed / num_samples,
                    "samples_per_second": completed / elapsed,
                },
            )
        if verbose:
            crossed_progress_step = completed % progress_step < current_batch_size
            if crossed_progress_step or completed == num_samples:
                print(f"Generated {completed} / {num_samples} samples")

    quality = GenerationQualitySummary(
        total_samples=completed,
        legal_samples=legal_count,
        distinct_legal_samples=len(distinct_legal),
        legal_rate=legal_count / completed,
        legal_unique_rate=len(distinct_legal) / legal_count if legal_count else 0.0,
    )
    return RandomGenerationEvaluation(
        quality=quality,
        coverage=points,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """对已排序的数值按线性插值计算一个百分位数。"""

    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    )


def summarize_values(values: Sequence[float]) -> MetricSummary:
    r"""计算一组 surprisal 数值的基本分布统计量。
    :param values: 待统计的有限浮点数序列
    :return: 数量、均值、中位数、P90/P95/P99、最小值和最大值
    :raise ValueError: 输入为空或含有非有限值
    """

    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("统计数据不能为空")
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("统计数据必须全部是有限数值")
    ordered = sorted(numbers)
    return MetricSummary(
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        median=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def summarize_surprisal(
    passwords: Sequence[str],
    surprisal_bits: Sequence[float],
) -> SurprisalSummary:
    r"""统计测试集密码的 raw surprisal 和 bits/token。
    :param passwords: 测试集密码序列
    :param surprisal_bits: 与密码顺序对应、包含 EOS 的 surprisal bits
    :return: 两种尺度下的分布统计量
    :raise ValueError: 输入数量不一致、为空或分数非法

    每条密码的 token 数按“字符数 + 1 个 EOS”计算。空密码仍有一个 EOS，
    因此归一化时不会出现除零。
    """

    password_list = list(passwords)
    score_list = list(surprisal_bits)
    if len(password_list) != len(score_list):
        raise ValueError("密码和 surprisal 分数的数量必须一致")
    if not password_list:
        raise ValueError("surprisal 数据不能为空")
    if not all(isinstance(password, str) for password in password_list):
        raise TypeError("passwords 中的每个元素必须是 str")
    raw_values = [float(score) for score in score_list]
    raw_summary = summarize_values(raw_values)
    token_counts = [len(password) + 1 for password in password_list]
    normalized_values = [score / count for score, count in zip(raw_values, token_counts)]
    return SurprisalSummary(
        count=len(password_list),
        raw_surprisal_bits=raw_summary,
        bits_per_token=summarize_values(normalized_values),
    )


def _prepare_plot_values(
    results: Mapping[str, Sequence[float]],
) -> tuple[list[str], list[list[float]]]:
    """校验并复制绘图数据，避免绘图过程消费调用方的可迭代对象。"""

    if not results:
        raise ValueError("绘图数据不能为空")
    names = list(results)
    values = [list(map(float, results[name])) for name in names]
    for model_values in values:
        summarize_values(model_values)
    return names, values


def plot_surprisal_histogram(
    results: Mapping[str, Sequence[float]],
    ax=None,
    bins: int = 50,
    value_label: str = "Surprisal (bits)",
):
    r"""绘制多个模型的 surprisal 分布直方图。
    :param results: 模型名称到逐条 surprisal bits 的映射
    :param ax: 可选的 matplotlib 坐标轴
    :param bins: 直方图分箱数
    :param value_label: x-axis label, for example ``Surprisal (bits/token)``
    :return: 绘制完成的 matplotlib 坐标轴

    直方图用于观察整体分布和长尾。多个模型使用半透明叠加，输入应当是
    raw surprisal 或 bits/token 中的一种，不在函数内混合两种尺度。
    """

    if bins <= 0:
        raise ValueError("bins 必须大于 0")
    names, values = _prepare_plot_values(results)
    if ax is None:
        _, ax = plt.subplots()
    for name, model_values in zip(names, values):
        ax.hist(model_values, bins=bins, alpha=0.45, density=True, label=name)
    ax.set_title("Surprisal Distribution")
    ax.set_xlabel(value_label)
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()
    return ax


def plot_surprisal_boxplot(
    results: Mapping[str, Sequence[float]],
    ax=None,
    value_label: str = "Surprisal (bits)",
):
    r"""绘制多个模型的 surprisal 箱线图。
    :param results: 模型名称到逐条 surprisal bits 的映射
    :param ax: 可选的 matplotlib 坐标轴
    :param value_label: y-axis label, for example ``Surprisal (bits/token)``
    :return: 绘制完成的 matplotlib 坐标轴

    箱线图突出显示中位数、四分位距和异常长尾，适合直接比较模型之间的
    分布位置；调用方应保证所有模型使用同一 surprisal 尺度。
    """

    names, values = _prepare_plot_values(results)
    if ax is None:
        _, ax = plt.subplots()
    ax.boxplot(values, tick_labels=names, showfliers=False)
    ax.set_title("Surprisal Distribution Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel(value_label)
    ax.grid(True, axis="y")
    return ax


def read_test_passwords(
    path: str | Path,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[str]:
    r"""读取测试密码，并可选地确定性抽取一个无重复子集。
    :param path: 测试集文本文件路径，每行一条密码
    :param sample_size: 抽样数量；为 `None` 时返回全部测试密码
    :param seed: 抽样使用的独立随机种子，不修改全局随机状态
    :return: 测试密码列表，顺序由文件或固定种子抽样决定
    :raise ValueError: 测试集有重复、抽样数量非法或超过测试集大小
    """

    passwords = Path(path).read_text(encoding="utf-8").splitlines()
    if len(passwords) != len(set(passwords)):
        raise ValueError("测试集包含重复密码")
    if sample_size is None:
        return passwords
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")
    if sample_size > len(passwords):
        raise ValueError("sample_size 不能超过测试集大小")
    return random.Random(seed).sample(passwords, sample_size)


def coverage_curve(
    candidates: Sequence[GenerationCandidate],
    test_passwords: Sequence[str],
    checkpoint_step: int = 100,
) -> list[CoveragePoint]:
    r"""按搜索顺序计算测试集的累计覆盖率和搜索效率曲线。
    :param candidates: 已按搜索顺序排列的生成候选
    :param test_passwords: 无重复的测试集密码序列
    :param checkpoint_step: 每隔多少次搜索记录一个检查点
    :return: 从第 `checkpoint_step` 次开始的覆盖率检查点列表
    :raise ValueError: 检查点非法、测试集为空/重复或候选重复

    候选和测试集都按字符串去重约束处理，但不会改变候选顺序。每个检查点
    的 `coverage` 是截至该次搜索的累计命中比例，`efficiency` 是该比例除以
    搜索次数，用于观察搜索预算增加后的边际效率变化。
    """

    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step 必须大于 0")
    test_set = set(test_passwords)
    if not test_set:
        raise ValueError("测试集不能为空")
    if len(test_set) != len(test_passwords):
        raise ValueError("测试集包含重复密码")

    seen_candidates: set[str] = set()
    seen_hits: set[str] = set()
    points: list[CoveragePoint] = []
    # 按候选产生顺序累计去重命中，在固定步长处快照覆盖率和单位搜索效率。
    for attempts, candidate in enumerate(candidates, start=1):
        text = candidate.text
        if text in seen_candidates:
            raise ValueError("候选结果包含重复密码")
        seen_candidates.add(text)
        if text in test_set:
            seen_hits.add(text)
        if attempts % checkpoint_step == 0:
            # coverage 是累计命中比例；efficiency 保留搜索次数尺度，不换算为预算比例。
            coverage = len(seen_hits) / len(test_set)
            points.append(
                CoveragePoint(
                    attempts=attempts,
                    hits=len(seen_hits),
                    coverage=coverage,
                    efficiency=coverage / attempts,
                )
            )
    return points


def random_generation_coverage_curve(
    generated_passwords: Sequence[str],
    test_passwords: Sequence[str],
    checkpoint_step: int = 100,
) -> list[CoveragePoint]:
    r"""按随机采样顺序计算测试集累计覆盖率和搜索效率。
    :param generated_passwords: 随机生成文本，可以包含重复值
    :param test_passwords: 无重复的测试集密码序列
    :param checkpoint_step: 每隔多少次采样记录一个检查点
    :return: 从第 `checkpoint_step` 次开始的覆盖率检查点列表

    与搜索候选不同，随机采样允许重复；横轴仍然是实际采样次数，命中统计只对
    测试集密码去重。因此重复样本会消耗采样预算，但不会重复增加覆盖率。
    """

    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step 必须大于 0")
    test_set = set(test_passwords)
    if not test_set:
        raise ValueError("测试集不能为空")
    if len(test_set) != len(test_passwords):
        raise ValueError("测试集包含重复密码")
    if not all(isinstance(password, str) for password in generated_passwords):
        raise TypeError("generated_passwords 中的每个元素必须是 str")

    seen_hits: set[str] = set()
    points: list[CoveragePoint] = []
    # 保留重复样本的采样次数，同时只累计测试集中的不同命中。
    for attempts, password in enumerate(generated_passwords, start=1):
        if password in test_set:
            seen_hits.add(password)
        if attempts % checkpoint_step == 0:
            coverage = len(seen_hits) / len(test_set)
            points.append(
                CoveragePoint(
                    attempts=attempts,
                    hits=len(seen_hits),
                    coverage=coverage,
                    efficiency=coverage / attempts,
                )
            )
    return points


def _plot_coverage_curve(
    results: dict[str, Sequence[CoveragePoint]],
    ax=None,
    *,
    title: str,
    attempts_label: str,
):
    r"""绘制累计覆盖率曲线的共享实现。"""

    if ax is None:
        _, ax = plt.subplots()
    for name, points in results.items():
        ax.plot(
            [point.attempts for point in points],
            [point.coverage for point in points],
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel(attempts_label)
    ax.set_ylabel("Cumulative Coverage")
    ax.grid(True)
    ax.legend()
    return ax


def plot_coverage_curve(
    results: dict[str, Sequence[CoveragePoint]],
    ax=None,
):
    r"""绘制累计覆盖率随搜索次数变化的曲线。
    :param results: 模型名称到覆盖率检查点序列的映射
    :param ax: 可选的 matplotlib 坐标轴；不传入时新建坐标轴
    :return: 绘制完成的 matplotlib 坐标轴

    横轴直接使用检查点中的 `attempts`，纵轴使用 `coverage`。函数只负责
    将已经计算好的评测结果可视化，不重新统计候选命中，也不改变结果对象。
    """

    return _plot_coverage_curve(
        results,
        ax=ax,
        title="Cumulative Coverage vs. Search Attempts",
        attempts_label="Search Attempts",
    )


def plot_random_coverage_curve(
    results: dict[str, Sequence[CoveragePoint]],
    ax=None,
):
    r"""绘制随机采样累计覆盖率随采样次数变化的曲线。
    :param results: 模型名称到随机采样覆盖率检查点序列的映射
    :param ax: 可选的 matplotlib 坐标轴；不传入时新建坐标轴
    :return: 绘制完成的 matplotlib 坐标轴

    随机采样允许重复，因此横轴表示实际采样次数而不是去重后的候选数；
    单独的标题和坐标轴标签用于避免把随机采样误读为搜索过程。
    """

    return _plot_coverage_curve(
        results,
        ax=ax,
        title="Cumulative Coverage vs. Sampling Attempts",
        attempts_label="Sampling Attempts",
    )


def plot_efficiency_curve(
    results: dict[str, Sequence[CoveragePoint]],
    ax=None,
):
    r"""绘制累计覆盖率除以搜索次数随搜索次数变化的曲线。
    :param results: 模型名称到覆盖率检查点序列的映射
    :param ax: 可选的 matplotlib 坐标轴；不传入时新建坐标轴
    :return: 绘制完成的 matplotlib 坐标轴

    横轴使用搜索次数 `attempts`，纵轴使用 `efficiency`。它与
    :func:`plot_coverage_curve` 使用同一批检查点，因此两张图可以直接比较
    同一个搜索过程的累计效果和单位搜索效率。
    """

    if ax is None:
        _, ax = plt.subplots()
    for name, points in results.items():
        ax.plot(
            [point.attempts for point in points],
            [point.efficiency for point in points],
            label=name,
        )
    ax.set_title("Search Efficiency vs. Search Attempts")
    ax.set_xlabel("Search Attempts")
    ax.set_ylabel("Cumulative Coverage / Search Attempts")
    ax.grid(True)
    ax.legend()
    return ax


def plot_random_efficiency_curve(
    results: dict[str, Sequence[CoveragePoint]],
    ax=None,
):
    r"""绘制随机采样效率随采样次数变化的曲线。"""

    if ax is None:
        _, ax = plt.subplots()
    for name, points in results.items():
        ax.plot(
            [point.attempts for point in points],
            [point.efficiency for point in points],
            label=name,
        )
    ax.set_title("Sampling Efficiency vs. Sampling Attempts")
    ax.set_xlabel("Sampling Attempts")
    ax.set_ylabel("Cumulative Coverage / Sampling Attempts")
    ax.grid(True)
    ax.legend()
    return ax


def plot_generation_quality(
    results: Mapping[str, GenerationQualitySummary],
    ax=None,
    width: float = 0.15,
):
    r"""绘制随机生成样本的合法率和合法唯一率分组柱状图。
    :param results: 模型名称到随机生成质量摘要的映射
    :param ax: 可选的 matplotlib 坐标轴；不传入时新建坐标轴
    :param width: 每根柱的宽度
    :return: 绘制完成的 matplotlib 坐标轴

    两个指标都是比例，因此纵轴固定为 0 到 1。每个模型使用两根并列柱，便于直接
    比较生成结果是否符合规则，以及合法结果是否集中重复。
    """

    if not results:
        raise ValueError("绘图数据不能为空")
    if width <= 0:
        raise ValueError("width 必须大于 0")
    if ax is None:
        _, ax = plt.subplots()
    names = list(results)
    legal_rates = [results[name].legal_rate for name in names]
    unique_rates = [results[name].legal_unique_rate for name in names]
    positions = list(range(len(names)))
    # 两组柱共享模型位置，分别展示合法率和合法样本内部的唯一率。
    ax.bar(
        [position - width / 2 for position in positions],
        legal_rates,
        width=width,
        label="Legal Rate",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        unique_rates,
        width=width,
        label="Legal Unique Rate",
    )
    ax.set_xticks(positions, names)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Random Generation Quality")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rate")
    ax.grid(True, axis="y")
    ax.legend()
    return ax


__all__ = [
    "CoveragePoint",
    "MetricSummary",
    "SurprisalSummary",
    "GenerationQualitySummary",
    "RandomGenerationEvaluation",
    "EvaluationArtifactPaths",
    "evaluation_artifact_paths",
    "save_evaluation_summary",
    "save_surprisal_npz",
    "save_coverage_npz",
    "read_test_passwords",
    "coverage_curve",
    "random_generation_coverage_curve",
    "summarize_values",
    "summarize_surprisal",
    "plot_coverage_curve",
    "plot_efficiency_curve",
    "generate_random_passwords",
    "summarize_generation_quality",
    "evaluate_random_generation",
    "evaluate_random_generation_with_coverage",
    "plot_generation_quality",
    "plot_random_coverage_curve",
    "plot_random_efficiency_curve",
    "plot_surprisal_histogram",
    "plot_surprisal_boxplot",
]
