"""将训练工作区中的可部署文件同步到 Streamlit artifact 目录。
运行命令
```bash
conda run --no-capture-output -n passmini python -m app.package_artifacts
```
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from zipfile import is_zipfile

import numpy as np

from app.frontend.catalog import (
    DEFAULT_CATALOG_PATH,
    ModelCatalog,
    ModelRecord,
    load_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILENAMES = ("model.pt", "tokenizer.json", "inference.json", "history.json")
EVALUATION_FILENAMES = (
    "surprisal.npz",
    "random_coverage.npz",
    "best_first_coverage.npz",
    "summary.json",
)
COVERAGE_MAX_POINTS = 2_500
COVERAGE_BUDGETS = {
    "random_coverage.npz": (100_000_000,),
    "best_first_coverage.npz": (500_000,),
}
OBSOLETE_MODEL_FILENAMES = ("config.json",)
OBSOLETE_EVALUATION_FILENAMES = ("best_first.json",)


def _source_model_dir(record: ModelRecord, output_root: Path) -> Path:
    if record.tier == "baseline":
        return output_root / record.model_type
    return output_root / record.tier / record.model_type


def _atomic_copy(source: Path, destination: Path) -> None:
    """先复制并校验临时文件，再原子替换部署 artifact。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(source, temporary)
        if destination.suffix == ".npz":
            # np.load 按需解压字段；逐项读取可发现复制期间产生的截断或 CRC 错误。
            if not is_zipfile(temporary):
                raise ValueError(f"复制后的 NPZ 文件无效: {source}")
            with np.load(temporary, allow_pickle=False) as payload:
                for name in payload.files:
                    np.asarray(payload[name])
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _coverage_indices(
    attempts: np.ndarray,
    max_points: int,
    preserved_budgets: tuple[int, ...],
) -> np.ndarray:
    """等距选择覆盖率点，并保留首尾及前端总览使用的预算检查点。"""

    if max_points < 2:
        raise ValueError("coverage max_points 必须至少为 2")
    size = attempts.size
    if size <= max_points:
        return np.arange(size)

    required = {0, size - 1}
    for budget in preserved_budgets:
        index = int(np.searchsorted(attempts, budget, side="right") - 1)
        if index >= 0:
            required.add(index)

    # 为强制保留点预留容量；linspace 负责覆盖其余时间轴且不会引入随机性。
    uniform_count = max_points - len(required) + 2
    uniform = np.linspace(0, size - 1, num=uniform_count, dtype=np.int64)
    return np.asarray(sorted(required | set(uniform.tolist())), dtype=np.int64)


def _atomic_copy_coverage(
    source: Path,
    destination: Path,
    max_points: int = COVERAGE_MAX_POINTS,
) -> None:
    """校验并压缩 coverage NPZ，避免前端保存和缓存无用的百万级点位。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with np.load(source, allow_pickle=False) as payload:
            required = {"test_size", "attempts", "coverage"}
            if not required.issubset(payload.files):
                raise ValueError(f"coverage NPZ 缺少必需字段: {source}")
            test_size = np.asarray(payload["test_size"]).copy()
            attempts = np.asarray(payload["attempts"]).copy()
            coverage = np.asarray(payload["coverage"]).copy()

        if test_size.ndim != 0 or attempts.ndim != 1 or coverage.ndim != 1:
            raise ValueError(f"coverage NPZ 字段维度无效: {source}")
        if attempts.size == 0 or attempts.size != coverage.size:
            raise ValueError(f"coverage NPZ 点位为空或长度不一致: {source}")
        if np.any(np.diff(attempts) <= 0):
            raise ValueError(f"coverage attempts 必须严格递增: {source}")
        if not np.all(np.isfinite(coverage)):
            raise ValueError(f"coverage 必须全部为有限值: {source}")

        indices = _coverage_indices(
            attempts,
            max_points,
            COVERAGE_BUDGETS.get(source.name, ()),
        )
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                test_size=test_size,
                attempts=attempts[indices],
                coverage=coverage[indices],
            )
        if not is_zipfile(temporary):
            raise ValueError(f"压缩后的 coverage NPZ 文件无效: {source}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _prune_obsolete_artifacts(catalog_path: Path, catalog: ModelCatalog) -> None:
    """删除旧版本打包过、但当前 App 已无消费者的部署文件。"""

    for metadata_file in catalog_path.parent.rglob(".DS_Store"):
        metadata_file.unlink()
    for record in catalog.enabled_models:
        for filename in OBSOLETE_MODEL_FILENAMES:
            (record.artifact_dir / filename).unlink(missing_ok=True)
        for filename in OBSOLETE_EVALUATION_FILENAMES:
            (record.evaluation_dir / filename).unlink(missing_ok=True)
        if record.tier == "baseline":
            record.history_path.unlink(missing_ok=True)


def _evaluation_source(
    record: ModelRecord,
    output_root: Path,
    filename: str,
) -> Path | None:
    """返回按 tier/model 分层的单模型评测文件。"""

    source = output_root / "evaluation" / record.tier / record.model_type / filename
    return source if source.is_file() else None


def package_artifacts(
    output_root: str | Path = REPOSITORY_ROOT / "output",
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    include_evaluation: bool = True,
) -> dict[str, list[str]]:
    """同步启用模型的最小部署文件，并返回逐模型复制清单。"""

    source_root = Path(output_root)
    resolved_catalog_path = Path(catalog_path)
    catalog = load_catalog(resolved_catalog_path)
    _prune_obsolete_artifacts(resolved_catalog_path, catalog)
    report: dict[str, list[str]] = {}
    for record in catalog.enabled_models:
        copied: list[str] = []
        model_source = _source_model_dir(record, source_root)
        for filename in MODEL_FILENAMES:
            if record.tier == "baseline" and filename == "history.json":
                continue
            source = model_source / filename
            if source.is_file():
                _atomic_copy(source, record.artifact_dir / filename)
                copied.append(filename)

        if include_evaluation:
            for filename in EVALUATION_FILENAMES:
                source = _evaluation_source(record, source_root, filename)
                if source is not None:
                    destination = record.evaluation_dir / filename
                    if filename in COVERAGE_BUDGETS:
                        _atomic_copy_coverage(source, destination)
                    else:
                        _atomic_copy(source, destination)
                    copied.append(f"evaluation/{filename}")
        report[record.id] = copied
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="同步 PassMini Streamlit 部署 artifact")
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="只同步模型、tokenizer 和配置，不读取正在生成的评测文件",
    )
    arguments = parser.parse_args()
    report = package_artifacts(include_evaluation=not arguments.models_only)
    for model_id, files in report.items():
        print(f"{model_id}: {len(files)} files")


if __name__ == "__main__":
    main()
