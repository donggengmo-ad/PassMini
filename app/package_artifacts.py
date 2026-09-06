"""将训练工作区中的可部署文件同步到 Streamlit artifact 目录。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from app.frontend.catalog import DEFAULT_CATALOG_PATH, ModelRecord, load_catalog


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILENAMES = ("model.pt", "tokenizer.json", "inference.json", "history.json", "config.json")
EVALUATION_FILENAMES = (
    "surprisal.npz",
    "random_coverage.npz",
    "best_first_coverage.npz",
)


def _source_model_dir(record: ModelRecord, output_root: Path) -> Path:
    if record.tier == "baseline":
        return output_root / record.model_type
    return output_root / record.tier / record.model_type


def _atomic_copy(source: Path, destination: Path) -> None:
    """先写临时文件再替换，避免前端读取到复制一半的 artifact。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _evaluation_source(
    record: ModelRecord,
    output_root: Path,
    filename: str,
) -> Path | None:
    """兼容当前平铺输出和未来 tier/model 分目录，但不改变评测文件内容。"""

    evaluation_root = output_root / "evaluation"
    candidates = (
        evaluation_root / record.tier / record.model_type / filename,
        evaluation_root / record.tier / f"{record.model_type}_{filename}",
        evaluation_root / f"{record.model_type}_{filename}",
    )
    return next((path for path in candidates if path.is_file()), None)


def _collect_summary(record: ModelRecord, output_root: Path) -> dict:
    """从组合摘要中只提取当前模型的条目。"""

    evaluation_root = output_root / "evaluation"
    key = record.model_type.capitalize()
    if record.model_type == "tcn":
        key = "TCN"
    if record.model_type == "gru":
        key = "GRU"

    result: dict = {}
    summary_path = evaluation_root / "surprisal_summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            result["surprisal"] = payload[key]

    quality_path = evaluation_root / "generation_quality.json"
    if quality_path.is_file():
        payload = json.loads(quality_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            result["generation_quality"] = payload[key]
    return result


def package_artifacts(
    output_root: str | Path = REPOSITORY_ROOT / "output",
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    include_evaluation: bool = True,
) -> dict[str, list[str]]:
    """同步启用模型的最小部署文件，并返回逐模型复制清单。"""

    source_root = Path(output_root)
    catalog = load_catalog(catalog_path)
    report: dict[str, list[str]] = {}
    for record in catalog.enabled_models:
        copied: list[str] = []
        model_source = _source_model_dir(record, source_root)
        for filename in MODEL_FILENAMES:
            source = model_source / filename
            if source.is_file():
                _atomic_copy(source, record.artifact_dir / filename)
                copied.append(filename)

        if include_evaluation:
            for filename in EVALUATION_FILENAMES:
                source = _evaluation_source(record, source_root, filename)
                if source is not None:
                    _atomic_copy(source, record.evaluation_dir / filename)
                    copied.append(f"evaluation/{filename}")
            summary = _collect_summary(record, source_root)
            if summary:
                destination = record.evaluation_dir / "summary.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                copied.append("evaluation/summary.json")
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
