"""部署模型目录及其路径约定。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = APP_ROOT / "artifacts" / "catalog.json"


@dataclass(frozen=True)
class ModelRecord:
    """描述一个可展示或可推理的部署模型。"""

    id: str
    tier: str
    model_type: str
    display_name: str
    artifact_dir: Path
    parameter_count: int
    color: str
    enabled: bool = True

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "model.pt"

    @property
    def tokenizer_path(self) -> Path:
        return self.artifact_dir / "tokenizer.json"

    @property
    def inference_path(self) -> Path:
        return self.artifact_dir / "inference.json"

    @property
    def history_path(self) -> Path:
        return self.artifact_dir / "history.json"

    @property
    def config_path(self) -> Path:
        return self.artifact_dir / "config.json"

    @property
    def evaluation_dir(self) -> Path:
        return self.artifact_dir / "evaluation"

    @property
    def inference_ready(self) -> bool:
        """返回实时推理需要的三个文件是否齐全。"""

        return all(
            path.is_file()
            for path in (self.model_path, self.tokenizer_path, self.inference_path)
        )


@dataclass(frozen=True)
class ModelCatalog:
    """保存按固定顺序排列的全部模型记录。"""

    models: tuple[ModelRecord, ...]

    def by_id(self, model_id: str) -> ModelRecord:
        """根据稳定 ID 返回模型，不存在时抛出明确错误。"""

        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(f"catalog 中不存在模型 {model_id!r}")

    def select(self, model_ids: list[str] | tuple[str, ...]) -> list[ModelRecord]:
        """保持用户选择顺序返回模型记录。"""

        return [self.by_id(model_id) for model_id in model_ids]

    @property
    def enabled_models(self) -> tuple[ModelRecord, ...]:
        return tuple(model for model in self.models if model.enabled)


def _require_str(payload: Mapping, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"catalog 的 {key} 必须是非空字符串")
    return value


def _resolve_artifact_dir(catalog_path: Path, raw_path: str) -> Path:
    """将相对路径限制在 app 目录内，避免 catalog 指向任意文件。"""

    app_root = catalog_path.resolve().parent.parent
    artifact_dir = (app_root / raw_path).resolve()
    if not artifact_dir.is_relative_to(app_root):
        raise ValueError("artifact_dir 必须位于 app 目录内")
    return artifact_dir


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> ModelCatalog:
    """读取并校验前端模型目录。"""

    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("catalog 顶层必须包含 models 列表")

    records: list[ModelRecord] = []
    seen_ids: set[str] = set()
    for item in payload["models"]:
        if not isinstance(item, dict):
            raise ValueError("catalog 中的模型记录必须是对象")
        model_id = _require_str(item, "id")
        if model_id in seen_ids:
            raise ValueError(f"catalog 模型 ID 重复: {model_id}")
        seen_ids.add(model_id)
        raw_parameter_count = item.get("parameter_count")
        if not isinstance(raw_parameter_count, int) or raw_parameter_count < 0:
            raise ValueError("parameter_count 必须是非负整数")
        raw_enabled = item.get("enabled", True)
        if not isinstance(raw_enabled, bool):
            raise ValueError("enabled 必须是布尔值")
        records.append(
            ModelRecord(
                id=model_id,
                tier=_require_str(item, "tier"),
                model_type=_require_str(item, "model_type"),
                display_name=_require_str(item, "display_name"),
                artifact_dir=_resolve_artifact_dir(
                    catalog_path, _require_str(item, "artifact_dir")
                ),
                parameter_count=raw_parameter_count,
                color=_require_str(item, "color"),
                enabled=raw_enabled,
            )
        )
    return ModelCatalog(tuple(records))
