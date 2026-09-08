"""Warehouse 的元信息读取和模型资源加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from scripts.inference import load_inference_model
from scripts.models import PasswordModel
from scripts.tokenizer import CharTokenizer

from .catalog import ModelRecord


@dataclass(frozen=True)
class ModelMetadata:
    """保存 Warehouse 页面直接展示的模型信息。"""

    inference: dict
    history: dict
    files: dict[str, bool]
    epochs: int | None
    best_validation_loss: float | None


def _read_optional_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 顶层必须是 JSON 对象")
    return value


@st.cache_data(show_spinner=False, max_entries=32)
def read_model_metadata(record: ModelRecord) -> ModelMetadata:
    """读取一个模型的静态元信息，并缓存跨 rerun 复用。"""

    inference = _read_optional_json(record.inference_path)
    history = _read_optional_json(record.history_path)
    train_loss = history.get("train_loss")
    valid_loss = history.get("valid_loss")
    epochs = len(train_loss) if isinstance(train_loss, list) else None
    best_validation_loss = (
        min(float(value) for value in valid_loss)
        if isinstance(valid_loss, list) and valid_loss
        else None
    )
    files: dict[str, bool] = {
        "model": record.model_path.is_file(),
        "tokenizer": record.tokenizer_path.is_file(),
        "inference": record.inference_path.is_file(),
        "surprisal": (record.evaluation_dir / "surprisal.npz").is_file(),
        "random coverage": (record.evaluation_dir / "random_coverage.npz").is_file(),
        "best-first coverage": (
            record.evaluation_dir / "best_first_coverage.npz"
        ).is_file(),
        "summary": (record.evaluation_dir / "summary.json").is_file(),
    }
    if record.tier != "baseline":
        files["history"] = record.history_path.is_file()
    return ModelMetadata(
        inference=inference,
        history=history,
        files=files,
        epochs=epochs,
        best_validation_loss=best_validation_loss,
    )


@st.cache_resource(show_spinner=False, max_entries=12)
def load_runtime_model(
    model_id: str,
    model_path: str,
    tokenizer_path: str,
    inference_path: str,
) -> tuple[PasswordModel, CharTokenizer]:
    """按模型 ID 缓存 CPU 推理模型；路径同时进入缓存键。"""

    del model_id
    return load_inference_model(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        config_path=inference_path,
        device="cpu",
    )


def get_runtime_model(record: ModelRecord) -> tuple[PasswordModel, CharTokenizer]:
    """校验 artifact 后返回缓存的模型和 tokenizer。"""

    if not record.inference_ready:
        raise FileNotFoundError(f"{record.display_name} 的推理 artifact 不完整")
    return load_runtime_model(
        record.id,
        str(record.model_path),
        str(record.tokenizer_path),
        str(record.inference_path),
    )
