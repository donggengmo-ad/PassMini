# PassMini 密码生成与评估器

PassMini 是一个面向 PyTorch、机器学习和深度学习学习的字符级密码语言模型项目。
项目在离线环境中使用公开数据完成数据处理、Bigram 基线、自回归神经模型训练、生成和评测。

## 当前结构

- `scripts/tokenizer.py`、`scripts/data.py`：字符词表、Dataset 和动态 padding。
- `scripts/models.py`：`PasswordModel`、`AutoregressivePasswordModel` 以及 Bigram、GRU、TCN、Transformer。
- `scripts/training.py`：训练、验证、梯度裁剪、scheduler 和 checkpoint。
- `scripts/pipeline.py`：notebook 和脚本共用的配置、数据、模型与 artifact 流水线。
- `scripts/inference.py`、`scripts/evaluation.py`：独立加载、生成、评分、搜索、覆盖率和可视化。
- `scripts/evaluation.py` 可将完整评分等距限点导出为 surprisal NPZ，并以最小字段导出覆盖率 NPZ，
  供后续动态图表读取。
- `autorg_*.ipynb`：canonical 训练和评测入口；notebook 只表达配置、调用和结果展示。
- `app/app.py`、`app/app_pages/`：使用 `st.navigation` 组织的 Streamlit 三层应用入口与页面。
- `app/frontend/`、`app/artifacts/`：共享展示逻辑、交互式 Altair 图表和精选部署 artifact。
- `.streamlit/config.toml`：应用主题；不在页面代码中注入 CSS。

## 运行验证

```bash
conda run --no-capture-output -n passmini python -m pytest -q
conda run --no-capture-output -n passmini python -m compileall -q app scripts tests
```

本地启动 Streamlit：

```bash
conda run --no-capture-output -n passmini streamlit run app/streamlit_app.py
```

训练或评测完成后，将可部署文件同步到 `app/artifacts/`：

```bash
conda run --no-capture-output -n passmini python -m app.package_artifacts
```

阶段 9 后模型配置只接受 `bigram`、`gru`、`tcn`、`transformer`，推理配置统一保存为
`inference.json`。正式 GRU 训练使用 CUDA；单元测试和冒烟实验使用 CPU；Bigram 保持 CPU。

原始数据、处理数据、训练 checkpoint 和 `output/` 工作区不提交 Git。`app/artifacts/` 只保留
前端运行所需的精选模型权重、tokenizer、配置、训练历史和评测数组，可以随应用部署。
