# PassMini

> 基于 PyTorch 的轻量字符级密码建模实验

PassMini 将密码视为字符序列，用自回归模型预测下一个字符来学习密码分布。从 Bigram 出发，扩展到固定上下文 MLP、GRU、TCN 和 Transformer，并搭建用于模型评估和交互的 Streamlit 应用。

## Streamlit 应用
[点击此处访问](https://passmini.streamlit.app)

- **Warehouse**：存储所有模型及参数量等信息，可从中选择模型使用。
- **Library**：展示训练曲线、分布直方图等，可以评估模型能力。
- **Playground**：用模型玩交互式小游戏。

## 模型

| 架构        | 建模机制     | 角色             |
|-------------|--------------|------------------|
| Bigram      | 字符转移计数 | 统计模型基线     |
| MLP         | 马尔科夫假设 | 神经网络模型基线 |
| GRU         | 隐藏状态     | 轻量序列模型     |
| TCN         | 扩张因果卷积 | 有限感受野模型   |
| Transformer | 自注意力     | 全前缀序列模型   |

- 所有模型使用同一字符级 tokenizer 和数据集。
- 所有模型共享生成、评分、搜索等接口。
- 神经网络模型提供 Low-Medium-High 三个档位。
- 神经网络模型都执行字符 Embedding，并使用权重绑定。

## 建模流程

```text
密码序列
   | tokenizer
   ↓
[BOS] + tokens
   | 自回归模型
   ↓
概率分布
   ├── 交叉熵损失
   ├── 概率与 Surprisal
   ├── 随机采样生成
   └── 束搜索与堆搜索
```

其中
$\text{Surprisal}(x) = -\log_2 P(x)$
表示密码在模型建模下的意外程度，或理解为估计猜对需要的猜测次数，可以大致体现：
- 某个模型对密码分布的拟合度
- 某条密码的安全性（不常见性）

## 指标

PassMini 提供以下可视化和比较指标

- 训练、验证损失和二者差值
- 总 Surprisal 与 token 平均 Surprisal 分布
- 生成合法率、唯一率
- 随机采样和搜索在测试集上的覆盖率
- Pairwise 获胜率对比（比谁 Surprisal 低）
- 结合参数量、FLOPs 和质量指标的 Model Zoo。

## 本地运行

推荐用 Python 3.12

```bash
git clone https://github.com/donggengmo-ad/PassMini.git
cd PassMini

python -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt

streamlit run app/app.py
```

仓库已包含前端展示需要的模型和必要数据。

## 项目结构
主要文件结构如下。
```text
PassMini/
├── app/
│   ├── app.py                 # Streamlit 入口
│   ├── requirements.txt       # Community Cloud 运行依赖
│   ├── app_pages/             # Warehouse / Library / Playground 页面
│   ├── frontend/              # 数据加载、交互逻辑和可视化
│   └── artifacts/             # 模型与评测产物
├── .streamlit/config.toml     # 应用主题配置
├── scripts/
|   ├── tokenizer.py           # 字符级 tokenizer
|   ├── data.py                # 密码数据处理
│   ├── models.py              # 各种自回归模型
│   ├── training.py            # 训练工具
│   ├── inference.py           # 加载模型、评分、批量生成与搜索接口
│   └── evaluation.py          # 覆盖率、统计摘要与 NPZ 导出
├── autorg_*.ipynb             # 训练与评测用的 notebook
└── 开发日记.md                # 开发时遇到的一些问题和解决记录
```

## 训练与复现
项目根目录下各个 `.ipynb` 格式 notebook 文件复用一套流水线，对各个模型分别训练、统一评估。

开发验证命令：

```bash
python -m pip install -r requirements.in
python -m pytest -q
python -m compileall -q app scripts tests utils
```

## 数据与安全
本项目仅用于学习、研究或安全防护目的，请勿用于密码攻击等违法用途。
- 仓库没有上传原始密码数据集。
- 您在 Playground 中的任何输入不会被记录。
- 项目仅支持离线生成评分或交互游戏，不支持在线登录或密码爆破。
- Surprisal 分数只能提供安全性估计，不具备现实安全保证。

## 致谢
- PassMini 的问题设定受到 [PassGPT](https://arxiv.org/abs/2306.01545) 启发。
- 使用了 GPT-5.6 Sol 辅助开发和调试
