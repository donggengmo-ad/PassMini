"""PassMini 的密码语言模型、公共接口和自回归状态。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .data import PasswordDataset, collate_batch
from .tokenizer import CharTokenizer


@dataclass
class AutoregressiveState:
    """保存自回归模型下一步预测所需的统一状态。

    `next_logits` 始终为 `[B, V]`，`cache` 由具体模型保存自己的 hidden 或
    前缀信息。推理层只传递 cache，不解析其内部结构。
    """

    next_logits: torch.Tensor
    cache: object


class PasswordModel(ABC):
    """规定所有密码模型共享的字符串级应用接口。"""

    tokenizer: CharTokenizer

    @property
    def pad_id(self) -> int:
        """返回 tokenizer 中 PAD 的 id。"""

        return self.tokenizer.pad_id

    @property
    def vocab_size(self) -> int:
        """返回 tokenizer 词表大小。"""

        return self.tokenizer.vocab_size

    @abstractmethod
    def generate(
        self,
        max_length: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> str:
        """生成一条密码字符串。"""
        ...

    @abstractmethod
    def generate_batch(
        self,
        batch_size: int,
        max_length: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[str]:
        """批量生成密码字符串。"""
        ...

    @abstractmethod
    def log_probability(self, text: str) -> torch.Tensor:
        """计算包含 EOS 的单条密码自然对数概率。"""
        ...

    def log_probabilities(self, texts: Sequence[str]) -> torch.Tensor:
        """按输入顺序计算多条密码的自然对数概率。"""

        values = [self.log_probability(text) for text in texts]
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.stack(values)

    def surprisal_bits(self, text: str) -> torch.Tensor:
        """将单条密码的自然对数概率转换为比特惊讶度。"""

        return -self.log_probability(text) / math.log(2.0)

    def surprisal_bits_batch(self, texts: Sequence[str]) -> torch.Tensor:
        """按输入顺序返回多条密码的比特惊讶度。"""

        return -self.log_probabilities(texts) / math.log(2.0)


class AutoregressivePasswordModel(nn.Module, PasswordModel):
    """所有当前自回归密码模型的公共实现。

    子类只需实现 `forward()`、`initial_state()`、`advance_state()` 和
    `log_probability()`；采样、EOS 处理、特殊 token 屏蔽和批量固定形状
    逻辑在这里统一完成。
    """

    model_type: str

    def _mask_pad_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """屏蔽共享 embedding 权重中 PAD 行的梯度。"""

        masked = gradient.clone()
        masked[self.pad_id] = 0
        return masked

    @property
    def device(self) -> torch.device:
        """返回模型执行设备；没有参数的模型默认在 CPU。"""

        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @abstractmethod
    def initial_state(self, batch_size: int) -> AutoregressiveState:
        """消费一批 BOS，返回下一步 logits 和模型缓存。"""
        ...

    @abstractmethod
    def advance_state(
        self,
        state: AutoregressiveState,
        token_ids: torch.Tensor,
    ) -> AutoregressiveState:
        """消费形状 `[B]` 的 token，返回新的下一步状态。"""
        ...

    @abstractmethod
    def _select_cache(self, cache: object, indices: torch.Tensor) -> object:
        """按 batch 索引选择、复制或重排具体模型的缓存。"""
        ...

    @abstractmethod
    def _stack_caches(self, caches: Sequence[object]) -> object:
        """沿 batch 维合并一组结构兼容的具体模型缓存。"""
        ...

    def select_state(
        self,
        state: AutoregressiveState,
        indices: torch.Tensor | Sequence[int],
    ) -> AutoregressiveState:
        r"""按 batch 索引选择、复制或重排自回归状态。

        搜索算法用它把保留下来的父节点缓存复制给多个子节点。`indices`
        可以包含重复值，因此一次选择也能完成 beam 的父状态扩展。
        """

        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=state.next_logits.device,
        )
        if index.ndim != 1:
            raise ValueError("indices 必须是一维整数索引")
        selected_logits = state.next_logits.index_select(0, index)
        return AutoregressiveState(selected_logits, self._select_cache(state.cache, index))

    def stack_states(self, states: Sequence[AutoregressiveState]) -> AutoregressiveState:
        r"""沿 batch 维合并多个结构兼容的自回归状态。

        Best-First Search 会用它合并同一深度的独立堆节点，再进行一次批量
        前推。不同深度的 Transformer KV cache 长度不同，不能直接合并。
        """

        values = list(states)
        if not values:
            raise ValueError("states 不能为空")
        for state in values:
            if state.next_logits.ndim != 2 or state.next_logits.size(1) != self.vocab_size:
                raise ValueError("每个 next_logits 必须是 [B, vocab_size] 张量")
        next_logits = torch.cat([state.next_logits for state in values], dim=0)
        return AutoregressiveState(
            next_logits,
            self._stack_caches([state.cache for state in values]),
        )

    def generate(
        self,
        max_length: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> str:
        """通过统一批量路径生成一条密码。"""

        return self.generate_batch(1, max_length, temperature, generator)[0]

    def _next_token_probs(
        self,
        logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """对 `[B, V]` logits 屏蔽特殊 token 并转换为概率。"""

        if logits.ndim != 2 or logits.size(1) != self.vocab_size:
            raise ValueError("next_logits 必须是 [B, vocab_size] 张量")
        if temperature <= 0 or not isfinite(temperature):
            raise ValueError("temperature 必须是有限正数")
        scores = logits / temperature
        scores = scores.clone()
        invalid_ids = [self.pad_id, self.tokenizer.bos_id, self.tokenizer.unk_id]
        scores[:, invalid_ids] = -torch.inf
        if not torch.isfinite(scores).any(dim=1).all():
            raise ValueError("模型没有可生成的合法 token")
        return torch.softmax(scores, dim=-1)

    def generate_batch(
        self,
        batch_size: int,
        max_length: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[str]:
        r"""使用固定 batch 自回归生成多条密码。

        `[B, 1]` 的当前输入和 `[B, V]` 的下一步 logits 在每轮保持固定形状。
        某行采样到 EOS 后只更新 `finished`，仍保留在 batch 中占位；这样无需
        为不同长度样本建立多个循环，也避免在 GPU 上逐行同步 Python。
        """

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")
        if temperature <= 0 or not isfinite(temperature):
            raise ValueError("temperature 必须是有限正数")

        with torch.inference_mode():
            self.eval()
            device = self.device
            state = self.initial_state(batch_size)
            if state.next_logits.device != device:
                raise ValueError("模型状态与模型设备不一致")
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            generated_ids = torch.full(
                (batch_size, max_length),
                self.pad_id,
                dtype=torch.long,
                device=device,
            )
            for step in range(max_length):
                # [B,V] -> [B]：所有样本一次采样，再用掩码写入当前时间步。
                probs = self._next_token_probs(state.next_logits, temperature)
                next_ids = torch.multinomial(
                    probs,
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
                active = ~finished
                eos_rows = next_ids == self.tokenizer.eos_id
                write_mask = active & ~eos_rows
                generated_ids[write_mask, step] = next_ids[write_mask]
                finished |= eos_rows
                # 仅在仍需下一步 logits 时推进固定 batch，避免末轮构造无用状态。
                if bool(finished.all()) or step + 1 == max_length:
                    break
                state = self.advance_state(state, next_ids)
        return [self.tokenizer.decode(ids) for ids in generated_ids.cpu().tolist()]


def _teacher_forcing_log_probabilities(
    model: AutoregressivePasswordModel,
    texts: Sequence[str],
) -> torch.Tensor:
    """将字符串整理为一个 teacher-forcing batch，并用一次前向计算 `[B]` 分数。"""

    text_list = list(texts)
    if not text_list:
        return torch.empty(0, dtype=torch.float32, device=model.device)
    dataset = PasswordDataset(text_list, model.tokenizer)
    input_ids, target_ids = collate_batch(
        [dataset[index] for index in range(len(dataset))],
        pad_id=model.pad_id,
    )
    input_ids = input_ids.to(model.device)
    target_ids = target_ids.to(model.device)

    # [B,L,V] -> [B,L]：只取每个真实目标 token 的 log probability，再忽略 PAD 求和。
    logits = model(input_ids)
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(target_ids == model.pad_id, 0.0)
    return token_log_probs.sum(dim=-1)


class AutoregressiveBigram(AutoregressivePasswordModel):
    r"""使用 `[V, V]` 转移计数表的字符级 Bigram 基线。"""

    model_type = "bigram"

    def __init__(self, tokenizer: CharTokenizer, alpha: float = 1.0):
        r"""初始化 Bigram 模型。
        :param tokenizer: 分词器
        :param alpha: add-alpha 平滑参数，要求为有限正数
        """

        super().__init__()
        if alpha <= 0 or not isfinite(alpha):
            raise ValueError("alpha 必须是有限正数")
        self.tokenizer = tokenizer
        self.alpha = float(alpha)
        # count 是普通 long 成员：Bigram 只通过 fit/save/load 管理统计量。
        self.count = torch.zeros(
            (tokenizer.vocab_size, tokenizer.vocab_size), dtype=torch.long
        )

    def fit(
        self,
        dataset: PasswordDataset,
        verbose: bool = False,
        batch_size: int = 4096,
    ) -> None:
        """使用 torch.bincount 批量统计所有 token 转移。"""

        if dataset.tokenizer != self.tokenizer:
            raise ValueError("dataset 的 tokenizer 与模型的 tokenizer 不一致")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.count.zero_()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, pad_id=self.pad_id),
        )
        processed = 0
        # 将二维 token 转移编码为一维索引，统计后还原为 [V,V]。
        for input_ids, target_ids in dataloader:
            valid_mask = target_ids != self.pad_id
            pair_index = input_ids[valid_mask] * self.vocab_size + target_ids[valid_mask]
            batch_count = torch.bincount(
                pair_index, minlength=self.vocab_size * self.vocab_size
            )
            self.count += batch_count.reshape(self.count.shape)
            processed += input_ids.size(0)
            if verbose and (processed % 5000 < input_ids.size(0) or processed == len(dataset)):
                print(f"Processed {processed} / {len(dataset)} samples")

    def save(self, path: str | Path) -> None:
        """保存 tokenizer、平滑参数和 count。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "tokenizer": list(self.tokenizer.id_to_token),
                "alpha": self.alpha,
                "count": self.count.cpu(),
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AutoregressiveBigram":
        """从 Bigram artifact 恢复模型。"""

        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or artifact.get("model_type") != "bigram":
            raise ValueError("Bigram 模型文件的 model_type 必须为 'bigram'")
        tokens = artifact.get("tokenizer")
        if not isinstance(tokens, (list, tuple)):
            raise ValueError("Bigram 模型文件缺少 tokenizer")
        tokenizer = CharTokenizer(list(tokens))
        model = cls(tokenizer, alpha=artifact.get("alpha", 1.0))
        count = artifact.get("count")
        expected_shape = (tokenizer.vocab_size, tokenizer.vocab_size)
        if not isinstance(count, torch.Tensor):
            raise ValueError("Bigram 模型文件缺少 count 张量")
        if tuple(count.shape) != expected_shape:
            raise ValueError("count 的形状必须与 tokenizer.vocab_size 匹配")
        if count.dtype != torch.long or bool(torch.any(count < 0)):
            raise ValueError("count 必须是非负整数张量")
        model.count.copy_(count)
        return model

    def next_token_probs(self, current_id: int) -> torch.Tensor:
        """返回给定 token 后的平滑概率分布。"""

        row = self.count[current_id].to(torch.float32)
        return (row + self.alpha) / (row.sum() + self.vocab_size * self.alpha)

    def log_probability(self, text: str) -> torch.Tensor:
        """计算从 BOS 到文本再到 EOS 的自然对数概率。"""

        ids = [self.tokenizer.bos_id, *self.tokenizer.encode(text), self.tokenizer.eos_id]
        value = torch.tensor(0.0)
        for current_id, next_id in zip(ids, ids[1:]):
            value += torch.log(self.next_token_probs(current_id)[next_id])
        return value

    def initial_state(self, batch_size: int) -> AutoregressiveState:
        """批量消费 BOS，返回 Bigram 转移 logits。"""

        current_ids = torch.full((batch_size,), self.tokenizer.bos_id, dtype=torch.long)
        next_logits = torch.log(self.count[current_ids].float() + self.alpha)
        return AutoregressiveState(next_logits=next_logits, cache=current_ids)

    def advance_state(
        self,
        state: AutoregressiveState,
        token_ids: torch.Tensor,
    ) -> AutoregressiveState:
        """以新 token 查询下一批 Bigram 转移行。"""

        current_ids = token_ids.detach().cpu().long()
        next_logits = torch.log(self.count[current_ids].float() + self.alpha)
        return AutoregressiveState(next_logits=next_logits, cache=current_ids)

    def _select_cache(self, cache: object, indices: torch.Tensor) -> torch.Tensor:
        """沿第 0 维选择 Bigram 当前 token。"""

        if not isinstance(cache, torch.Tensor):
            raise ValueError("Bigram state.cache 必须是 current_ids tensor")
        return cache.index_select(0, indices.to(cache.device))

    def _stack_caches(self, caches: Sequence[object]) -> torch.Tensor:
        """沿第 0 维合并 Bigram 当前 token。"""

        if not all(isinstance(cache, torch.Tensor) for cache in caches):
            raise ValueError("Bigram cache 必须全部是 tensor")
        return torch.cat(list(caches), dim=0)


class AutoregressiveMLP(AutoregressivePasswordModel):
    r"""基于固定长度 Markov 上下文的权重共享 MLP 密码模型。

    每个位置只观察包含当前输入在内的最近 ``tau`` 个 token；序列开头不足
    ``tau`` 时使用 PAD 左填充。训练时一次构造所有滑动窗口，推理时只缓存
    最近 ``tau`` 个 token，因此不需要随序列长度增长的状态。
    """

    model_type = "mlp"

    def __init__(
        self,
        tokenizer: CharTokenizer,
        tau: int = 4,
        embedding_dim: int = 64,
        hidden_size: int = 128,
    ):
        r"""初始化固定上下文 MLP。
        :param tokenizer: 分词器对象
        :param tau: 每次预测使用的最近 token 数量
        :param embedding_dim: 字符嵌入和共享输出空间维度
        :param hidden_size: MLP 隐藏层维度

        输出路径固定为 ``tau 个 embedding 拼接 -> hidden -> embedding 空间
        -> embedding.weight.T``，最后一层与输入 embedding 权重共享。
        """

        super().__init__()
        if tau <= 0 or embedding_dim <= 0 or hidden_size <= 0:
            raise ValueError("tau、embedding_dim、hidden_size 必须为正整数")
        self.tokenizer = tokenizer
        self.tau = int(tau)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=self.pad_id
        )
        # 权重共享会让 PAD 行参与输出 logits；屏蔽梯度以保持左填充始终为零向量。
        self.embedding.weight.register_hook(self._mask_pad_gradient)
        self.input_projection = nn.Linear(
            self.tau * self.embedding_dim, self.hidden_size
        )
        self.output_projection = nn.Linear(self.hidden_size, self.embedding_dim)

    def _context_logits(self, context_ids: torch.Tensor) -> torch.Tensor:
        """将 ``[..., tau]`` 上下文批量映射为 ``[..., vocab_size]`` logits。"""

        if context_ids.ndim < 2 or context_ids.size(-1) != self.tau:
            raise ValueError("MLP 上下文最后一维必须等于 tau")
        embeddings = self.embedding(context_ids)
        flattened = embeddings.flatten(start_dim=-2)
        hidden = F.gelu(self.input_projection(flattened))
        projected = self.output_projection(hidden)
        return projected @ self.embedding.weight.T

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """批量前向，输入 ``[B,L]``，输出 ``[B,L,V]``。"""

        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [B,L] 张量")
        # [B,L] -> [B,L,tau]：左侧补 PAD 后一次展开所有位置，避免逐位置 Python 循环。
        padded_ids = F.pad(input_ids, (self.tau - 1, 0), value=self.pad_id)
        context_ids = padded_ids.unfold(dimension=1, size=self.tau, step=1)
        return self._context_logits(context_ids)

    def initial_state(self, batch_size: int) -> AutoregressiveState:
        """消费一批 BOS，缓存左填充后的 ``[B,tau]`` 最近 token。"""

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        context_ids = torch.full(
            (batch_size, self.tau),
            self.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        context_ids[:, -1] = self.tokenizer.bos_id
        return AutoregressiveState(self._context_logits(context_ids), context_ids)

    def advance_state(
        self,
        state: AutoregressiveState,
        token_ids: torch.Tensor,
    ) -> AutoregressiveState:
        """丢弃最旧 token、追加 ``[B]`` 新 token，并批量计算下一步 logits。"""

        if not isinstance(state.cache, torch.Tensor):
            raise ValueError("MLP state.cache 必须是 context_ids tensor")
        if token_ids.ndim != 1 or token_ids.size(0) != state.cache.size(0):
            raise ValueError("token_ids 必须是与 cache batch 相同的 [B] 张量")
        next_ids = token_ids.to(self.device, dtype=torch.long).unsqueeze(-1)
        context_ids = torch.cat([state.cache[:, 1:], next_ids], dim=-1)
        return AutoregressiveState(self._context_logits(context_ids), context_ids)

    def _select_cache(self, cache: object, indices: torch.Tensor) -> torch.Tensor:
        """沿第 0 维选择、复制或重排 MLP 上下文缓存。"""

        if not isinstance(cache, torch.Tensor):
            raise ValueError("MLP state.cache 必须是 context_ids tensor")
        return cache.index_select(0, indices.to(cache.device))

    def _stack_caches(self, caches: Sequence[object]) -> torch.Tensor:
        """沿第 0 维合并固定宽度的 MLP 上下文缓存。"""

        if not all(isinstance(cache, torch.Tensor) for cache in caches):
            raise ValueError("MLP cache 必须全部是 tensor")
        return torch.cat(list(caches), dim=0)

    def log_probabilities(self, texts: Sequence[str]) -> torch.Tensor:
        """使用一次滑动窗口前向批量计算字符串序列的自然对数概率。"""

        return _teacher_forcing_log_probabilities(self, texts)

    def log_probability(self, text: str) -> torch.Tensor:
        """计算从 BOS 到文本再到 EOS 的自然对数概率。"""

        return self.log_probabilities([text])[0]


class AutoregressiveGRU(AutoregressivePasswordModel):
    """使用 Embedding、GRU 和固定权重共享输出层的密码模型。"""

    model_type = "gru"

    def __init__(
        self,
        tokenizer: CharTokenizer,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        r"""初始化 GRU。
        :param tokenizer: 分词器对象
        :param embedding_dim: 字符嵌入向量维度
        :param hidden_size: GRU 隐藏状态维度
        :param num_layers: GRU 层数

        输出路径固定为 `hidden -> Linear(hidden_size, embedding_dim) ->
        embedding.weight.T`，因此不再存在可选 MLP 或 weight_tying 参数。
        """

        super().__init__()
        if embedding_dim <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("embedding_dim、hidden_size、num_layers 必须为正整数")
        self.tokenizer = tokenizer
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=self.pad_id
        )
        # 权重共享会让 PAD 行同时参与输出 logits；显式屏蔽该行梯度以保持 PAD 不训练。
        self.embedding.weight.register_hook(self._mask_pad_gradient)
        self.gru = nn.GRU(
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播，输入 `[B,L]`，输出 `[B,L,V]`。"""

        embeddings = self.embedding(input_ids)
        output, hidden = self.gru(embeddings, hidden)
        projected = self.output_projection(output)
        logits = projected @ self.embedding.weight.T
        return (logits, hidden) if return_hidden else logits

    def initial_state(self, batch_size: int) -> AutoregressiveState:
        """批量消费 BOS，缓存 `[N,B,H]` hidden。"""

        input_ids = torch.full(
            (batch_size, 1), self.tokenizer.bos_id, dtype=torch.long, device=self.device
        )
        logits, hidden = self.forward(input_ids, return_hidden=True)
        return AutoregressiveState(logits[:, -1, :], hidden)

    def advance_state(
        self,
        state: AutoregressiveState,
        token_ids: torch.Tensor,
    ) -> AutoregressiveState:
        """用一列 token 和父 hidden 执行一步 GRU。"""

        if not isinstance(state.cache, torch.Tensor):
            raise ValueError("GRU state.cache 必须是 hidden tensor")
        input_ids = token_ids.to(self.device).unsqueeze(1)
        logits, hidden = self.forward(input_ids, state.cache, return_hidden=True)
        return AutoregressiveState(logits[:, -1, :], hidden)

    def _select_cache(self, cache: object, indices: torch.Tensor) -> torch.Tensor:
        """沿 `[N,B,H]` hidden 的第 1 维选择 batch。"""

        if not isinstance(cache, torch.Tensor):
            raise ValueError("GRU state.cache 必须是 hidden tensor")
        return cache.index_select(1, indices.to(cache.device))

    def _stack_caches(self, caches: Sequence[object]) -> torch.Tensor:
        """沿 `[N,B,H]` hidden 的第 1 维合并 batch。"""

        if not all(isinstance(cache, torch.Tensor) for cache in caches):
            raise ValueError("GRU cache 必须全部是 tensor")
        return torch.cat(list(caches), dim=1)

    def _log_probabilities_from_ids(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按 PasswordDataset 的 input/target 计算 `[B]` 对数概率。"""

        if input_ids.shape != target_ids.shape:
            raise ValueError("input_ids 和 target_ids 的形状必须相同")
        if input_ids.device != target_ids.device:
            raise ValueError("input_ids 和 target_ids 必须在同一设备上")
        logits = self.forward(input_ids, hidden)
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs.masked_fill(target_ids == self.pad_id, 0.0)
        return token_log_probs.sum(dim=-1)

    def log_probabilities(self, texts: Sequence[str]) -> torch.Tensor:
        """使用 teacher-forcing batch 计算字符串序列的对数概率。"""

        text_list = list(texts)
        if not text_list:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        dataset = PasswordDataset(text_list, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(text_list),
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, pad_id=self.pad_id),
        )
        input_ids, target_ids = next(iter(dataloader))
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        return self._log_probabilities_from_ids(input_ids, target_ids)

    def log_probability(self, text: str) -> torch.Tensor:
        """计算单条密码包含 EOS 的自然对数概率。"""

        return self.log_probabilities([text])[0]


@dataclass(frozen=True)
class _TCNCache:
    """保存各卷积层输入端的有限历史窗口。"""

    layer_inputs: tuple[torch.Tensor, ...]


class AutoregressiveTCN(AutoregressivePasswordModel):
    """使用权重绑定和逐层历史缓存的轻量因果卷积密码模型。"""

    model_type = "tcn"

    def __init__(
        self,
        tokenizer: CharTokenizer,
        embedding_dim: int = 64,
        channels: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ):
        r"""初始化因果卷积模型。
        :param tokenizer: 分词器对象
        :param embedding_dim: 字符嵌入和共享输出空间维度
        :param channels: 每层因果卷积的输出通道数
        :param kernel_size: 卷积核大小
        :param dilations: 各层卷积的膨胀系数

        输出路径固定为 `channels -> output_projection -> embedding.weight.T`。
        """

        super().__init__()
        if embedding_dim <= 0 or channels <= 0 or kernel_size <= 0:
            raise ValueError("embedding_dim、channels、kernel_size 必须为正整数")
        if not dilations or any(d <= 0 for d in dilations):
            raise ValueError("dilations 必须是正整数序列")
        self.tokenizer = tokenizer
        self.embedding_dim = int(embedding_dim)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=self.pad_id
        )
        self.embedding.weight.register_hook(self._mask_pad_gradient)
        layers: list[nn.Module] = []
        in_channels = self.embedding_dim
        for dilation in self.dilations:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    self.channels,
                    self.kernel_size,
                    dilation=dilation,
                )
            )
            in_channels = self.channels
        self.convs = nn.ModuleList(layers)
        self.output_projection = nn.Linear(self.channels, self.embedding_dim)

    def _to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """将卷积特征投影到嵌入空间，并复用 embedding 权重输出 logits。"""

        return self.output_projection(hidden) @ self.embedding.weight.T

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行严格因果卷积，输入 `[B,L]`，输出 `[B,L,V]`。"""

        x = self.embedding(input_ids).transpose(1, 2)
        for conv in self.convs:
            left = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x = F.pad(x, (left, 0))
            x = torch.relu(conv(x))
        return self._to_logits(x.transpose(1, 2))

    def _empty_cache(self, batch_size: int) -> _TCNCache:
        """为每层建立与左侧零填充等价的固定长度历史。"""

        histories = []
        in_channels = self.embedding_dim
        for conv in self.convs:
            left = (conv.kernel_size[0] - 1) * conv.dilation[0]
            histories.append(
                torch.zeros(
                    batch_size,
                    in_channels,
                    left,
                    device=self.device,
                    dtype=self.embedding.weight.dtype,
                )
            )
            in_channels = conv.out_channels
        return _TCNCache(tuple(histories))

    def _step(self, token_ids: torch.Tensor, cache: _TCNCache) -> AutoregressiveState:
        """只计算一个新时间步，并返回更新后的逐层输入缓存。"""

        if token_ids.ndim != 1:
            raise ValueError("TCN token_ids 必须是 [B] 张量")
        if len(cache.layer_inputs) != len(self.convs):
            raise ValueError("TCN cache 层数与卷积层数不一致")
        x = self.embedding(token_ids.to(self.device))
        next_histories: list[torch.Tensor] = []

        # 每层缓存的是该层输入而非输出；拼接新输入后只计算窗口最右侧位置。
        for conv, history in zip(self.convs, cache.layer_inputs, strict=True):
            if history.size(0) != x.size(0) or history.size(1) != x.size(1):
                raise ValueError("TCN cache 的 batch 或通道维度不匹配")
            window = torch.cat([history, x.unsqueeze(-1)], dim=-1)
            left = history.size(-1)
            next_histories.append(window[..., -left:] if left else window[..., :0])
            x = torch.relu(conv(window)).squeeze(-1)

        return AutoregressiveState(
            next_logits=self._to_logits(x),
            cache=_TCNCache(tuple(next_histories)),
        )

    def initial_state(self, batch_size: int) -> AutoregressiveState:
        """消费一批 BOS，并建立固定大小的逐层卷积历史缓存。"""

        bos_ids = torch.full(
            (batch_size,), self.tokenizer.bos_id, dtype=torch.long, device=self.device
        )
        return self._step(bos_ids, self._empty_cache(batch_size))

    def advance_state(
        self,
        state: AutoregressiveState,
        token_ids: torch.Tensor,
    ) -> AutoregressiveState:
        """消费一批新 token，不重算已经缓存的历史时间步。"""

        if not isinstance(state.cache, _TCNCache):
            raise ValueError("TCN state.cache 必须是逐层历史缓存")
        return self._step(token_ids, state.cache)

    def _select_cache(self, cache: object, indices: torch.Tensor) -> _TCNCache:
        """沿第 0 维选择每层卷积历史的 batch。"""

        if not isinstance(cache, _TCNCache):
            raise ValueError("TCN state.cache 必须是逐层历史缓存")
        return _TCNCache(
            tuple(
                history.index_select(0, indices.to(history.device))
                for history in cache.layer_inputs
            )
        )

    def _stack_caches(self, caches: Sequence[object]) -> _TCNCache:
        """逐层合并具有相同窗口形状的卷积历史。"""

        if not all(isinstance(cache, _TCNCache) for cache in caches):
            raise ValueError("TCN cache 必须全部是逐层历史缓存")
        typed_caches = list(caches)
        layer_count = len(typed_caches[0].layer_inputs)
        if any(len(cache.layer_inputs) != layer_count for cache in typed_caches):
            raise ValueError("TCN cache 层数不一致")
        return _TCNCache(
            tuple(
                torch.cat([cache.layer_inputs[layer_index] for cache in typed_caches], dim=0)
                for layer_index in range(layer_count)
            )
        )

    def log_probabilities(self, texts: Sequence[str]) -> torch.Tensor:
        """使用一次因果卷积前向批量计算字符串序列的自然对数概率。"""

        return _teacher_forcing_log_probabilities(self, texts)

    def log_probability(self, text: str) -> torch.Tensor:
        """计算从 BOS 到文本再到 EOS 的自然对数概率。"""

        return self.log_probabilities([text])[0]


@dataclass(frozen=True)
class _TransformerLayerCache:
    """保存一层自注意力已经投影的 Key 和 Value。"""

    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class _TransformerCache:
    """保存所有 Transformer 层的 KV cache 和已消费长度。"""

    layers: tuple[_TransformerLayerCache, ...]
    length: int


class _CausalTransformerBlock(nn.Module):
    """同时支持完整因果前向传播和单步 KV cache 推理的 Transformer block。"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.attention_output = nn.Linear(d_model, d_model)
        self.feedforward_input = nn.Linear(d_model, dim_feedforward)
        self.feedforward_output = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = 0.1

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        """将 `[B,L,D]` 变换为 `[B,H,L,D/H]`。"""

        batch_size, length, _ = values.shape
        return values.reshape(batch_size, length, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, values: torch.Tensor) -> torch.Tensor:
        """将 `[B,H,L,D/H]` 还原为 `[B,L,D]`。"""

        batch_size, _, length, _ = values.shape
        return values.transpose(1, 2).reshape(batch_size, length, self.d_model)

    def _residual_layers(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """执行注意力和前馈网络之后的残差、dropout 与层归一化。"""

        x = self.norm1(x + F.dropout(attention, self.dropout, self.training))
        feedforward = self.feedforward_output(
            F.dropout(
                F.relu(self.feedforward_input(x)),
                self.dropout,
                self.training,
            )
        )
        return self.norm2(x + F.dropout(feedforward, self.dropout, self.training))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对完整序列执行 masked self-attention。"""

        query, key, value = self.qkv_projection(x).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = self.attention_output(self._merge_heads(attention))
        return self._residual_layers(x, attention)

    def step(
        self,
        x: torch.Tensor,
        cache: _TransformerLayerCache,
    ) -> tuple[torch.Tensor, _TransformerLayerCache]:
        """只投影当前 token，并让 Query 读取缓存中的全部历史 Key/Value。"""

        query, key, value = self.qkv_projection(x).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        keys = torch.cat([cache.keys, key], dim=-2)
        values = torch.cat([cache.values, value], dim=-2)
        # 当前 Query 对应序列末端，缓存只含过去和当前位置，因此无需再次使用因果 mask。
        attention = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attention = self.attention_output(self._merge_heads(attention))
        output = self._residual_layers(x, attention)
        return output, _TransformerLayerCache(keys, values)


class AutoregressiveTransformer(AutoregressivePasswordModel):
    """使用权重绑定和逐层 KV cache 的轻量因果 Transformer 密码模型。"""

    model_type = "transformer"

    def __init__(
        self,
        tokenizer: CharTokenizer,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_length: int = 16,
    ):
        r"""初始化 decoder-only 因果 Transformer。
        :param tokenizer: 分词器对象
        :param d_model: token、位置和隐藏状态的共同维度
        :param nhead: 自注意力头数
        :param num_layers: Transformer block 数量
        :param dim_feedforward: block 内前馈网络隐藏维度
        :param max_length: 包含 BOS 在内可输入模型的最大 token 数
        """

        super().__init__()
        if d_model <= 0 or nhead <= 0 or num_layers <= 0 or dim_feedforward <= 0 or max_length <= 0:
            raise ValueError("Transformer 配置必须为正整数")
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        self.tokenizer = tokenizer
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=self.pad_id
        )
        self.embedding.weight.register_hook(self._mask_pad_gradient)
        self.position = nn.Embedding(self.max_length, self.d_model)
        self.layers = nn.ModuleList(
            [
                _CausalTransformerBlock(
                    self.d_model,
                    self.nhead,
                    self.dim_feedforward,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """复用 token embedding 权重，将隐藏状态映射为词表 logits。"""

        return hidden @ self.embedding.weight.T

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行 masked self-attention，输入 `[B,L]`，输出 `[B,L,V]`。"""

        if input_ids.ndim != 2:
            raise ValueError("Transformer input_ids 必须是 [B,L] 张量")
        if input_ids.size(1) > self.max_length:
            raise ValueError("输入长度超过 Transformer max_length")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.embedding(input_ids) + self.position(positions).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self._to_logits(x)

    def _empty_cache(self, batch_size: int) -> _TransformerCache:
        """建立每层形状为 `[B,H,0,D/H]` 的空 KV cache。"""

        shape = (batch_size, self.nhead, 0, self.head_dim)
        layers = tuple(
            _TransformerLayerCache(
                torch.empty(shape, device=self.device, dtype=self.embedding.weight.dtype),
                torch.empty(shape, device=self.device, dtype=self.embedding.weight.dtype),
            )
            for _ in self.layers
        )
        return _TransformerCache(layers=layers, length=0)

    def _step(
        self,
        token_ids: torch.Tensor,
        cache: _TransformerCache,
    ) -> AutoregressiveState:
        """只计算当前位置，并向每层 KV cache 追加一个时间步。"""

        if token_ids.ndim != 1:
            raise ValueError("Transformer token_ids 必须是 [B] 张量")
        if cache.length >= self.max_length:
            raise ValueError("缓存长度超过 Transformer max_length")
        if len(cache.layers) != len(self.layers):
            raise ValueError("Transformer cache 层数与模型层数不一致")
        x = self.embedding(token_ids.to(self.device))
        x = x + self.position.weight[cache.length]
        x = x.unsqueeze(1)
        next_layers: list[_TransformerLayerCache] = []

        # 每层只为当前 token 计算 Q/K/V，旧 token 的 K/V 直接从 cache 读取。
        for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
            if layer_cache.keys.size(0) != x.size(0):
                raise ValueError("Transformer cache 的 batch 维度不匹配")
            x, next_cache = layer.step(x, layer_cache)
            next_layers.append(next_cache)
        next_cache = _TransformerCache(tuple(next_layers), cache.length + 1)
        return AutoregressiveState(self._to_logits(x[:, -1, :]), next_cache)

    def initial_state(self, batch_size: int) -> AutoregressiveState:
        """消费一批 BOS，并初始化每层的 KV cache。"""

        bos_ids = torch.full(
            (batch_size,), self.tokenizer.bos_id, dtype=torch.long, device=self.device
        )
        return self._step(bos_ids, self._empty_cache(batch_size))

    def advance_state(
        self,
        state: AutoregressiveState,
        token_ids: torch.Tensor,
    ) -> AutoregressiveState:
        """消费一批新 token，不重新投影缓存中的历史 Key/Value。"""

        if not isinstance(state.cache, _TransformerCache):
            raise ValueError("Transformer state.cache 必须是逐层 KV cache")
        return self._step(token_ids, state.cache)

    def _select_cache(self, cache: object, indices: torch.Tensor) -> _TransformerCache:
        """沿第 0 维选择每层 KV cache 的 batch。"""

        if not isinstance(cache, _TransformerCache):
            raise ValueError("Transformer state.cache 必须是逐层 KV cache")
        return _TransformerCache(
            layers=tuple(
                _TransformerLayerCache(
                    layer.keys.index_select(0, indices.to(layer.keys.device)),
                    layer.values.index_select(0, indices.to(layer.values.device)),
                )
                for layer in cache.layers
            ),
            length=cache.length,
        )

    def _stack_caches(self, caches: Sequence[object]) -> _TransformerCache:
        """逐层合并长度相同的 KV cache。"""

        if not all(isinstance(cache, _TransformerCache) for cache in caches):
            raise ValueError("Transformer cache 必须全部是逐层 KV cache")
        typed_caches = list(caches)
        length = typed_caches[0].length
        layer_count = len(typed_caches[0].layers)
        if any(cache.length != length for cache in typed_caches):
            raise ValueError("不同序列长度的 Transformer KV cache 不能合并")
        if any(len(cache.layers) != layer_count for cache in typed_caches):
            raise ValueError("Transformer cache 层数不一致")
        return _TransformerCache(
            layers=tuple(
                _TransformerLayerCache(
                    torch.cat([cache.layers[layer_index].keys for cache in typed_caches], dim=0),
                    torch.cat([cache.layers[layer_index].values for cache in typed_caches], dim=0),
                )
                for layer_index in range(layer_count)
            ),
            length=length,
        )

    def log_probabilities(self, texts: Sequence[str]) -> torch.Tensor:
        """使用一次 masked self-attention 前向批量计算字符串序列的自然对数概率。"""

        return _teacher_forcing_log_probabilities(self, texts)

    def log_probability(self, text: str) -> torch.Tensor:
        """计算从 BOS 到文本再到 EOS 的自然对数概率。"""

        return self.log_probabilities([text])[0]


__all__ = [
    "PasswordModel",
    "AutoregressiveState",
    "AutoregressivePasswordModel",
    "AutoregressiveBigram",
    "AutoregressiveMLP",
    "AutoregressiveGRU",
    "AutoregressiveTCN",
    "AutoregressiveTransformer",
]
