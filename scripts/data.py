import torch
from .tokenizer import CharTokenizer
from pathlib import Path
import hashlib
import heapq
import random

class PasswordDataset(torch.utils.data.Dataset):
    def __init__(self, passwords: list[str], tokenizer: CharTokenizer):
        self.tokenizer = tokenizer
        self.passwords = passwords

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, index: int):
        # get id list
        password = self.passwords[index]
        raw_ids = self.tokenizer.encode(password)
        # construct input and target ids, eg:
            # input = [BOS, 1, 2, 3]
            # target = [1, 2, 3, EOS]
        input_ids = [self.tokenizer.bos_id] + raw_ids
        target_ids = raw_ids + [self.tokenizer.eos_id]
        # convert to long type [L] tensor
        input_tensor = torch.LongTensor(input_ids)
        target_tensor = torch.LongTensor(target_ids)
        return input_tensor, target_tensor

def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]],
                  pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""将 input, target 的 tensor 列表转换为维度 [B, L] 的 batch tensor
    :param batch: 包含多个 (input_tensor, target_tensor) 的列表
    :param pad_id: 用于填充的 token id
    :return: 返回填充后的 input 和 target 的 batch tensor
    """
    # separate inputs and targets
    input_batch = []
    target_batch = []
    for input_tensor, target_tensor in batch:
        input_batch.append(input_tensor)
        target_batch.append(target_tensor)
    # pad the two lists of [L] tensors into [B, L] tensors
    input_batch_padded = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=pad_id)
    target_batch_padded = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=pad_id)
    return input_batch_padded, target_batch_padded

def normalize_password(
    raw_line: str,
    min_length: int = 1,
    max_length: int = 12,
) -> str | None:
    """
    对密码进行规范化处理，去除不符合要求的密码。
    :param raw_line: 原始密码字符串
    :param min_length: 密码最小长度
    :param max_length: 密码最大长度
    :return: 规范化后的密码字符串，如果不符合要求则返回 None
    """
    # 删除分隔符
    line = raw_line.rstrip('\n\r')
    # 检查长度
    length_legal = min_length <= len(line) <= max_length
    # 只接受可打印 ASCII
    ascii_legal = all(32 <= ord(c) <= 126 for c in line)
    return line if length_legal and ascii_legal else None

def count_lines(file_path: Path) -> int:
    """
    统计文件中的行数。
    :param file_path: 文件路径
    :return: 文件中的行数
    """
    with open(file_path, "r", encoding="utf-8", errors="surrogateescape") as f:
        for i, _ in enumerate(f, 1):
            pass
    return i

def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    min_length: int = 1,
    max_length: int = 12
) -> dict[str, int]:
    """
    将原始密码数据集划分为训练集、验证集和测试集，并保存到指定目录中。
    :param source_path: 原始密码数据集文件路径
    :param output_dir: 输出目录路径
    :param train_size: 训练集大小
    :param val_size: 验证集大小
    :param test_size: 测试集大小
    :param seed: 随机种子
    :param min_length: 密码最小长度
    :param max_length: 密码最大长度
    :return: 包含统计信息的字典
    """
    # 流式读取
    ret = dict(
        total_lines=0,
        valid_lines=0,
        unique_lines=0,
        selected_lines=0,
        train_size=0,
        val_size=0,
        test_size=0
    )
    passwords = []
    # 只保留摘要值最小的 sample_size 个不同密码
    sample_size = train_size + val_size + test_size
    if train_size <= 0 or val_size < 0 or test_size < 0:
        raise ValueError("训练集大小必须 > 0，验证集和测试集大小必须 >= 0。")
    with  open(source_path, "r",
               encoding="utf-8",
               errors="surrogateescape") as f:
        unique_passwords = set()
        for line in f:
            ret['total_lines'] += 1
            # 规范化密码
            normalized = normalize_password(line, min_length=min_length, max_length=max_length)
            if normalized is None:
                continue
            ret['valid_lines'] += 1
            # 去重
            if normalized in unique_passwords:
                continue
            ret['unique_lines'] += 1
            # 哈希摘要
            hash_value = hashlib.blake2b(f'{seed}\0{normalized}'.encode('ascii'),
                                         digest_size=16).digest()
            hash_value = int.from_bytes(hash_value, 'big')
            # 堆未满直接推入
            if len(passwords) < sample_size:
                heapq.heappush(passwords, (-hash_value, normalized))
                unique_passwords.add(normalized)
                ret['selected_lines'] += 1
            # 堆满 且 新摘要 < threshold: 替换堆顶
            elif hash_value < -passwords[0][0]:
                _, removed = heapq.heapreplace(passwords, (-hash_value, normalized))
                unique_passwords.remove(removed)
                unique_passwords.add(normalized)
            # 堆满 且 新摘要 >= threshold: 忽略

    if len(passwords) < sample_size:
        raise ValueError(f"需要至少 {sample_size} 个不同密码，实际只有 {len(passwords)} 个。")
    passwords = [p for _, p in sorted(passwords, key=lambda x: (-x[0], x[1]))]
    # 打乱顺序并划分数据集
    rng = random.Random(seed)
    rng.shuffle(passwords)
    train = passwords[:train_size]
    val = passwords[train_size:train_size + val_size]
    test = passwords[train_size + val_size:train_size + val_size + test_size]
    # 记录
    ret['train_size'] = len(train)
    ret['val_size'] = len(val)
    ret['test_size'] = len(test)
    # 保存数据集
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train))
    with open(output_dir / 'val.txt', 'w', encoding='utf-8') as w:
        w.write('\n'.join(val))
    with open(output_dir / 'test.txt', 'w', encoding='utf-8') as t:
        t.write('\n'.join(test))
    return ret

def read_dataset(source_dir: Path,
                 train_limit: int | None = None,
                 val_limit: int | None = None,
                 test_limit: int | None = None) -> dict[str, list[str]]:
    """
    读取数据集文件并返回包含训练集、验证集和测试集的字典。
    :param source_dir: 数据集目录路径
    :param train_limit: 训练集限制数量（用于快速测试）
    :param val_limit: 验证集限制数量（用于快速测试）
    :param test_limit: 测试集限制数量（用于快速测试）
    :return: 包含训练集、验证集和测试集的字典
    """
    dataset = {}
    for split, limit in [('train', train_limit), ('val', val_limit), ('test', test_limit)]:
        file_path = source_dir / f'{split}.txt'
        if not file_path.exists():
            raise FileNotFoundError(f"数据集文件 {file_path} 不存在。")
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset[split] = [line.rstrip('\n\r') for line in f]
        if limit is not None:
            dataset[split] = dataset[split][:limit]
    return dataset