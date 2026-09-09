import json
from pathlib import Path

class CharTokenizer:
    r"""
    在文本和 token id 之间转换，使用 `from_text` 从文本新建，`from_json` 从 JSON 文件加载
    """
    PAD_TOKEN = 'PAD'
    BOS_TOKEN = 'BOS'
    EOS_TOKEN = 'EOS'
    UNK_TOKEN = 'UNK'
    SPECIAL_TOKENS = (
        PAD_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        UNK_TOKEN
    )
    def __init__(self, id_to_token: list[str]):
        """
        :param id_to_token: id 到 token 的映射表
        :ivar self.token_to_id: token 到 id 的映射
        :ivar self.id_to_token: id 到 token 的映射
        """
        # mapping tables
        self.id_to_token = tuple(id_to_token)
        if self.id_to_token[:len(self.SPECIAL_TOKENS)] != self.SPECIAL_TOKENS:
            raise ValueError("词表必须以 PAD、BOS、EOS、UNK 按固定顺序开始")
        if any(not isinstance(token, str) or len(token) != 1
               for token in self.id_to_token[len(self.SPECIAL_TOKENS):]):
            raise ValueError("普通 token 必须是单个字符")
        self.token_to_id: dict[str, int] = {
            token: idx
            for idx, token in enumerate(self.id_to_token)
        }
        if len(self.token_to_id) != len(self.id_to_token):
            raise ValueError("词表不能包含重复 token")
        self._special_tokens_border = len(self.SPECIAL_TOKENS) # special if lower than the borderer

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CharTokenizer) and self.id_to_token == other.id_to_token

    # save and create
    @classmethod
    def from_text(cls, text: list[str]):
        r"""根据训练文本新建 tokenizer
        :param text: 训练文本
        """
        # extract uniq characters from text
        characters = sorted({
            char
            for sentence in text
            for char in sentence
        })
        # cat characters with special tokens
        id_to_token = list(cls.SPECIAL_TOKENS) + characters
        return cls(id_to_token)

    @classmethod
    def from_json(cls, path: Path):
        r"""从 JSON 文件加载 tokenizer
        :param path: 文件路径
        """
        with path.open('r', encoding='utf-8') as stream:
            id_to_token = json.load(stream)
        return cls(id_to_token)

    def dump(self, path: Path):
        r"""保存为 JSON 文件
        :param path: 保存路径
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            json.dump(self.id_to_token, stream)

    # 对齐 pytorch 的 state_dict 接口
    def state_dict(self) -> dict:
        r"""返回 tokenizer 的状态字典
        :return: 状态字典
        """
        return {
            'id_to_token': self.id_to_token
        }

    def load_state_dict(self, state_dict: dict):
        r"""加载 tokenizer 的状态字典
        :param state_dict: 状态字典
        """
        self.__init__(state_dict['id_to_token'])

    # util methods
    @property
    def vocab_size(self) -> int:
        r"""词汇表大小"""
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id['PAD']

    @property
    def eos_id(self) -> int:
        return self.token_to_id['EOS']

    @property
    def unk_id(self) -> int:
        return self.token_to_id['UNK']

    @property
    def bos_id(self) -> int:
        return self.token_to_id['BOS']

    def is_special_token(self, idx: int) -> bool:
        r"""判断 id 是否为特殊token
        :param idx: token id
        :return: 是否为特殊 token
        """
        return 0 <= idx < self._special_tokens_border

    # main function encode and decode
    def encode(self, text: str, add_bos: bool=False, add_eos: bool=False) -> list[int]:
        """
        文本 -> token id
        :param text: 文本内容
        :param add_bos: 是否在文本开头添加 BOS token
        :param add_eos: 是否在文本结尾添加 EOS token
        :return: token id 列表
        """
        tokens = []
        # add BOS
        if add_bos:
            tokens.append(self.token_to_id['BOS'])
        # convert chars to ids
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id['UNK'])
        # add EOS
        if add_eos:
            tokens.append(self.token_to_id['EOS'])
        return tokens

    def decode(self, ids: list[int], stop_at_eos: bool=True, skip_special_tokens: bool=True) -> str:
        r"""token id -> 文本
        :param ids: token id 列表
        :param stop_at_eos: 是否在遇到 EOS 时停止解码
        :param skip_special_tokens: 是否跳过特殊 token（BOS, EOS, UNK, PAD）
        :return: 解码后的文本
        """
        chars = []
        # convert ids to chars
        for idx in ids:
            # stop at EOS
            if stop_at_eos and idx == self.token_to_id['EOS']:
                break
            # skip special tokens
            if skip_special_tokens and self.is_special_token(idx):
                continue
            # convert legal id
            if  0 <= idx < self.vocab_size:
                chars.append(self.id_to_token[idx])
        return ''.join(chars)

