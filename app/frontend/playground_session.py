"""Playground 会话内的有限结果快照；不写磁盘、不使用跨用户缓存。"""

from collections.abc import MutableMapping, Sequence


RESULTS_KEY = "playground_results"


def remember_result(state: MutableMapping, slot: str, model_ids: Sequence[str], payload) -> None:
    """每个实验只保留最后一次成功结果，并绑定当时的模型集合及顺序。"""

    state.setdefault(RESULTS_KEY, {})[slot] = (tuple(model_ids), payload)


def recall_result(state: MutableMapping, slot: str, model_ids: Sequence[str]):
    """恢复当前集合的结果；模型选择改变后丢弃旧快照，避免混用标签和分数。"""

    results = state.get(RESULTS_KEY, {})
    saved = results.get(slot)
    if saved is None:
        return None
    if saved[0] != tuple(model_ids):
        results.pop(slot, None)
        return None
    return saved[1]


def clear_playground_results(state: MutableMapping) -> None:
    """清除实验结果及演示输入，只作用于当前会话。"""

    state.pop(RESULTS_KEY, None)
    for key in ("score_demo_password", "beam_demo_prefix", "character_lab_input",
                "character_lab_input_text"):
        state.pop(key, None)
