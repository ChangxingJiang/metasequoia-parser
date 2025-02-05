"""
创建 Item0Set 对象之间的关联关系（原地更新）
"""

from typing import Dict

from metasequoia_parser.common import Item0
from metasequoia_parser.common import Item0Set
from metasequoia_parser.exceptions import LR0GrammarError

__all__ = [
    "build_relation_between_item0_set"
]


def build_relation_between_item0_set(core_to_item0_set_hash: Dict[Item0, Item0Set]) -> None:
    """创建 Item0Set 对象之间的关联关系（原地更新）

    Parameters
    ----------
    core_to_item0_set_hash : Dict[Item0, Item0Set]
        核心项目到项目集闭包的映射（项目集闭包中包含项目列表，但不包含项目闭包之间关联关系）
    """
    for item0, item0_set in core_to_item0_set_hash.items():
        successor_visited = set()
        for sub_item_0 in item0_set.all_item_list:
            successor_symbol = sub_item_0.successor_symbol

            if successor_symbol is None:
                continue  # 如果子项目没有后继项目，则不需要继续处理

            if successor_symbol in successor_visited:
                raise LR0GrammarError(f"{item0} 中的 {successor_symbol} 存在 LR(0) 文法冲突")
            successor_visited.add(successor_symbol)

            # 计算子项目有后继项目，则为当前项目集闭包增加到后继项目所在项目集闭包的关系
            item0_set.set_successor(successor_symbol, core_to_item0_set_hash[sub_item_0.successor_item])
