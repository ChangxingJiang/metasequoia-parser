"""
根据文法对象 Grammar 计算出所有项目（Item0 对象）的列表，并生成项目之间的后继关系
"""

from typing import List

from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item0, ItemType

__all__ = [
    "cal_all_item0_list"
]


def cal_all_item0_list(grammar: Grammar) -> List[Item0]:
    """根据文法对象 Grammar 计算出所有项目（Item0 对象）的列表，并生成项目之间的后继关系

    Parameters
    ----------
    grammar : Grammar
        存储文法结构的语法类

    Returns
    -------
    List[Item0]
        所有项目的列表
    """
    item_list = []
    for product in grammar.get_product_list():
        if grammar.is_entrance_symbol(product.nonterminal_id):
            # 当前生成式是入口生成式
            last_item_type = ItemType.ACCEPT
            first_item_type = ItemType.INIT
        else:
            last_item_type = ItemType.REDUCE
            first_item_type = ItemType.SHIFT

        # 如果为 %empty，则仅构造一个规约项目
        if len(product.symbol_id_list) == 0:
            item_list.append(Item0(
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(),
                after_handle=tuple(),
                action=product.action,
                item_type=last_item_type,
                successor_symbol=None,  # 规约项目不存在后继项目
                successor_item=None,  # 规约项目不存在后继项目
                sr_priority_idx=product.sr_priority_idx,
                sr_combine_type=product.sr_combine_type,
                rr_priority_idx=product.rr_priority_idx
            ))
            continue

        # 添加句柄在结束位置（最右侧）的项目（规约项目）
        last_item = Item0(
            nonterminal_id=product.nonterminal_id,
            before_handle=tuple(product.symbol_id_list),
            after_handle=tuple(),
            action=product.action,
            item_type=last_item_type,
            successor_symbol=None,  # 规约项目不存在后继项目
            successor_item=None,  # 规约项目不存在后继项目
            sr_priority_idx=product.sr_priority_idx,
            sr_combine_type=product.sr_combine_type,
            rr_priority_idx=product.rr_priority_idx
        )
        item_list.append(last_item)

        # 从右向左依次添加句柄在中间位置（不是最左侧和最右侧）的项目（移进项目），并将上一个项目作为下一个项目的后继项目
        for i in range(len(product.symbol_id_list) - 1, 0, -1):
            now_item = Item0(
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(product.symbol_id_list[:i]),
                after_handle=tuple(product.symbol_id_list[i:]),
                action=product.action,
                item_type=ItemType.SHIFT,
                successor_symbol=product.symbol_id_list[i],
                successor_item=last_item,
                sr_priority_idx=product.sr_priority_idx,
                sr_combine_type=product.sr_combine_type,
                rr_priority_idx=product.rr_priority_idx
            )
            item_list.append(now_item)
            last_item = now_item

        # 添加添加句柄在开始位置（最左侧）的项目（移进项目或入口项目）
        item_list.append(Item0(
            nonterminal_id=product.nonterminal_id,
            before_handle=tuple(),
            after_handle=tuple(product.symbol_id_list),
            action=product.action,
            item_type=first_item_type,
            successor_symbol=product.symbol_id_list[0],
            successor_item=last_item,
            sr_priority_idx=product.sr_priority_idx,
            sr_combine_type=product.sr_combine_type,
            rr_priority_idx=product.rr_priority_idx
        ))

    return item_list
