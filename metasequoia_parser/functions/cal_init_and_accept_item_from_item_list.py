"""
从项目列表中获取入口项目（INIT）、接受项目（ACCEPT）
"""

from typing import List, NewType

from metasequoia_parser.common import ItemBase
from metasequoia_parser.exceptions import ParserError

__all__ = [
    "cal_init_item_from_item_list",
    "cal_accept_item_from_item_list"
]

T = NewType(name="T", tp=ItemBase)


def cal_init_item_from_item_list(item_list: List[T]) -> T:
    """从项目列表中获取入口项目

    Parameters
    ----------
    item_list : List[T]
        所有项目的列表（泛型 T 为 ItemBase 的子类）

    Returns
    -------
    T
        入口项目
    """
    for item in item_list:
        if item.is_init():
            return item
    raise ParserError("未从项目列表中获取到 INIT 项目")


def cal_accept_item_from_item_list(item_list: List[T]) -> T:
    """从项目列表中获取接受项目

    Parameters
    ----------
    item_list : List[T]
        所有项目的列表

    Returns
    -------
    T
        接受项目
    """
    for item in item_list:
        if item.is_accept():
            return item
    raise ParserError("未从项目列表中获取到 ACCEPT 项目")
