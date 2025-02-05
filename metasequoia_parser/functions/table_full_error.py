"""
将 ACTION 表和 GOTO 表中所有没有填充的位置全部置为 ERROR 行为（原地更新）
"""

__all__ = [
    "table_full_error",
]

from typing import Callable, List

from metasequoia_parser.common import ActionError


def table_full_error(table: List[List[Callable]]) -> None:
    """将 ACTION 表和 GOTO 表中所有没有填充的位置全部置为 ERROR 行为（原地更新）

    Parameters
    ----------
    table : List[List[Callable]]
        ACTION 表 + GOTO 表
    """
    n_status = len(table)
    assert n_status >= 1, "ACTION 表和 GOTO 表中没有状态"

    n_symbol = len(table[0])

    # 将没有填充的位置全部填充为 Error 行为
    for i in range(n_status):
        for j in range(n_symbol):
            if table[i][j] is None:
                table[i][j] = ActionError()
