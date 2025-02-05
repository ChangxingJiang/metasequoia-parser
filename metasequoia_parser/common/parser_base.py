"""
解析器的抽象基类
"""

import abc
from typing import Callable, List, Tuple

from metasequoia_parser.common.grammar import Grammar
from metasequoia_parser.lexical.lexical_iterator import LexicalBase
from metasequoia_parser.common.parsing_context import ParsingContext
from metasequoia_parser.exceptions import ParseError

__all__ = [
    "ParserBase",  # 基于 ACTION TABLE 和 GOTO TABLE 的解析器的抽象基类
]


class ParserBase(abc.ABC):
    """基于 ACTION TABLE 和 GOTO TABLE 的解析器的抽象基类"""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar

        # 计算 ACTION TABLE 和 GROUP TABLE 以及初始状态
        self.table, self.entrance_status = self.create_action_table_and_goto_table()

        # print("---------- ACTION + GOTO ----------")
        # for i, row in enumerate(self.table):
        #     print(f"【状态:{i}】", ", ".join([f"{self.grammar.get_symbol_name(symbol_id)}:{action}"
        #                                     for symbol_id, action in enumerate(row) if repr(action) != "<Error>"]))
        # print(f"状态数量: {len(self.table)}")

    @abc.abstractmethod
    def create_action_table_and_goto_table(self) -> Tuple[List[List[Callable]], int]:
        """计算 ACTION TABLE 和 GROUP TABLE 以及初始状态

        Returns
        -------
        action_table : List[List[Callable]]
            ACTION 表 + GOTO 表
        int
            初始状态
        """

    def parse(self, lexical_iterator: LexicalBase):
        """解析逻辑"""
        # 初始化状态管理器
        context: ParsingContext = ParsingContext()

        # 初始化状态管理器：添加 -1 状态和 Start 元素
        context.push_status(self.entrance_status)

        # print("---------- Parsing ----------")

        terminal = lexical_iterator.lex()
        # print("初始更新", terminal)

        while True:
            # 读取当前终结符
            terminal_id = terminal.symbol_id

            # 读取当前状态
            last_status = context.top_status()

            print(f"当前终结符={terminal_id}({self.grammar.get_symbol_name(terminal_id)}) "
                  f"{context.print(grammar=self.grammar)}")
            # print(self.table[last_status][terminal_id])

            # 反查 ACTION 表获取当前行为
            action = self.table[last_status][terminal_id]

            # 执行行为
            action_res = action(context, terminal)

            if action_res == 1:  # 移进行为
                terminal = lexical_iterator.lex()
                # print("移进更新", terminal)
            elif action_res == 2:  # 规约行为
                # 读取规约行为生成的非终结符
                non_terminal_id = context.top_symbol().symbol_id

                # 读取当前状态
                last_status = context.top_status()

                # 反查 GOTO 表获取当前行为
                goto_action = self.table[last_status][non_terminal_id]
                # print("GOTO:", last_status, self.grammar.n_terminal, non_terminal_id)

                # 执行 GOTO 行为
                goto_action_res = goto_action(context, terminal)

                if goto_action_res != 3:
                    raise ParseError(f"语法解析失败: goto_action_res = {goto_action_res}")
                continue
            elif action_res == 3:  # GOTO 行为
                raise ParseError("ACTION 表中出现 GOTO 行为")
            elif action_res == 4:  # ACCEPT 行为
                return context.top_symbol().symbol_value
            else:
                raise ParseError(f"语法解析失败: action_res = {action_res}")
