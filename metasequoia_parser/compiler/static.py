"""
静态名称
"""

FN_PREFIX_SHIFT = "h"  # 移进函数名称前缀
FN_PREFIX_REDUCE = "r"  # 规约函数名称前缀
FN_NAME_ACCEPT = "p"  # 接受函数名称
FN_NAME_ERROR = "e"  # 错误函数名称（实际不创建）

PARAM_STATUS_STACK = "a"  # 状态栈参数名称
PARAM_SYMBOL_STACK = "b"  # 对象栈参数名称
PARAM_TERMINAL = "c"  # 终结符参数名称
VAR_MOVE_ACTION = "m"  # move_action 变量名称

VAR_PREFIX_SYMBOL_SET = "E"  # 符号 ID 集合的变量名前缀

TERMINAL_SET_NAME = "R"  # 状态函数：终结符 ID 集合

TERMINAL_SYMBOL_ID_NAME = "i"  # 终结符 ID 值的参数名称（在 metasequoia_parser/common/symbol.py 中定义）
TERMINAL_SYMBOL_VALUE_NAME = "v"  # 终结符实际值值的参数名称（在 metasequoia_parser/common/symbol.py 中定义）

OPT_SPACE = ""  # 可选是否添加的空格（如为空字符串则不添加额外的空格）
