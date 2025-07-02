"""
语法解析器自动编译逻辑

因为需要从规约函数中解析出源代码，所以支持的 action 写法有限，具体包括如下形式：

GRule.create(symbols=["a", "B", "c"], action=lambda x: f"{x[0]}{x[1]}{x[2]}")

GRule.create(["a", "B", "c"], lambda x: f"{x[0]}{x[1]}{x[2]}")

GRule(symbols=("b",), action=lambda x: f"{x[0]}")

def test(x):
    return f"{x[0]}{x[1]}{x[2]}"
GRule.create(symbols=["a", "B", "c"], action=test)

test = lambda x: f"{x[0]}{x[1]}{x[2]}"
GRule.create(symbols=["a", "B", "c"], action=test)
"""

import collections
from typing import Callable, Dict, Iterable, List, TextIO

from metasequoia_parser.common import ActionAccept, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import TerminalType
from metasequoia_parser.common.grammar import GGroup, GRule, GrammarBuilder
from metasequoia_parser.compiler.common import dec_to_base36
from metasequoia_parser.compiler.compile_reduce_function import CompileError, compile_reduce_function
from metasequoia_parser.compiler.static import *  # pylint: disable=W0401,W0614
from metasequoia_parser.parser.lalr1 import ParserLALR1
from metasequoia_parser.utils import LOGGER

# ---------------------------------------- 【名称构造器】移进函数名称 ----------------------------------------

ACTION_SHIFT_FUNCTION_CODE_HASH = {}


def create_action_shift_function_name(status: int) -> str:
    """创建移进函数名称

    Parameters
    ----------
    status: int
        移进函数需要压入栈中的状态

    Returns
    -------
    str
        移进函数名称
    """
    if status not in ACTION_SHIFT_FUNCTION_CODE_HASH:
        code = dec_to_base36(len(ACTION_SHIFT_FUNCTION_CODE_HASH))
        ACTION_SHIFT_FUNCTION_CODE_HASH[status] = f"{FN_PREFIX_SHIFT}{code}"
    return ACTION_SHIFT_FUNCTION_CODE_HASH[status]


# ---------------------------------------- 【名称构造器】规约函数名称 ----------------------------------------

ACTION_REDUCE_CODE_HASH = {}


def create_action_reduce_function_name(symbol: int, function: Callable) -> str:
    """创建规约函数名称

    Parameters
    ----------
    symbol: int
        规约生成的非终结符 ID
    function: Callable
        规约函数可调用对象

    Returns
    -------
    str
        规约函数名称
    """
    if (symbol, function) not in ACTION_REDUCE_CODE_HASH:
        code = dec_to_base36(len(ACTION_REDUCE_CODE_HASH))
        ACTION_REDUCE_CODE_HASH[(symbol, function)] = f"{FN_PREFIX_REDUCE}{code}"
    return ACTION_REDUCE_CODE_HASH[(symbol, function)]


# ---------------------------------------- 【名称构造器】通用入口 ----------------------------------------


def create_function_name(table: List[List[Callable]], status: int, symbol: int) -> str:
    """创建函数名称

    Parameters
    ----------
    table: List[List[Callable]]
        状态转移表
    status : int
        状态 ID
    symbol : int
        符号 ID

    Returns
    -------
    str
        函数名称
    """
    action = table[status][symbol]
    if isinstance(action, ActionShift):
        return create_action_shift_function_name(action.status)
    if isinstance(action, ActionReduce):
        return create_action_reduce_function_name(action.reduce_name, action.reduce_function)
    if isinstance(action, ActionAccept):
        return FN_NAME_ACCEPT
    return FN_NAME_ERROR


# ---------------------------------------- 【名称构造器】状态处理函数 ----------------------------------------


def create_state_function_name(status: int) -> str:
    """创建状态处理函数名称

    Parameters
    ----------
    status: int
        状态 ID

    Returns
    -------
    str
        状态处理函数名称
    """
    return f"{FN_PREFIX_STATUS}{dec_to_base36(status)}"


# ---------------------------------------- 【代码生成】移进行为函数 ----------------------------------------


BUILT_SHIFT_ACTION_SET = set()


def write_action_shift(f: TextIO, status_id_to_code_hash: Dict[int, int], action: ActionShift) -> None:
    """写出移进行为函数

    Parameters
    ----------
    f: TextIO
        输出文件对象
    status_id_to_code_hash : Dict[int, int]
        状态 ID 到代码的映射
    action: ActionShift
        移进动作
    """
    if action.status in BUILT_SHIFT_ACTION_SET:
        return

    BUILT_SHIFT_ACTION_SET.add(action.status)
    function_name = create_action_shift_function_name(action.status)
    status_id = status_id_to_code_hash[action.status]
    f.write(f"""
def {function_name}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL}):
    {PARAM_STATUS_STACK}.append({action.status})
    {PARAM_SYMBOL_STACK}.append({PARAM_TERMINAL}.{TERMINAL_SYMBOL_VALUE_NAME})
    return s{status_id},True
""")


# ---------------------------------------- 【代码生成】符号 ID 的集合 ----------------------------------------


GLOBAL_VARIABLE_INT_SET_HASH = {}


def write_int_set(f: TextIO, int_set: Iterable[int]) -> str:
    """【代码生成】定义符号 ID 或状态 ID 的集合，并返回构造的 Python 集合的变量名

    输入：{1, 2, 3}
    构造：E1={1,2,3}
    返回：E1

    Parameters
    ----------
    f: TextIO
        输出文件对象
    int_set: Set[int]
        符号 ID 或状态 ID 的集合

    Returns
    -------
    str
        符号集合的变量名
    """
    unique_code = tuple(sorted(int_set))
    if unique_code not in GLOBAL_VARIABLE_INT_SET_HASH:
        code = dec_to_base36(len(GLOBAL_VARIABLE_INT_SET_HASH))  # 计算变量序号
        variable_name = f"{VAR_PREFIX_SYMBOL_SET}{code}"  # 构造变量名
        variable_value = ",".join([str(symbol_id) for symbol_id in unique_code])  # 构造变量值
        f.write(f"""
{variable_name}={{{variable_value}}}
""")  # 写出变量名的定义
        GLOBAL_VARIABLE_INT_SET_HASH[unique_code] = variable_name
    return GLOBAL_VARIABLE_INT_SET_HASH[unique_code]


class Lalr1Compiler:
    """LALR(1) 编译器"""

    def __init__(self, f: TextIO, parser: ParserLALR1, import_list: List[str]):
        """初始化

        Parameters
        ----------
        f : TextIO
            输出文件对象
        parser: ParserLALR1
            LALR(1) 解析器
        import_list: List[str]
            导入的模块列表
        """
        self.f = f
        self.parser = parser
        self.import_list = import_list

        self.table = parser.table
        self.n_status = len(self.table)

        # 初始化 GOTO 映射：goto_hash[非中介符][状态]=新状态
        self.goto_hash = collections.defaultdict(dict)
        for i in range(self.n_status):
            for j in range(parser.grammar.n_terminal, parser.grammar.n_symbol):
                action = self.table[i][j]
                if isinstance(action, ActionGoto):
                    self.goto_hash[j][i] = action.status

        # ------------------------------ 【构造】合并 ACTION 相同的状态函数 ------------------------------
        # 需要考虑是否存在 ACTION 表完全相同的状态：样例 https://blog.51cto.com/u_15279775/5130206
        visited_status_hash = {}
        self.status_id_to_code_hash = {}
        for i in range(self.n_status):
            status_core_list = []
            for j in range(parser.grammar.n_terminal):
                function_name = create_function_name(self.table, i, j)
                status_core_list.append(function_name)
            status_tuple = tuple(status_core_list)
            if status_tuple not in visited_status_hash:
                visited_status_hash[status_tuple] = i
            self.status_id_to_code_hash[i] = visited_status_hash[status_tuple]

        # 初始化
        self._cache_write_reduce_status_hash = {}

    def write_reduce_function_mode_1(self, action: ActionReduce):
        """【代码生成】写出第 1 种格式的 Reduce 函数：无论之前状态如何，都转移到相同的状态

        Parameters
        ----------
        action: ActionReduce
            规约行为
        """
        symbol = action.reduce_name
        function = action.reduce_function
        n_param = action.n_param

        # 计算将转移到的新状态的状态处理函数名
        new_status_id = list(set(self.goto_hash[symbol].values()))[0]
        new_status_code = self.status_id_to_code_hash[new_status_id]

        # 计算规约函数名
        function_name = create_action_reduce_function_name(symbol, function)

        # 构造状态验证集
        set_var_name = write_int_set(self.f, self.goto_hash[symbol])

        # 解析规约函数，重新构造 Python 代码
        function_row_list = compile_reduce_function(function, n_param)
        function_row_list_source = "\n    ".join(function_row_list)

        # 计算符号栈和状态栈的处理语句
        if n_param > 0:
            process_symbol_stack = f"{PARAM_SYMBOL_STACK}[-{n_param}:]=[v]"
            process_status_stack = f"{PARAM_STATUS_STACK}[-{n_param}:]=[{new_status_id}]"
        else:
            process_symbol_stack = f"{PARAM_SYMBOL_STACK}.append(v)"
            process_status_stack = f"{PARAM_STATUS_STACK}.append({new_status_id})"

        # 生成规约函数代码
        self.f.write(f"""
def {function_name}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},_):
    {function_row_list_source}
    assert {PARAM_STATUS_STACK}[-{n_param + 1}] in {set_var_name}
    {process_symbol_stack}
    {process_status_stack}
    return s{new_status_code},False
""")

    def write_reduce_status_hash(self, status_hash: Dict[int, int]):
        """【代码生成】构造规约函数中旧状态到新状态的映射表（Python 字典），并返回 Python 字典的变量名"""
        unique_code = tuple(sorted(status_hash.items()))
        if unique_code not in self._cache_write_reduce_status_hash:
            variable_name_code = dec_to_base36(len(self._cache_write_reduce_status_hash))  # 计算变量序号
            self._cache_write_reduce_status_hash[unique_code] = f"{VAR_PREFIX_REDUCE_STATUS_HASH}{variable_name_code}"
        return self._cache_write_reduce_status_hash[unique_code]

    def final_write_reduce_status_hash(self):
        for unique_code, var_name in self._cache_write_reduce_status_hash.items():
            n_item = len(unique_code)  # 哈希表中的元素数量

            # 统计转移到每种状态的符号数量
            grouped_item_hash = collections.defaultdict(list)
            for key, value in unique_code:
                grouped_item_hash[value].append(key)

            # 计算需要通过集合添加的符号：绝对值超过 3 且大于等于 50%
            big_item_set = {value for value, key_list in grouped_item_hash.items()
                            if len(key_list) >= 3 and len(key_list) * 2 >= n_item}

            self.f.write(f"{var_name}={{")
            for old_status_id, new_status_id in unique_code:
                if new_status_id not in big_item_set:
                    new_status_code = self.status_id_to_code_hash[new_status_id]
                    self.f.write(f"{old_status_id}:({new_status_id},s{new_status_code}),")
            self.f.write("}\n")

            # 补充通过集合添加的符号
            for new_status_id in big_item_set:
                new_status_code = self.status_id_to_code_hash[new_status_id]
                symbol_set_var_name = write_int_set(self.f, grouped_item_hash[new_status_id])
                self.f.write(f"{var_name}.update({{v:({new_status_id},s{new_status_code}) "
                             f"for v in {symbol_set_var_name}}})\n")

    def write_reduce_function_mode_2(self, action: ActionReduce):
        """写出第 2 种格式的 Reduce 函数：针对之前状态不同，转移到不同的状态

        Parameters
        ----------
        action: ActionReduce
            规约行为
        """
        symbol = action.reduce_name
        function = action.reduce_function
        n_param = action.n_param

        # 计算规约函数名
        function_name = create_action_reduce_function_name(symbol, function)

        # 解析规约函数，重新构造 Python 代码
        function_row_list = compile_reduce_function(function, n_param)
        function_row_list_source = "\n    ".join(function_row_list)

        # 构造映射信息
        hash_var_name = self.write_reduce_status_hash(self.goto_hash[symbol])

        self.f.write(
            f"def {function_name}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},_):\n"
        )
        self.f.write(f"    {function_row_list_source}\n")
        self.f.write(
            f"    n,k={hash_var_name}[{PARAM_STATUS_STACK}[-{n_param + 1}]]\n"
        )
        if n_param > 0:
            self.f.writelines([
                f"    {PARAM_SYMBOL_STACK}[-{n_param}:]=[v]\n",
                f"    {PARAM_STATUS_STACK}[-{n_param}:]=[n]\n",
            ])
        else:
            self.f.writelines([
                f"    {PARAM_SYMBOL_STACK}.append(v)\n",
                f"    {PARAM_STATUS_STACK}.append(n)\n",
            ])

        self.f.writelines([
            f"    return k,False\n",
            "\n",
            "\n"
        ])

    def write(self) -> None:
        """编译 LALR(1) 解析器"""
        f = self.f
        parser = self.parser
        import_list = self.import_list
        n_status = self.n_status
        table = self.table

        LOGGER.info("[Write] START")

        f.write("\"\"\"\n"
                "Auto generated by Metasequoia Parser\n"
                "\"\"\"\n"
                "\n"
                "import metasequoia_parser as ms_parser\n"
                "\n"
                )

        # 写入引用信息
        for import_line in import_list:
            f.write(f"{import_line}\n")

        f.write("\n\n")

        # 如果 ACTION + GOTO 表为空，则抛出异常
        if len(table) == 0 or len(table[0]) == 0:
            raise CompileError("ACTION + GOTO 表为空")

        # ------------------------------ 【构造】移进行为函数 ------------------------------
        for i in range(n_status):
            for j in range(parser.grammar.n_terminal):
                action = table[i][j]
                if isinstance(action, ActionShift):
                    write_action_shift(f, self.status_id_to_code_hash, action)

        # ------------------------------ 【构造】规约行为函数 ------------------------------
        reduce_function_hash = set()
        for i in range(n_status):
            for j in range(parser.grammar.n_terminal):
                action = table[i][j]
                if isinstance(action, ActionReduce):
                    nonterminal_id = action.reduce_name
                    reduce_function = action.reduce_function

                    # 如果当前非终结符的相同规约逻辑已处理，则不需要重复添加
                    if (nonterminal_id, reduce_function) in reduce_function_hash:
                        continue
                    reduce_function_hash.add((nonterminal_id, reduce_function))

                    # 生成规约行为函数的名称
                    if len(set(self.goto_hash[nonterminal_id].values())) == 1:
                        self.write_reduce_function_mode_1(action)
                    else:
                        self.write_reduce_function_mode_2(action)

        # ------------------------------ 【构造】接收行为函数 ------------------------------
        # pylint: disable=C0301
        f.writelines([
            f"def {FN_NAME_ACCEPT}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL}):\n",
            f"    return None,True\n",
            "\n",
            "\n"
        ])

        # ------------------------------ 【构造】终结符 > 行为函数的字典；状态函数 ------------------------------
        for i in range(n_status):
            # 构造：终结符 > 行为函数的字典
            if self.status_id_to_code_hash[i] != i:
                continue

            # 构造所有终结符和函数的映射
            symbol_function_hash = {}
            for j in range(parser.grammar.n_terminal):
                function_name = create_function_name(table, i, j)
                if function_name == FN_NAME_ERROR:
                    continue
                symbol_function_hash[j] = function_name

            # 【第 1 种状态函数】只有 1 种可选的终结符
            if len(symbol_function_hash) == 1:
                symbol_id, function_name = list(symbol_function_hash.items())[0]
                f.writelines([
                    f"def s{i}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL}):\n",
                    f"    assert {PARAM_TERMINAL}.{TERMINAL_SYMBOL_ID_NAME}=={symbol_id}\n",
                    f"    return {function_name}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL})\n",
                    "\n",
                    "\n"
                ])

            # 【第 2 种状态函数】有大于 1 种可选的终结符，但所有终结符都转移到相同状态
            elif len(set(symbol_function_hash.values())) == 1:
                symbol_set_var_name = write_int_set(f, symbol_function_hash)
                function_name = list(symbol_function_hash.values())[0]
                f.writelines([
                    f"def s{i}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL}):\n",
                    f"    assert {PARAM_TERMINAL}.{TERMINAL_SYMBOL_ID_NAME} in {symbol_set_var_name}\n",
                    f"    return {function_name}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL})\n",
                    "\n",
                    "\n"
                ])

            # 【第 3 种状态函数】有大于 1 种可选的终结符和大于 1 种可选的新状态
            else:
                # 统计转移到每种状态的符号数量
                function_count = collections.defaultdict(list)
                for symbol_id, function_name in symbol_function_hash.items():
                    function_count[function_name].append(symbol_id)

                # 计算需要通过集合添加的符号：绝对值超过 3 且大于等于 50%
                n_change = len(symbol_function_hash)  # 可接收状态转移总数
                big_function_set = {function_name for function_name, symbol_id_list in function_count.items()
                                    if len(symbol_id_list) >= 3 and len(symbol_id_list) * 2 >= n_change}

                # 构造直接定义的映射
                f.write(f"SH{i}={{")
                for symbol_id, function_name in symbol_function_hash.items():
                    if function_name not in big_function_set:
                        f.write(f"{symbol_id}:{function_name},")
                f.write("}\n")

                # 构造通过集合定义的映射
                for function_name in big_function_set:
                    symbol_set_var_name = write_int_set(f, function_count[function_name])
                    f.write(f"SH{i}.update({{v:{function_name} for v in {symbol_set_var_name}}})\n")

                f.write("\n")
                f.write("\n")

                # 构造：状态函数
                # pylint: disable=C0301
                f.writelines([
                    f"def s{i}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL}):\n",
                    f"    {VAR_MOVE_ACTION}=SH{i}[{PARAM_TERMINAL}.{TERMINAL_SYMBOL_ID_NAME}]\n",
                    f"    return {VAR_MOVE_ACTION}({PARAM_STATUS_STACK},{PARAM_SYMBOL_STACK},{PARAM_TERMINAL})\n",
                    "\n",
                    "\n"
                ])

        # ------------------------------ 【写入】规约函数的映射字典 ------------------------------
        self.final_write_reduce_status_hash()

        # ------------------------------ 【构造】主函数 ------------------------------
        f.write(f"""
def parse(lexical_iterator: ms_parser.lexical.LexicalBase):
    {PARAM_STATUS_STACK} = [{parser.entrance_status_id}]
    {PARAM_SYMBOL_STACK} = []

    action = s{parser.entrance_status_id}
    {PARAM_TERMINAL} = lexical_iterator.lex()
    next_terminal = False
    try:
        while action:
            if next_terminal is True:
                {PARAM_TERMINAL} = lexical_iterator.lex()
            action, next_terminal = action({PARAM_STATUS_STACK}, {PARAM_SYMBOL_STACK}, {PARAM_TERMINAL})
    except KeyError as e:
        next_terminal_list = []
        for _ in range(10):
            if {PARAM_TERMINAL}.is_end:
                break
            next_terminal_list.append({PARAM_TERMINAL}.{TERMINAL_SYMBOL_VALUE_NAME})
            {PARAM_TERMINAL} = lexical_iterator.lex()
        next_terminal_text = \"\".join(next_terminal_list)
        raise KeyError(\"解析失败:\", next_terminal_text) from e

    return {PARAM_SYMBOL_STACK}[0]
""")

        LOGGER.info("[Write] END")


def compress_compile_lalr1(f: TextIO, parser: ParserLALR1, import_list: List[str], debug: bool = False) -> None:
    compiler = Lalr1Compiler(f, parser, import_list)
    compiler.write()


if __name__ == "__main__":
    grammar = GrammarBuilder(
        groups=[
            GGroup.create(
                name="T",
                rules=[
                    GRule.create(symbols=["a", "B", "d"], action=lambda x: f"{x[0]}{x[1]}{x[2]}"),
                    GRule.create(symbols=[], action=lambda x: "")
                ]
            ),
            GGroup.create(
                name="B",
                rules=[
                    GRule.create(symbols=["T", "b"], action=lambda x: f"{x[0]}{x[1]}"),
                    GRule.create(symbols=[], action=lambda x: "")
                ]
            ),
        ],
        terminal_type_enum=TerminalType.create_by_terminal_name_list(["a", "b", "d"]),
        start="T"
    ).build()
    parser_ = ParserLALR1(grammar)

    source_code_ = compress_compile_lalr1(parser_, [])
    print("")
    print("------------------------------ 编译结果 ------------------------------")
    print("")
    print("\n".join(source_code_))

    LOGGER.info("[Write] END")
