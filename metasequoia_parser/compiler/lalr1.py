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

import ast
import inspect
import typing
from typing import Callable, List, Union

from metasequoia_parser.common import ActionAccept, ActionBase, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import TerminalType
from metasequoia_parser.common.grammar import GGroup, GRule, GrammarBuilder
from metasequoia_parser.parser.lalr1 import ParserLALR1


class CompileError(Exception):
    """编译错误"""


def compile_reduce_function(reduce_function: Callable, n_param: int) -> List[str]:
    """将 reduce_function 转换为 Python 源码"""
    # 获取 reduce_function 的源码，并剔除首尾的空格、英文半角逗号和换行符
    reduce_function_code = inspect.getsource(reduce_function)
    reduce_function_code = reduce_function_code.strip(" ,\n")

    # 将 reduce_function 的源码解析为抽象语法树
    try:
        tree_node_module = ast.parse(reduce_function_code)
    except SyntaxError as e:
        # 末尾存在无法匹配的 ')'，可能是将更包含了更外层的括号
        if e.msg == "unmatched ')'" and reduce_function_code.endswith(")"):
            reduce_function_code = reduce_function_code[:-1]
            tree_node_module = ast.parse(reduce_function_code)
        else:
            raise e

    # 如果 reduce_function 的源码中包含多个表达式，则抛出异常
    if len(tree_node_module.body) > 1:
        raise CompileError(f"规约函数源码包含多条表达式: {reduce_function_code}")

    tree_node = tree_node_module.body[0]
    try:
        return _compile_tree_node(tree_node, n_param)
    except CompileError as e:
        raise CompileError(f"解析失败的源码: {reduce_function_code}") from e


def _compile_tree_node(tree_node: Union[ast.stmt, ast.expr], n_param: int) -> List[str]:
    # pylint: disable=R0911
    # pylint: disable=R0912
    """解析 Python 抽象语法树的节点"""

    # 函数定义的形式
    if isinstance(tree_node, ast.FunctionDef):
        return _compile_function(tree_node, n_param)

    # 使用赋值表达式定义 lambda 表达式的形式
    if isinstance(tree_node, ast.Assign):
        return _compile_tree_node(tree_node.value, n_param)

    # 包含类型描述的，通过赋值语句中的 lambda 表达式
    # 样例：DEFAULT_ACTION: Callable[[GrammarActionParams], Any] = lambda x: x[0]
    if isinstance(tree_node, ast.AnnAssign):
        return _compile_tree_node(tree_node.value, n_param)

    # lambda 表达式形式（可以通过赋值语句中递归触发调用）
    if isinstance(tree_node, ast.Lambda):
        return _compile_lambda(tree_node, n_param)

    # Expr(value=...) —— 表达式层级（lambda 表达式）
    if isinstance(tree_node, ast.Expr):
        return _compile_tree_node(tree_node.value, n_param)

    # Call(func=..., args=[...], keywords=[...]) —— 函数调用（lambda 表达式）
    if isinstance(tree_node, ast.Call):
        func = tree_node.func
        args = tree_node.args
        keywords = tree_node.keywords

        # GRule.create(symbols=..., action=...)
        # create_rule(symbols=..., action=...)
        # 如果 action 的源码在这一行，则说明一定是 lambda 表达式，否则源码位于函数定义的位置
        # pylint: disable=R0916
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and (func.value.id == "GRule" and func.attr == "create" or
                     func.value.id == "ms_parser" and func.attr == "create_rule")):
            if len(args) >= 2:  # 如果使用顺序参数，则应该是第 2 个参数
                lambda_node = args[1]
                if isinstance(lambda_node, ast.Lambda):
                    return _compile_lambda(lambda_node, n_param)
                raise CompileError("GRule.create 的第 2 个参数不是 lambda 表达式")
            for keyword in keywords:
                if keyword.arg == "action":
                    lambda_node = keyword.value
                    if isinstance(lambda_node, ast.Lambda):
                        return _compile_lambda(lambda_node, n_param)
                    raise CompileError("GRule.create 的关键字参数 action 不是 lambda 表达式")
            raise CompileError("GRule.create 函数中没有 action 参数")

        GRule(symbols=("b",), action=lambda x: f"{x[0]}")
        if isinstance(func, ast.Name) and func.id == "GRule":
            for keyword in keywords:
                if keyword.arg == "action":
                    lambda_node = keyword.value
                    if isinstance(lambda_node, ast.Lambda):
                        return _compile_lambda(lambda_node, n_param)
                    raise CompileError("GRule.create 的关键字参数 action 不是 lambda 表达式")
            raise CompileError("GRule 的初始化方法中没有 action 参数")

    raise CompileError(f"未知元素: {ast.dump(tree_node)}")


def _compile_lambda(lambda_node: ast.Lambda, n_param: int) -> List[str]:
    """解析 lambda 表达式形式的递归函数"""
    # 获取参数名
    args = lambda_node.args.args
    if len(args) > 1:
        raise CompileError("递归逻辑函数的参数超过 1 个")
    arg_name = args[0].arg

    # 遍历 lambda 表达式中所有节点，修改参数引用中的参数名和切片值
    lambda_body = lambda_node.body
    lambda_body = typing.cast(ast.AST, lambda_body)
    for node in ast.walk(lambda_body):
        # 跳过非参数引用节点
        if not isinstance(node, ast.Subscript):
            continue
        node_value = node.value
        node_slice = node.slice
        if not isinstance(node_value, ast.Name) or not node_value.id == arg_name:
            continue

        # 将参数名修改为 symbol_stack（直接从符号栈中获取）
        node_value.id = "symbol_stack"

        # 将切片器中的正数改为负数
        if not isinstance(node_slice, ast.Constant):
            raise CompileError("引用参数的切片值不是常量")
        if node_slice.value < 0:
            raise CompileError("引用参数的切片值只允许使用正数")
        node_slice.value = -n_param + node_slice.value

    # 将 lambda 表达式中的逻辑部分反解析为 Python 源码
    lambda_body = typing.cast(ast.AST, lambda_body)
    source_code = ast.unparse(lambda_body)

    # 为 lambda 表达式增加返回值
    source_code = f"symbol_value = {source_code}"

    return [source_code]


def _compile_function(function_node: ast.FunctionDef, n_param: int) -> List[str]:
    """解析函数定义形式的递归函数"""
    # 获取参数名
    args = function_node.args.args
    if len(args) > 1:
        raise CompileError("递归逻辑函数的参数超过 1 个")
    arg_name = args[0].arg

    # 遍历 lambda 表达式中所有节点，修改参数引用中的参数名和切片值
    function_body = function_node.body
    result_list = []
    for function_stmt in function_body:
        function_stmt = typing.cast(ast.AST, function_stmt)
        for node in ast.walk(function_stmt):
            # 跳过非参数引用节点
            if not isinstance(node, ast.Subscript):
                continue
            node_value = node.value
            node_slice = node.slice
            if not isinstance(node_value, ast.Name) or not node_value.id == arg_name:
                continue

            # 将参数名修改为 symbol_stack（直接从符号栈中获取）
            node_value.id = "symbol_stack"

            # 将切片器中的正数改为负数
            if not isinstance(node_slice, ast.Constant):
                raise CompileError("引用参数的切片值不是常量")
            if node_slice.value < 0:
                raise CompileError("引用参数的切片值只允许使用正数")
            node_slice.value = -n_param + node_slice.value

        # 如果表达式为 Return 表达式，则将 Return 表达式改为 Assign 表达式
        if isinstance(function_stmt, ast.Return):
            return_value = function_stmt.value
            return_value = typing.cast(ast.AST, return_value)
            source_code = f"symbol_value = {ast.unparse(return_value)}"
        else:
            # 如果不是 Return 表达式，则将 lambda 表达式中的逻辑部分反解析为 Python 源码
            function_stmt = typing.cast(ast.AST, function_stmt)
            source_code = ast.unparse(function_stmt)

        # 为 lambda 表达式增加返回值
        result_list.append(source_code)

    return result_list


def compile_lalr1(parser: ParserLALR1, import_list: List[str], debug: bool = False) -> List[str]:
    # pylint: disable=R0912
    # pylint: disable=R0914
    # pylint: disable=R0915
    """编译 LALR(1) 解析器"""
    table = parser.table

    source_script = ["\"\"\""]
    for symbol_name, symbol_id in parser.grammar.symbol_name_id_hash.items():
        if symbol_id in parser.nonterminal_id_to_start_lr0_id_list_hash:
            lr0_id_list = parser.nonterminal_id_to_start_lr0_id_list_hash[symbol_id]
            lr0_list = [parser.lr0_list[lr0_id] for lr0_id in lr0_id_list]
            source_script.append(f"{symbol_name}({symbol_id}): {lr0_list}")
        else:
            source_script.append(f"{symbol_name}({symbol_id}): 终结符")
    source_script.append("\"\"\"")

    # 最终生成的源码列表
    source_script.extend([
        "",
        "from typing import Any, Callable, List, Optional, Tuple",
        "",
        "import metasequoia_parser as ms_parser",
        "",
    ])
    source_script.extend(import_list)
    source_script.append("")
    source_script.append("")

    # 如果 ACTION + GOTO 表为空，则抛出异常
    if len(table) == 0 or len(table[0]) == 0:
        raise CompileError("ACTION + GOTO 表为空")

    n_status = len(table)

    # ------------------------------ 【构造】移进行为函数 ------------------------------
    built_shift_action = set()
    for i in range(n_status):
        for j in range(parser.grammar.n_terminal):
            action: ActionBase = table[i][j]
            if isinstance(action, ActionShift):
                if action.status not in built_shift_action:
                    built_shift_action.add(action.status)
                    function_name = f"action_shift_{action.status}"
                    # pylint: disable=C0301
                    source_script.extend([
                        f"def {function_name}(status_stack: List[int], symbol_stack: List[Any], terminal: ms_parser.symbol.Terminal) -> Tuple[Optional[Callable], bool]:",
                        f"    status_stack.append({action.status})  # 向状态栈中压入常量",
                        "    symbol_stack.append(terminal.symbol_value)  # 向符号栈中压入当前终结符的值",
                        f"    return status_{action.status}, True  # 返回状态栈常量状态的终结符行为函数",
                        "",
                        "",
                    ])

    # ------------------------------ 【构造】规约行为函数 ------------------------------
    reduce_function_hash = {}
    for i in range(n_status):
        reduce_function_idx = 1
        for j in range(parser.grammar.n_terminal):
            action: ActionBase = table[i][j]
            if isinstance(action, ActionReduce):
                nonterminal_id = action.reduce_name
                reduce_function = action.reduce_function

                # 如果当前非终结符的相同规约逻辑已处理，则不需要重复添加
                if (nonterminal_id, reduce_function) in reduce_function_hash:
                    continue

                # 生成规约行为函数的名称
                function_name = f"action_reduce_{i}_{reduce_function_idx}"
                reduce_function_idx += 1
                reduce_function_hash[(nonterminal_id, reduce_function)] = function_name

                # 添加规约行为函数
                n_param = action.n_param
                # pylint: disable=C0301
                source_script.append(
                    f"def {function_name}(status_stack: List[int], symbol_stack: List[Any], _: ms_parser.symbol.Terminal) -> Tuple[Optional[Callable], bool]:"
                )
                for source_row in compile_reduce_function(reduce_function, n_param):
                    source_script.append(f"    {source_row}")
                # pylint: disable=C0301
                source_script.append(
                    f"    next_status = STATUS_SYMBOL_GOTO_HASH[(status_stack[-{n_param + 1}], {nonterminal_id})]  # 根据状态和生成的非终结符获取需要 GOTO 的状态"
                )
                if n_param > 0:
                    source_script.extend([
                        f"    symbol_stack[-{n_param}:] = [symbol_value]  # 出栈 {n_param} 个参数，入栈新生成的非终结符的值",
                        f"    status_stack[-{n_param}:] = [next_status]  # 出栈 {n_param} 个参数，入栈 GOTO 的新状态",
                    ])
                else:
                    source_script.extend([
                        f"    symbol_stack.append(symbol_value)  # 出栈 {n_param} 个参数（不出栈），入栈新生成的非终结符的值",
                        f"    status_stack.append(next_status)  # 出栈 {n_param} 个参数（不出栈），入栈 GOTO 的新状态",
                    ])

                source_script.extend([
                    "    return STATUS_HASH[next_status], False  # 返回新状态的行为函数",
                    "",
                    ""
                ])

    # ------------------------------ 【构造】接收行为函数 ------------------------------
    # pylint: disable=C0301
    source_script.extend([
        "def action_accept(_1: List[int], _2: List[Any], _3: ms_parser.symbol.Terminal) -> Tuple[Optional[Callable], bool]:",
        "    return None, True",
        "",
        ""
    ])

    # ------------------------------ 【构造】终结符 > 行为函数的字典；状态函数 ------------------------------
    for i in range(n_status):
        # 构造：终结符 > 行为函数的字典
        source_script.append(f"STATUS_{i}_TERMINAL_ACTION_HASH = {{")
        for j in range(parser.grammar.n_terminal):
            action: ActionBase = table[i][j]
            if isinstance(action, ActionShift):
                function_name = f"action_shift_{action.status}"
            elif isinstance(action, ActionReduce):
                nonterminal_id = action.reduce_name
                reduce_function = action.reduce_function
                function_name = reduce_function_hash[(nonterminal_id, reduce_function)]
            elif isinstance(action, ActionAccept):
                function_name = "action_accept"
            else:
                continue  # 抛出异常，不需要额外处理

            source_script.append(f"    {j}: {function_name},")
        source_script.append("}")
        source_script.append("")
        source_script.append("")

        # 构造：状态函数
        # pylint: disable=C0301
        source_script.extend([
            f"def status_{i}(status_stack: List[int], symbol_stack: List[Any], terminal: ms_parser.symbol.Terminal) -> Tuple[Optional[Callable], bool]:",
            f"    move_action = STATUS_{i}_TERMINAL_ACTION_HASH[terminal.symbol_id]  # 通过哈希映射获取行为函数",
            "    return move_action(status_stack, symbol_stack, terminal)  # 执行行为函数",
            "",
            ""
        ])

    # ------------------------------ 【构造】状态 + 非终结符 > GOTO 状态的字典 ------------------------------
    source_script.append("STATUS_SYMBOL_GOTO_HASH = {")
    for i in range(n_status):
        for j in range(parser.grammar.n_terminal, parser.grammar.n_symbol):
            action: ActionBase = table[i][j]
            if isinstance(action, ActionGoto):
                source_script.append(f"    ({i}, {j}): {action.status}, ")
    source_script.append("}")
    source_script.append("")
    source_script.append("")

    # ------------------------------ 【构造】状态 > 状态函数的字典 ------------------------------
    source_script.append("# 状态 > 状态函数的字典")
    source_script.append("STATUS_HASH = {")
    for i in range(n_status):
        source_script.append(f"    {i}: status_{i},")
    source_script.append("}")
    source_script.append("")
    source_script.append("")

    # ------------------------------ 【构造】主函数 ------------------------------
    source_script.extend([
        "def parse(lexical_iterator: ms_parser.lexical.LexicalBase):",
        f"    status_stack = [{parser.entrance_status_id}]  # 初始化状态栈",
        "    symbol_stack = []  # 初始化对象栈",
        "",
        f"    action = status_{parser.entrance_status_id}  # 初始化状态函数",
        "    terminal = lexical_iterator.lex()  # 词法解析出下一个终结符",
        "    next_terminal = False",
        "    try:",
        "        while action:",
        "            if next_terminal is True:",
        "                terminal = lexical_iterator.lex()  # 词法解析出下一个终结符",
        # "            print(terminal, status_stack, symbol_stack)",
        "            action, next_terminal = action(status_stack, symbol_stack, terminal)  # 调用状态函数",
        "    except KeyError as e:",
        "        next_terminal_list = []",
        "        for _ in range(10):",
        "            if terminal.is_end:",
        "                break",
        "            next_terminal_list.append(terminal.symbol_value)",
        "            terminal = lexical_iterator.lex()",
        "        next_terminal_text = \"\".join(next_terminal_list)",
        "        raise KeyError(\"解析失败:\", next_terminal_text) from e",
        "",
        "    return symbol_stack[0]  # 返回最终结果",
        "",
    ])

    return source_script


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

    source_code_ = compile_lalr1(parser_, [])
    print("")
    print("------------------------------ 编译结果 ------------------------------")
    print("")
    print("\n".join(source_code_))

    # exec("\n".join(source_code_))
