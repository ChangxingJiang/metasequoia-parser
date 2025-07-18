"""
解析规约函数
"""

import ast
import inspect
import typing
from typing import Callable, List, Union

from metasequoia_parser.common.grammar import GRule
from metasequoia_parser.compiler.static import *  # pylint: disable=W0401,W0614
from metasequoia_parser.utils import LOGGER


class CompileError(Exception):
    """编译错误"""


def compile_reduce_function(reduce_function: Callable, n_param: int) -> List[str]:
    """将 reduce_function 转换为 Python 源码"""
    # 获取 reduce_function 的源码位置
    filename = inspect.getsourcefile(reduce_function)  # 所在文件
    lineno = inspect.getsourcelines(reduce_function)[1]  # 起始行号

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
        return _compile_tree_node(tree_node, n_param, filename, lineno)
    except CompileError as e:
        raise CompileError(f"解析失败的源码: {reduce_function_code}") from e


def _compile_tree_node(tree_node: Union[ast.stmt, ast.expr], n_param: int, filename: str, lineno: int) -> List[str]:
    # pylint: disable=R0911
    # pylint: disable=R0912
    """解析 Python 抽象语法树的节点"""

    # 函数定义的形式
    if isinstance(tree_node, ast.FunctionDef):
        return _compile_function(tree_node, n_param, filename, lineno)

    # 使用赋值表达式定义 lambda 表达式的形式
    if isinstance(tree_node, ast.Assign):
        return _compile_tree_node(tree_node.value, n_param, filename, lineno)

    # 包含类型描述的，通过赋值语句中的 lambda 表达式
    # 样例：DEFAULT_ACTION: Callable[[GrammarActionParams], Any] = lambda x: x[0]
    if isinstance(tree_node, ast.AnnAssign):
        return _compile_tree_node(tree_node.value, n_param, filename, lineno)

    # lambda 表达式形式（可以通过赋值语句中递归触发调用）
    if isinstance(tree_node, ast.Lambda):
        return _compile_lambda(tree_node, n_param, filename, lineno)

    # Expr(value=...) —— 表达式层级（lambda 表达式）
    if isinstance(tree_node, ast.Expr):
        return _compile_tree_node(tree_node.value, n_param, filename, lineno)

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
                    return _compile_lambda(lambda_node, n_param, filename, lineno)
                raise CompileError("GRule.create 的第 2 个参数不是 lambda 表达式")
            for keyword in keywords:
                if keyword.arg == "action":
                    lambda_node = keyword.value
                    if isinstance(lambda_node, ast.Lambda):
                        return _compile_lambda(lambda_node, n_param, filename, lineno)
                    raise CompileError("GRule.create 的关键字参数 action 不是 lambda 表达式")
            raise CompileError("GRule.create 函数中没有 action 参数")

        GRule(symbols=("b",), action=lambda x: f"{x[0]}")
        if isinstance(func, ast.Name) and func.id == "GRule":
            for keyword in keywords:
                if keyword.arg == "action":
                    lambda_node = keyword.value
                    if isinstance(lambda_node, ast.Lambda):
                        return _compile_lambda(lambda_node, n_param, filename, lineno)
                    raise CompileError("GRule.create 的关键字参数 action 不是 lambda 表达式")
            raise CompileError("GRule 的初始化方法中没有 action 参数")

    raise CompileError(f"未知元素: {ast.dump(tree_node)}")


def _compile_lambda(lambda_node: ast.Lambda, n_param: int, filename: str, lineno:int) -> List[str]:
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
        node_value.id = PARAM_SYMBOL_STACK

        # 将切片器中的正数改为负数
        if not isinstance(node_slice, ast.Constant):
            raise CompileError("引用参数的切片值不是常量")
        if node_slice.value < 0:
            LOGGER.error(f"引用参数的切片值只允许使用正数: {node_slice.value} ({filename}: {lineno})")
        if node_slice.value >= n_param:
            LOGGER.error(f"引用参数的切片值超出范围: {node_slice.value} >= {n_param} ({filename}: {lineno})")
        node_slice.value = -n_param + node_slice.value

    # 将 lambda 表达式中的逻辑部分反解析为 Python 源码
    lambda_body = typing.cast(ast.AST, lambda_body)
    source_code = ast.unparse(lambda_body)

    # 为 lambda 表达式增加返回值
    source_code = f"v = {source_code}"

    return [source_code]


def _compile_function(function_node: ast.FunctionDef, n_param: int, filename: str, lineno: int) -> List[str]:
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
            node_value.id = PARAM_SYMBOL_STACK

            # 将切片器中的正数改为负数
            if not isinstance(node_slice, ast.Constant):
                raise CompileError("引用参数的切片值不是常量")
            if node_slice.value < 0:
                LOGGER.error(f"引用参数的切片值只允许使用正数: {node_slice.value} ({filename}: {lineno})")
            if node_slice.value >= n_param:
                LOGGER.error(f"引用参数的切片值超出范围: {node_slice.value} >= {n_param} ({filename}: {lineno})")
            node_slice.value = -n_param + node_slice.value

        # 如果表达式为 Return 表达式，则将 Return 表达式改为 Assign 表达式
        if isinstance(function_stmt, ast.Return):
            return_value = function_stmt.value
            return_value = typing.cast(ast.AST, return_value)
            source_code = f"v = {ast.unparse(return_value)}"
        else:
            # 如果不是 Return 表达式，则将 lambda 表达式中的逻辑部分反解析为 Python 源码
            function_stmt = typing.cast(ast.AST, function_stmt)
            source_code = ast.unparse(function_stmt)

        # 为 lambda 表达式增加返回值
        result_list.append(source_code)

    return result_list
