"""
编译器

编译器用于将定义的语法规则，生成解析代码。

- 为每个状态构造两个 **状态函数**，分别处理接收到终结符（Action 或 Reduce）和非终结符（GOTO）的逻辑
- 在每个 **状态函数** 中，定义一个终结符 / 非终结符到 **行为函数** 的字典；如果终结符 / 非终结符不在字典中，则抛出异常
- 在每个 **行为函数** 中，接收状态栈和对象栈并原地修改它们，并返回下一个需要执行的 **行为函数**
- 将规约函数的规约逻辑合并到行为函数中，如果需要 ACCEPT 行为，则返回 None 跳出解析循环
- 将相同状态压入状态栈的移进函数可以共用，因为它们都需要把参数中的终结符压入对象栈
- 将相同状态压入状态栈、且规约生成的非终结符相同、且规约逻辑相同的规约函数可以共用，因为它们的逻辑是完全相同的

因为 Python 不支持为递归，所以采用 while 循环的形式实现，不断调用 **行为函数** 直到 **行为函数** 返回 None 值表示已经 ACCEPT

## 解析逻辑的伪代码

接收参数：lexical_iterator（词法解析器结果迭代器）

status_stack = []  # 初始化状态栈
symbol_stack = []  # 初始化对象栈

status_action = init_action  # 初始化状态函数
while action:
    terminal = lexical_iterator.lex()  # 词法解析出下一个终结符
    status_action = status_action(status_stack, symbol_stack, terminal)  # 调用状态函数

return symbol_stack[0]  # 返回最终结果

## 状态函数的伪代码

在状态函数之外（保证单例），定义该状态函数下终结符 / 非终结符到 **行为函数** 的字典：

STATUS_xx_TERMINAL_ACTION_HASH = {
    terminal: move_action
}

在状态函数之内，伪代码逻辑如下：

move_action = STATUS_xx_TERMINAL_ACTION_HASH[terminal]
return move_action(status_stack, symbol_stack, terminal)

"""

from metasequoia_parser.compiler.lalr1 import compile_lalr1
from metasequoia_parser.compiler.compress_lalr1 import compress_compile_lalr1
