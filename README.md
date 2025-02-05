# metasequoia-parser 水杉语法解析器

水杉语法解析器提供了 LR(0)、SLR、LR(1)、LALR(1) 四种解析器，以及根据语法自动编译 LALR(1) 解析器的 Python 代码的功能。

## 安装方法

```bash
pip install metasequoia-parser
```

## 使用方法

### 定义语法规则

使用 `create_grammar` 函数创建语法类的构造器，使用构造器的 `group_append` 方法添加语义组，在语法定义完成后调用构造器的 `build` 方法构造语法类（`Grammar`）。

使用 `create_group` 函数创建语义组，其参数为语义组名称（`name`）和语义组备选规则的列表（`rules`）。

使用 `create_rule` 函数创建备选规则，其参数为符号列表（`symbols`）和规约函数（`action`）。

```python
import metasequoia_parser as ms_parser

grammar = ms_parser.create_grammar(
    start="S",
    terminal_type_enum=ms_parser.TerminalType.create_by_terminal_name_list(["a", "b", "c"])
).group_append(ms_parser.create_group(name="S", rules=[
    ms_parser.create_rule(symbols=["a", "B"], action=lambda x: f"{x[0]}{x[1]}"),
    ms_parser.create_rule(symbols=["a", "B", "c"], action=lambda x: f"{x[0]}{x[1]}{x[2]}"),
])).group_append(ms_parser.create_group(name="B", rules=[
    ms_parser.create_rule(symbols=["b"], action=lambda x: f"{x[0]}"),
])).build()
```

### 构造解析器

将语法类作为参数构造解析器：

```python
parser = ms_parser.parser.ParserLR0(grammar)  # 构造 LR(0) 解析器
parser = ms_parser.parser.ParserSLR(grammar)  # 构造 SLR 解析器
parser = ms_parser.parser.ParserLR1(grammar)  # 构造 LR(1) 解析器
parser = ms_parser.parser.ParserLALR1(grammar)  # 构造 LALR(1) 解析器
```

### 使用解析器对象

直接使用解析器解析词法解析的结果，适用于测试场景：

```python
from metasequoia_parser.lexical import create_lexical_iterator_by_name_list

parser.parse(create_lexical_iterator_by_name_list(grammar, ["a", "b"]))
```

在正式场景下，需实现词法解析器（`ms_parser.lexical.LexicalBase`）抽象基类，作为 `parse` 方法的参数。

### 编译解析器

使用 `ms_parser.compiler.compile_lalr1` 函数，可以将 LALR(1) 解析器对象编译为 Python 代码，适用于生产环境：

```python
source_code = ms_parser.compiler.compile_lalr1(parser, import_list=[
    "from my_package import my_tree_node"
])
with open("my_parser.py", "w+", encoding="UTF-8") as file:
    for row in source_code:
        file.write(f"{row}\n")
```

### 使用编译的解析器

引用编译的 Python 代码所在文件中的 parse 函数，其参数及返回值等价于解析器对象的 `parse` 方法。

```python
from metasequoia_parser.lexical import create_lexical_iterator_by_name_list
from my_parser import parse  # 引用编译的 Python 代码所在文件中的 parse 函数

parse(create_lexical_iterator_by_name_list(grammar, ["a", "b"]))  # 测试场景
```

在正式场景下，需实现词法解析器（`ms_parser.lexical.LexicalBase`）抽象基类，作为 `parser.parse` 函数的参数。