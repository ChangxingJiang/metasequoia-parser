from metasequoia_parser.common import TerminalType
from metasequoia_parser.common.grammar import GGroup, GRule, GrammarBuilder
from metasequoia_parser.compiler.compress_lalr1 import compress_compile_lalr1
from metasequoia_parser.parser.lalr1 import ParserLALR1

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

if __name__ == "__main__":
    parser_ = ParserLALR1(grammar)
    with open("demo_1_code.py", "w", encoding="UTF-8") as file:
        source_code_ = compress_compile_lalr1(file, parser_, [])
