import metasequoia_parser as ms_parser
from metasequoia_parser.common import TerminalType
from metasequoia_parser.common.grammar import GGroup, GrammarBuilder
from metasequoia_parser.compiler.compress_lalr1 import compress_compile_lalr1
from metasequoia_parser.parser.lalr1 import ParserLALR1


class DemoTerminalType(TerminalType):
    END = 0
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5


grammar = GrammarBuilder(
    groups=[
        GGroup.create(
            name="T",
            rules=[
                ms_parser.create_rule(
                    symbols=[DemoTerminalType.A, DemoTerminalType.B, "opt_c", DemoTerminalType.D, DemoTerminalType.E],
                    action=lambda x: f"{x[0]}{x[1]}{x[2]}{x[3]}{x[4]}")
            ]
        ),
        GGroup.create(
            name="opt_c",
            rules=[
                ms_parser.create_rule(symbols=[DemoTerminalType.C], action=ms_parser.template.action.RETURN_TRUE),
                ms_parser.create_rule(symbols=[], action=ms_parser.template.action.RETURN_FALSE)
            ]
        ),
    ],
    terminal_type_enum=DemoTerminalType,
    start="T"
).build()

if __name__ == "__main__":
    parser_ = ParserLALR1(grammar)
    with open("demo_2_code.py", "w", encoding="UTF-8") as file:
        source_code_ = compress_compile_lalr1(file, parser_, [])
