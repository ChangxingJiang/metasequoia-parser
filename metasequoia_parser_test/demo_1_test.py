from metasequoia_parser.lexical import create_lexical_iterator_by_name_list
from metasequoia_parser_test.demo_1_build import grammar
from metasequoia_parser_test.demo_1_code import parse

if __name__ == "__main__":
    print(parse(create_lexical_iterator_by_name_list(grammar, ["a", "d"])))
