"""
字符集常量：用于定义词法解析规则
"""

__all__ = [
    "ALPHA_UPPER",
    "ALPHA_LOWER",
    "ALPHA",
    "NUMBER",
    "HEX_NUMBER",
    "OCT_NUMBER",
]

# 大写字母
ALPHA_UPPER = frozenset({"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                         "T", "U", "V", "W", "X", "Y", "Z"})

# 小写字母
ALPHA_LOWER = frozenset({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                         "t", "u", "v", "w", "x", "y", "z"})

# 英文字母
ALPHA = frozenset(ALPHA_UPPER | ALPHA_LOWER)

# 数字
NUMBER = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})

# 十六进制数
HEX_NUMBER = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        "A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f"})

# 八进制数
OCT_NUMBER = frozenset({"0", "1", "2", "3", "4", "5", "6", "7"})
