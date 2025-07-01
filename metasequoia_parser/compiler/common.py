"""
工具函数
"""

__all__ = [
    "dec_to_base36"
]


def dec_to_base36(n: int) -> str:
    """
    将十进制整数转换为三十六进制字符串。

    三十六进制使用数字 0-9 和字母 a-z（小写）来表示从 0 到 35 的值。

    参数
    ----------
    n : int
        非负十进制整数，要转换为三十六进制的数值。

    返回
    -------
    str
        表示三十六进制的字符串。如果输入为 0，则返回 '0'。

    异常
    ------
    ValueError
        如果输入为负数，抛出 ValueError。

    示例
    --------
    >>> dec_to_base36(12345)
    '9ix'
    >>> dec_to_base36(0)
    '0'
    """
    if n < 0:
        raise ValueError("输入必须是非负整数")

    if n == 0:
        return "0"

    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""

    while n > 0:
        remainder = n % 36
        result = digits[remainder] + result
        n //= 36

    return result
