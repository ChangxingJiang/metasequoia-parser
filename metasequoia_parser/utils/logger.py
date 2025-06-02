"""
日志模块
"""

import logging

# 创建一个日志 Logger
LOGGER = logging.getLogger("metasequoia-parser")
LOGGER.setLevel(logging.INFO)

# 创建一个写出到控制台的 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter("%(asctime)s %(message)s")
console_handler.setFormatter(formatter)

# 将 Handler 添加给 Logger
LOGGER.addHandler(console_handler)
