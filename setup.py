from distutils.core import setup

from setuptools import find_packages

with open("README.md", "r", encoding="UTF-8") as file:
    long_description = file.read()

setup(
    name="metasequoia-parser",
    version="0.1.0",
    description="水杉语法解析器提供了 LR(0)、SLR、LR(1)、LALR(1) 四种解析器，以及根据语法自动编译 LALR(1) 解析器的 Python 代码的功能。",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="changxing",
    author_email="1278729001@qq.com",
    url="https://github.com/ChangxingJiang/metasequoia-parser",
    install_requires=[],
    license="MIT License",
    packages=[package for package in find_packages() if package.startswith("metasequoia_parser")],
    platforms=["all"],
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries"
    ]
)
