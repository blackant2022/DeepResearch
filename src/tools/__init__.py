"""工具包：导入时自动注册全部内置工具。"""
from src.tools import builtin_tools as _builtin  # noqa: F401

__all__ = ["registry"]

from src.tools.base import registry
