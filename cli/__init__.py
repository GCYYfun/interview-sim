"""
CLI 应用导出

提供控制台应用的统一入口
"""

from .app import InterviewSimApp, main
from .menus import MenuSystem

__all__ = [
    "InterviewSimApp",
    "main",
    "MenuSystem",
]
