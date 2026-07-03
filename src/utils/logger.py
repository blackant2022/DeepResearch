"""
src/utils/logger.py — 统一日志
彩色输出 + 结构化前缀，便于在多Agent运行时看清谁在做什么。
所有 Agent / 工具 / 中间件都用它，方便简历演示时展示清晰的执行轨迹。
"""
import logging
import sys

_COLORS = {
    "DEBUG": "\033[37m", "INFO": "\033[36m", "WARNING": "\033[33m",
    "ERROR": "\033[31m", "CRITICAL": "\033[41m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<7}{_RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter("%(levelname)s │ %(name)-18s │ %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
