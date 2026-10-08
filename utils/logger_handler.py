import logging
import os
from datetime import datetime

#日志工具



if __package__:
    from .path_tool import get_abs_path
else:
    # 支持直接执行：python utils/logger.py
    from path_tool import get_abs_path

# 日志保存根目录
LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志基本类型：error、info、debug、warning、critical
# 基础日志格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)


#配置日志生成器
def get_logger(
    name: str = "langchain_react",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)#获取入职前对象
    logger.setLevel(logging.DEBUG)  # 设置日志记录器的最低级别为DEBUG

    # 避免重复添加 Handler
    if logger.handlers:
        return logger


    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件 Handler
    if not log_file:
        log_file = os.path.join(
            LOG_ROOT,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log",
        )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()


# if __name__ == "__main__":
#     logger.info("信息日志")
#     logger.error("错误日志")
#     logger.warning("警告日志")
#     logger.debug("调试日志")
