import logging
import sys

def setup_logging(name=None, log_file=None):
    """设置 logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 控制台输出格式
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出格式（包含更多信息）
    if log_file:
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d:%(funcName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger