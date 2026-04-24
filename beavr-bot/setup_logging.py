import logging

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 记录 INFO 级别及以上的所有日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 定义日志格式
    handlers=[
        logging.FileHandler("beavr_run.log"), # 输出到文件
        logging.StreamHandler()               # 输出到屏幕
    ]
)