import logging

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 记录 INFO 级别及以上的所有日志
    format='[%(levelname)s] %(asctime)s %(processName)s %(name)s: %(message)s', # 定义日志格式
    datefmt="%H:%M:%S.%f",  # 添加毫秒格式
    handlers=[
        logging.FileHandler("beavr_run.log"), # 输出到文件
        logging.StreamHandler()               # 输出到屏幕
    ]
)