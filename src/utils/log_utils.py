import logging


def get_logger(filename, verbosity=1, name=None):
    """
    获取一个配置好的日志记录器。

    :param filename: 日志文件名
    :param verbosity: 日志输出级别（0: DEBUG, 1: INFO, 2: WARNING）
    :param name: 日志记录器的名称
    :return: 配置好的日志记录器
    """
    # 定义日志级别字典，键为级别名称，值为对应的logging模块中的级别常量
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

    # 定义日志格式
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    # 获取日志记录器实例，如果传入了name，则根据name获取；否则获取root日志记录器
    logger = logging.getLogger(name)

    # 设置日志记录器的日志级别
    logger.setLevel(level_dict[verbosity])

    # 创建并配置文件处理器（注释掉的部分）
    # fh = logging.FileHandler(filename, "w")
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # 创建并配置控制台输出处理器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
