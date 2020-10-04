# 用于记录训练过程

import tensorflow as tf


class Logger(object):
    """tensorflow记录器"""

    def __init__(self, log_dir):
        """初始化摘要编写器"""
        # summary.FileWriter指定一个文件用来保存图
        # 定义一个记录器在对应的文件夹中
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """添加标量摘要"""
        # 等价于summary.value.add(tag=tag,simple_value=value)
        # 添加对应的记录
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # add_summary()方法将训练过程数据保存在filewriter指定的文件中
        self.writer.add_summary(summary, step)
