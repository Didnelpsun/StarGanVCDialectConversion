# 主文件，串联其整个项目

import os
import argparse
# 引入定义的两个自定义文件
from solver import Solver
from data_loader import data_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in 'true'


def main(config):
    # 目的是为了快速训练
    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异
    # 大部分情况下，设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True
    # 如果目录不存在就创建目录
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    # 数据加载器
    # 调用自定义的数据加载器方法加载对应的数据
    dloader = data_loader(config.data_dir, batch_size=config.batch_size, mode=config.mode,
                          num_workers=config.num_workers)
    # 为训练和测试StarGAN的Solver类，传入数据加载器与配置
    solver = Solver(dloader, config)
    # 当模式为训练的时候就调用train方法进行训练，否则调用test方法进行测试
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    # 设置评价标准损失函数的各项权值
    parser.add_argument('--lambda_cycle', type=float, default=3, help='循环一致性损失的权值')
    parser.add_argument('--lambda_cls', type=float, default=2, help='域分类损失的权值')
    parser.add_argument('--lambda_identity', type=float, default=2, help='身份一致性损失的权值')
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=4, help='设置最小批处理大小')
    parser.add_argument('--num_iters', type=int, default=200000, help='训练的总迭代次数')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='衰减学习率的迭代次数')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='生成器G的学习频率')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='判别器D的学习频率')
    parser.add_argument('--c_lr', type=float, default=0.0001, help='域分类器C的学习频率')
    parser.add_argument('--n_critic', type=int, default=5, help='每次G更新时的D更新次数')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam优化器的beta2参数')
    parser.add_argument('--resume_iters', type=int, default=None, help='从此步骤恢复培训')
    # 测试配置
    parser.add_argument('--test_iters', type=int, default=200000, help='从这个步骤开始训练模型')
    parser.add_argument('--src_speaker', type=str, default=None, help='测试模型的源发音者')
    parser.add_argument('--trg_speaker', type=str, default="['SF1', 'TM1']",
                        help='目标发音者的字符串列表表示，例如“[a，b]”')
    # 其他配置
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # 目录
    # 数据目录
    parser.add_argument('--data_dir', type=str, default='data/processed')
    # 测试目录
    parser.add_argument('--test_dir', type=str, default='data/speakers_test')
    # 记录目录
    parser.add_argument('--log_dir', type=str, default='starganvc/logs')
    # 模型保存目录
    parser.add_argument('--model_save_dir', type=str, default='starganvc/models')
    # 样本目录
    parser.add_argument('--sample_dir', type=str, default='starganvc/samples')
    # 结果目录
    parser.add_argument('--result_dir', type=str, default='starganvc/results')

    # 步长
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=100000)
    # 获取所有的设置
    config = parser.parse_args()
    # 打印设置
    print(config)
    # 调用main函数
    main(config)
