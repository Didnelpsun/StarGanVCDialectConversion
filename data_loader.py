# 此文件用于数据载入
import glob  # glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件，类似于Windows下的文件搜索，支持通配符操作
import os  # #os库是Python标准库，包含几百个函数,常用路径操作、进程管理、环境参数等几类。os.path子库以path为入口，用于操作和处理文件路径。

# Python音频信号处理库函数librosa，提供了创建音乐信息检索系统所需的构建块
import librosa
# 数据分析模块numpy
import numpy as np
# Pytorch也被称为torch，是深度学习框架库，Tensorflow类似，FloatTensor是其基本数据单位
import torch
# sklearn是基础机器学习库，preprocessing为预处理库，LabelBinarizer用以one-hot编码转变，用以标签二值化，即将一般数据如真假值
# 13534等数据变为01组合
from sklearn.preprocessing import LabelBinarizer
# DataLoader为数据加载器，本身是一个可迭代对象，使用iter()访问，不能使用next()访问，
# 也可以使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问
# 并利用多进程来加速batch data的处理，使用yield来使用有限的内存

# pytorch的数据加载到模型的操作顺序是这样的：
# ① 创建一个 Dataset 对象
# ② 创建一个 DataLoader 对象
# ③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

# 从预处理库中进入对应的音频预处理对象，用于将数据转换为音频数据
from preprocess import (FEATURE_DIM, FFTSIZE, FRAMES, SAMPLE_RATE, world_features)
# utility为个人定义工具集库，Normalizer为归一化对象，speakers为发音者对象，处理不同发音人
from utility import Normalizer, speakers
# random为随机数库，用来生成随机数
import random


# 构建音频数据集，将数据集输入转换为音频数据集
class AudioDataset(Dataset):
    """对音频数据集的的说明"""

    # 初始化音频数据，datadir为数据集文件路由
    # 并定义音频数据集对象所拥有的方法与属性
    def __init__(self, datadir: str):
        # 对对象进行基本初始化
        super(AudioDataset, self).__init__()
        # datadir为对象数据集地址属性
        self.datadir = datadir
        # librosa.util.find_files获取目录或目录子树中已排序的(音频)文件列表
        # 找到对应音频模型数据文件，ext表示后缀，npy就是其后缀，npy文件由python程序将数据处理而成，只能由python来解析
        # files为对象的根据路由获取npy结尾的数据集的方法
        self.files = librosa.util.find_files(datadir, ext='npy')
        # 将spearkers对象使用LabelBinarizer的fit方法来变为one-hot数据作为编码器，即域标签。
        # encoder为对象将发音者特征转化为one-hot特征值方法
        self.encoder = LabelBinarizer().fit(speakers)

    def __getitem__(self, idx):
        # p为根据idx路由获取到对应的npy结尾数据集
        p = self.files[idx]
        # 返回path路由最后的文件名，如路由为D:\SVN\Git.png，那么文件名就是Git.png
        filename = os.path.basename(p)
        # 获取对应的发音者名，将文件名以_来分割，最多只分割一次，取第一个被分割出来的字符串，就是对应的发音者名字
        # 这里是因为它的数据集的命名是这样命名所以这样处理
        speaker = filename.split(sep='_', maxsplit=1)[0]
        # tranform()的作用是通过找中心和缩放等实现标准化
        # 获取标签，即将对应的发音者名通过transform方法标准化变为one-hot格式编码，取第一个数据。
        label = self.encoder.transform([speaker])[0]
        # 使用numpy.load方法来加载对应npy数据集文件
        mcep = np.load(p)
        # 使用torch.FloatTensor将npy数据集文件转为pytorch框架中的基本变量类型FloatTensor格式
        mcep = torch.FloatTensor(mcep)
        # 使用torch.unsqueeze解压数据，因为不用加入多余维数，所以第二个参数为0
        mcep = torch.unsqueeze(mcep, 0)
        # torch.tensor用于生成新的张量，即数据转换为对应的tensor数据，dtype属性为转换目标数据的类型，类型为LongTensor
        # 对应的FloatTensor方法将数据变为FloatTensor
        # 所以最后返回的数据为解压的音频数据集mecp，转换为LongTensor类型的名为speaker变量的speakers值（按键值对的键值取出）
        # 以及转换为FloatTensor类型的标签
        return mcep, torch.tensor(speakers.index(speaker), dtype=torch.long), torch.FloatTensor(label)

    # speaker_encoder返回数据集的解码器，即域标签
    def speaker_encoder(self):
        return self.encoder

    # __len__方法定义数据集对象的大小
    def __len__(self):
        return len(self.files)


# 调用数据集的文件数据载入方法，datadir为数据集文件路由，批处理每次的并行数据为4条，进行shuffle操作，模式为训练
# shuffile操作是针对处理大量和多进程操作时，为了优化网络和IO操作的处理方式，即对数据进行分区与合并加快数据处理，代价很大
# 使用两个子进程处理数据
def data_loader(datadir: str, batch_size=4, shuffle=True, mode='train', num_workers=2):
    """如果模式是训练数据，它应该包含所有npy文件的训练集
    或者，模式是测试，数据集路由应该只包含wav格式文件，即音频源文件
    """

    # 调用之前定义的转换音频数据集的方法AudioDataset将对应文件路由转换为可处理的数据集
    dataset = AudioDataset(datadir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 返回文件加载器
    return loader


class TestSet(object):
    """对于测试数据的的说明"""

    def __init__(self, datadir: str):
        super(TestSet, self).__init__()
        self.datadir = datadir
        # 定义对象norm方法为数据正则化方法
        self.norm = Normalizer()

    def choose(self):
        '''为测试数据选择一个发音者'''

        # 根据speakers这个序列中随机取出一个speaker作为目标发音者
        r = random.choice(speakers)
        return r

    # 默认目标发音者为空值
    def test_data(self, src_speaker=None):
        '''为转换数据选择一个发音者'''
        # 如果传入了目标发音者，即将这个参数赋值给r_s变量
        if src_speaker:
            r_s = src_speaker
        # 如果没有参数传入
        else:
            # 就自动调用对象的choose方法随机选取一个发音者
            r_s = self.choose()
        # 将这个目标发音者的名称和原本的数据集地址拼接在一起作为新路径
        # 因为这个数据集是以发音者的名字作为子集的，所以要找到对应发音者的子数据集必须这样处理
        # 如果采用不同的数据集命名方式那么处理方式就会不同
        p = os.path.join(self.datadir, r_s)
        # 根据路由找到该路径下的所有wav格式文件
        wavfiles = librosa.util.find_files(p, ext='wav')

        res = {}
        # 遍历这个数据集对象
        for f in wavfiles:
            # 获取对应的文件名
            filename = os.path.basename(f)
            # librosa.load方法f为文件地址，sr参数为采样率，如果保存原有采样率则赋值为None
            # dtype为精度，将返回一个音频时间序列和一个音频采样率
            # 因为音频采样率不被使用，所以以_作为变量名占位保存
            # 该方法还有mono ：bool，是否将信号转换为单声道
            # offset ：float，在此时间之后开始阅读（以秒为单位）
            # duration：float，仅加载这么多的音频（以秒为单位）
            wav, _ = librosa.load(f, sr=SAMPLE_RATE, dtype=np.float64)
            f0, timeaxis, sp, ap, coded_sp = world_features(wav, SAMPLE_RATE, FFTSIZE, FEATURE_DIM)
            coded_sp_norm = self.norm.forward_process(coded_sp.T, r_s)

            if not res.__contains__(filename):
                res[filename] = {}
            res[filename]['coded_sp_norm'] = np.asarray(coded_sp_norm)
            res[filename]['f0'] = np.asarray(f0)
            res[filename]['ap'] = np.asarray(ap)
        return res, r_s


if __name__ == '__main__':
    # t = TestSet('data/speakers_test')
    # # mcep, f0, speaker = t[0]
    # # print(speaker)
    # # print(mcep)
    # # print(f0)
    # # print(np.ma.log(f0))
    # d, speaker = t.test_data()

    # for filename, content in d.items():
    #     coded_sp_norm = content['coded_sp_norm']
    #     print(content['coded_sp_norm'].shape)
    #     f_len = coded_sp_norm.shape[1]
    #     if  f_len >= FRAMES: 
    #         pad_length = FRAMES-(f_len - (f_len//FRAMES) * FRAMES)
    #     elif f_len < FRAMES:
    #         pad_length = FRAMES - f_len

    #     coded_sp_norm = np.hstack((coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length))))
    #     print('after:' , coded_sp_norm.shape)
    # print(t[1])
    ad = AudioDataset('./data/processed')
    print(len(ad))

    data, s, label = ad[500]
    print(data, label)
    # loader = data_loader('./data/processed', batch_size=4)   

    # for i_batch, batch_data in enumerate(loader):
    #     # print(batch_data)
    #     # print(batch_data[0])
    #     print(batch_data[1])
    #     print(batch_data[2])
    #     if i_batch == 2:
    #         break
