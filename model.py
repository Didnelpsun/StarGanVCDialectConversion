# 该文件为模型文件，主要是构建神经网络模型

import torch
# torch.nn为一个神经网络构建工具箱
import torch.nn as nn


# 自定义模型需要继承nn.Module来实现，下采样网络
class Down2d(nn.Module):
    """将降维为二维方法的说明"""
    # 初始化，参数为输入信道数，输出信道数，卷积核数，步长，填充层数，即对输入的每一条边，补充0的层数
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        # 调用父方法init方法初始化
        super(Down2d, self).__init__()
        # nn.Conv2d用于构建二维卷积，针对图像之类的数据，对宽度和高度都进行卷积
        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        # InstanceNorm2dinstanceNorm在图像像素上，对HW做归一化，归一化层
        # 即是对batch中的单个样本的每一层特征图抽出来一层层求mean和variance，与batch size无关
        # 若特征层为1，即C=1，准则instance norm的值为输入本身
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        # 对输入进行两次卷积与归一化
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x2 = self.c2(x)
        x2 = self.n2(x2)
        # torch.sigmoid为sigmoid函数转换，激活
        x3 = x1 * torch.sigmoid(x2)
        return x3


# 上采样网络
class Up2d(nn.Module):
    """将升维为二维方法的说明"""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        # ConvTranspose2d方法为逆卷积，即合并特征扩充维数，为Conv2d方法的逆操作
        self.c1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    # 再进行逆卷积与归一化操作，并利用sigmoid函数转换激活
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x2 = self.c2(x)
        x2 = self.n2(x2)
        x3 = x1 * torch.sigmoid(x2)
        return x3


# 生成器模型
class Generator(nn.Module):
    """生成器的说明"""
    def __init__(self):
        super(Generator, self).__init__()
        # 定义一个降维样本
        # nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
        # 类似于nn.Module，但是这个容器构造神经网络模型会更简化
        self.downsample = nn.Sequential(
            # 利用自定义方法将元素全部规范化为二维
            Down2d(1, 32, (3, 9), (1, 1), (1, 4)),
            Down2d(32, 64, (4, 8), (2, 2), (1, 3)),
            Down2d(64, 128, (4, 8), (2, 2), (1, 3)),
            Down2d(128, 64, (3, 5), (1, 1), (1, 2)),
            Down2d(64, 5, (9, 5), (9, 1), (1, 2))
        )
        # 再定义四个升维样本
        self.up1 = Up2d(9, 64, (9, 5), (9, 1), (0, 2))
        self.up2 = Up2d(68, 128, (3, 5), (1, 1), (1, 2))
        self.up3 = Up2d(132, 64, (4, 8), (2, 2), (1, 3))
        self.up4 = Up2d(68, 32, (4, 8), (2, 2), (1, 3))
        # 再定义一些反卷积二维样本
        self.deconv = nn.ConvTranspose2d(36, 1, (3, 9), (1, 1), (1, 4))

    def forward(self, x, c):
        # 将x传入downsample方法将x加入对应的Sequential容器
        x = self.downsample(x)
        # .view方法返回具有相同数据但大小不同的新张量，即改变原有数据的维度上的大小
        # 这里改成四维数组，第一维为原第一维大小，第二维为原第二维大小，第三四维全部为1
        c = c.view(c.size(0), c.size(1), 1, 1)

        # tensor.repeat在张量的某个维度上复制
        # 如果参数与源数据相同维数就直接对应相乘，如果不同就进行平铺操作，从高维处增加多个新的维数并相乘，默认为1
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        # torch.cat将不同张量连接在一起，dim表示以哪个维度连接，dim=0, 横向连接，dim=1,纵向连接
        x = torch.cat([x, c1], dim=1)
        # 将x进行赋值给对象
        x = self.up1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.up4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.deconv(x)
        return x


# 判别器模型
class Discriminator(nn.Module):
    """判别器的说明"""

    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义对应降维样本
        self.d1 = Down2d(5, 32, (3, 9), (1, 1), (1, 4))
        self.d2 = Down2d(36, 32, (3, 8), (1, 2), (1, 3))
        self.d3 = Down2d(36, 32, (3, 8), (1, 2), (1, 3))
        self.d4 = Down2d(36, 32, (3, 6), (1, 2), (1, 2))
        # 定义对象二维卷积样本
        self.conv = nn.Conv2d(36, 1, (36, 5), (36, 1), (0, 2))
        # nn.AvgPool2d为二维平均池化函数
        self.pool = nn.AvgPool2d((1, 64))

    def forward(self, x, c):
        # 将c改变形状
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.d3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.d4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv(x)

        x = self.pool(x)
        # 再利用torch.squeeze将x中所有该维数数据大小为1的维数全部去掉
        x = torch.squeeze(x)
        # torch.tanh函数是将输入利用tanh函数进行变换
        x = torch.tanh(x)
        return x


# 域分类器模型
class DomainClassifier(nn.Module):
    """域分类器说明"""

    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.main = nn.Sequential(
            Down2d(1, 8, (4, 4), (2, 2), (5, 1)),
            Down2d(8, 16, (4, 4), (2, 2), (1, 1)),
            Down2d(16, 32, (4, 4), (2, 2), (0, 1)),
            Down2d(32, 16, (3, 4), (1, 2), (1, 1)),
            nn.Conv2d(16, 4, (1, 4), (1, 2), (0, 1)),
            nn.AvgPool2d((1, 16)),
            # nn.LogSoftmax()为对数的softmax函数，softmax输出都是0-1之间的，因此logsofmax输出的是小于0的数
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 取x的第一二四维的源数据，第三维的0到8的数据
        x = x[:, :, 0:8, :]
        # 调用域分类器的main方法生成对应数据
        x = self.main(x)
        x = x.view(x.size(0), x.size(1))
        return x


if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # train_loader = data_loader('data/processed', 1)
    # data_iter = iter(train_loader)
    # torch.rand返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
    t = torch.rand([1, 1, 36, 512])
    # 将对应参数转为FloatTensor格式
    l = torch.FloatTensor([[0, 1, 0, 0]])
    # 打印对应的大小
    print(l.size())
    # d1 = Down2d(1, 32, 1,1,1)
    # print(d1(t).shape)

    # u1 = Up2d(1,32,1,2,1)
    # print(u1(t).shape)
    # G = Generator()
    # o1 = G(t, l)
    # print(o1.shape)
    # 生成判别器
    D = Discriminator()
    # 将两个张量传入判别器
    o2 = D(t, l)
    print(o2.shape, o2)

    # C = DomainClassifier()
    # o3 = C(t)
    # print(o3.shape)
    # m = nn.Softmax()
    # input = torch.Tensor([[1,2],[5,5]])
    # output = m(input)
    # print(output)
