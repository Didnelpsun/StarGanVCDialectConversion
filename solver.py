# 该文件调用模型进行训练测试等操作
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from data_loader import TestSet
from model import Discriminator, DomainClassifier, Generator
from utility import Normalizer, speakers
from preprocess import FRAMES, SAMPLE_RATE, FFTSIZE
import random
from sklearn.preprocessing import LabelBinarizer
from pyworld import decode_spectral_envelope, synthesize
import librosa
import ast


class Solver(object):
    """训练操作的说明"""
    #  参数为数据加载器与配置
    def __init__(self, data_loader, config):
        # 进行赋值
        self.config = config
        self.data_loader = data_loader
        # 模型配置
        # 赋值三个损失函数，循环损失，域分类损失，身份映射损失
        self.lambda_cycle = config.lambda_cycle
        self.lambda_cls = config.lambda_cls
        self.lambda_identity = config.lambda_identity

        # 训练配置
        # 数据文件路由
        self.data_dir = config.data_dir
        # 测试文件路由
        self.test_dir = config.test_dir
        # 批处理大小
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        

        # 测试配置
        self.test_iters = config.test_iters
        # ast.literal_eval为解析函数，并安全地进行类型转换
        # 目标发音者
        self.trg_speaker = ast.literal_eval(config.trg_speaker)
        # 源发音者
        self.src_speaker = config.src_speaker

        # 其他配置
        # 是否使用tensorboard记录
        self.use_tensorboard = config.use_tensorboard
        # torch.device代表将torch.Tensor分配到的设备的对象。torch.device包含一个设备类型（‘cpu’或‘cuda’）和可选的设备序号。
        # 是使用cuda还是cpu计算
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 将speakers标签二值化
        self.spk_enc = LabelBinarizer().fit(speakers)
        # 字典
        # 记录字典
        self.log_dir = config.log_dir
        # 样本字典
        self.sample_dir = config.sample_dir
        # 模型字典
        self.model_save_dir = config.model_save_dir
        # 输出字典
        self.result_dir = config.result_dir

        # 步长
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # 建立模型与tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
    # 赋值三个模型器
    def build_model(self):
        self.G = Generator()
        self.D = Discriminator()
        self.C = DomainClassifier()
        # torch.optim.Adam用于实现Adam算法

        # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
        # 它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。

        # params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
        # lr (float, 可选) – 学习率（默认：1e-3）
        # 同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
        # betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）betas = （beta1，beta2）
        # beta1：一阶矩估计的指数衰减率（如 0.9）。
        # beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
        # eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。
        # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
 
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr,[self.beta1, self.beta2])
        
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.C, 'C')
        # Module.to方法用来移动和转换参数与缓冲区，类似于torch.Tensor.to，但是仅支持float类型
        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)
    
    def print_network(self, model, name):
        """打印出网络的相关信息"""
        num_params = 0
        # Module.parameters()获取网络的参数
        # 计算模型网络对应的参数频次
        for p in model.parameters():
            # numel返回数组中元素的个数
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
    # 使用tensorboard
    def build_tensorboard(self):
        """建立一个tensorboard记录器"""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):
        """生成器、判别器和域分类器的衰减学习率"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr

    def train(self):
        # 衰减的学习率缓存
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr

        start_iters = 0
        # 如果存在就跳过
        if self.resume_iters:
            pass
        # 调用定义的个性化标准化方法
        norm = Normalizer()
        # iter用来生成迭代器，这里用来迭代加载数据集
        data_iter = iter(self.data_loader)
        print('Start training......')
        # 记录当前时间，now函数取当前时间
        start_time = datetime.now()

        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                                 1.预处理输入数据                                    #
            # =================================================================================== #
            # 获取真实的图像和对应标签标签
            try:
                # next方法为迭代下一个迭代器
                # 利用自定义的加载器获取真实x值，发音者数据与源标签
                x_real, speaker_idx_org, label_org = next(data_iter)
            except:
                # 如果迭代器有问题就再转换为迭代器一次然后迭代
                data_iter = iter(self.data_loader)
                x_real, speaker_idx_org, label_org = next(data_iter)           

            # 随机生成目标域标签
            # torch.randperm返回一个从0到参数-1范围的随机数组
            rand_idx = torch.randperm(label_org.size(0))
            # 根据随机数组作为源标签的索引，打乱标签数组作为目标标签数组
            label_trg = label_org[rand_idx]
            # 同理得到随机发音者数组
            speaker_idx_trg = speaker_idx_org[rand_idx]
            
            x_real = x_real.to(self.device)           # 输入数据
            label_org = label_org.to(self.device)     # 源域one-hot格式标签
            label_trg = label_trg.to(self.device)     # 目标域ont-hot格式标签
            speaker_idx_org = speaker_idx_org.to(self.device) # 源域标签
            speaker_idx_trg = speaker_idx_trg.to(self.device) # 目标域标签

            # =================================================================================== #
            #                                      2.训练判别器                                   #
            # =================================================================================== #
            # 用真实音频数据计算损失
            # nn.CrossEntropyLoss()为交叉熵损失函数，但是不是普通的形式，而是主要是将softmax-log-NLLLoss合并到一块得到的结果。
            CELoss = nn.CrossEntropyLoss()
            # 调用分类器计算真实数据
            cls_real = self.C(x_real)
            # 计算对应的域分类损失，即用交叉熵实现
            cls_loss_real = CELoss(input=cls_real, target=speaker_idx_org)
            # 重置缓冲区，具体实现在下面
            self.reset_grad()
            # tensor.backward为自动求导函数
            cls_loss_real.backward()
            # optimizer.step这个方法会更新模型所有的参数以提升学习率，一般在backward函数后根据其计算的梯度来更新参数
            self.c_optimizer.step()
             # 记录中
            loss = {}
            # 从真实域分类损失张量中获取元素值
            # item()得到一个元素张量里面的元素值
            loss['C/C_loss'] = cls_loss_real.item()
            # 根据源真实数据与源标签来训练判别器
            out_r = self.D(x_real, label_org)
            # 用假音频帧计算损失
            # 根据真实样本与目标标签生成生成样本
            x_fake = self.G(x_real, label_trg)
            # detach截断反向传播的梯度流，从而让梯度不影响判别器D
            out_f = self.D(x_fake.detach(), label_trg)
            # torch.nn.Function.binary_cross_entropy_with_logits
            # 接受任意形状的输入，target要求与输入形状一致。切记：target的值必须在[0,N-1]之间，其中N为类别数，否则会出现莫名其妙的错误，比如loss为负数。
            # 计算其实就是交叉熵，不过输入不要求在0，1之间，该函数会自动添加sigmoid运算
            # 返回一个填充了标量值1的张量，其大小与输入相同。torch.ones_like(input)
            # 相当于torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)
            d_loss_t = F.binary_cross_entropy_with_logits(input=out_f,target=torch.zeros_like(out_f, dtype=torch.float)) + \ 
                F.binary_cross_entropy_with_logits(input=out_r, target=torch.ones_like(out_r, dtype=torch.float))
            # 将生成样本输入域分类器中
            out_cls = self.C(x_fake)
            # 交叉熵计算D的域分类损失
            d_loss_cls = CELoss(input=out_cls, target=speaker_idx_trg)

            # 计算梯度惩罚的损失
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # 计算x_hat
            # requires_grad_设置积分方法，将requires_grad是否积分的属性设置为真
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            # 得到输出源
            out_src = self.D(x_hat, label_trg)
            # 调用自定义方法得到损失
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            # 计算判别器的总体损失
            d_loss = d_loss_t + self.lambda_cls * d_loss_cls + 5*d_loss_gp
            # 调用自定义方法重置梯度变化缓冲区
            self.reset_grad()
            # 对D的损失求导
            d_loss.backward()
            # 更新模型判别器D参数
            self.d_optimizer.step()


            # loss['D/d_loss_t'] = d_loss_t.item()
            # loss['D/loss_cls'] = d_loss_cls.item()
            # loss['D/D_gp'] = d_loss_gp.item()
            # 获取判别器损失
            loss['D/D_loss'] = d_loss.item()

            # =================================================================================== #
            #                                       3.训练生成器                                  #
            # =================================================================================== #        
            # 进行模运算
            if (i+1) % self.n_critic == 0:
                # 源至目标域
                # 利用真实样本和目标标签生成生成样本
                x_fake = self.G(x_real, label_trg)
                #  判别生成样本与目标标签
                g_out_src = self.D(x_fake, label_trg)
                # 将生成与目标标签的损失与相同大小纯1张量计算交叉熵得到虚假损失
                g_loss_fake = F.binary_cross_entropy_with_logits(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))
                # 得到真实样本通过域分类器得到的类别
                out_cls = self.C(x_real)
                # 计算C计算结果与输入的类别的损失
                g_loss_cls = CELoss(input=out_cls, target=speaker_idx_org)

                # 目标至源域
                # 通过G将生成样本转换为源标签
                x_reconst = self.G(x_fake, label_org)
                # 得到循环一致性损失，即通过G转回来的损失，按道理这两个是同样的
                # l1_loss为L1损失函数，即平均绝对误差
                g_loss_rec = F.l1_loss(x_reconst, x_real )

                # 源到源域(身份一致性损失).
                # 通过真实样本与源标签生成，按道理也是生成x_real
                x_fake_iden = self.G(x_real, label_org)
                # 利用L1损失函数计算
                id_loss = F.l1_loss(x_fake_iden, x_real )

                # 后退和优化
                # 得到生成器的总体损失函数
                g_loss = g_loss_fake + self.lambda_cycle * g_loss_rec +\
                 self.lambda_cls * g_loss_cls + self.lambda_identity * id_loss
                # 重置梯度变化缓冲区
                self.reset_grad()
                # 对G损失求导
                g_loss.backward()
                # 更新生成器参数
                self.g_optimizer.step()

                # 记录
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_id'] = id_loss.item()
                loss['G/g_loss'] = g_loss.item()
            # =================================================================================== #
            #                                           4.其他                                    #
            # =================================================================================== #
            # 打印训练相关信息
            if (i+1) % self.log_step == 0:
                # 得到训练时间
                et = datetime.now() - start_time
                # 截取后面的时间段
                et = str(et)[:-7]
                # 耗时与迭代次数
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                # 打印对应损失值
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                # 如果调用tensorboard来记录训练过程
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        # 添加到log中
                        self.logger.scalar_summary(tag, value, i+1)

            # 翻译固定数据进行调试
            if (i+1) % self.sample_step == 0:
                # torch.no_grad是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
                # 所有依赖他的tensor会全部变成True，反向传播时就不会自动求导了，反向传播就不会保存梯度，因此大大节约了显存或者说内存。
                with torch.no_grad():
                    # 调用自定义方法，定义一个路由，并随机选取一个发音者作为测试数据
                    d, speaker = TestSet(self.test_dir).test_data()
                    # random.choice返回参数的随机项
                    # 随机在speakers中选择一个不是目标的发音者
                    target = random.choice([x for x in speakers if x != speaker])
                    # 将二值化的标签组取出第一个作为目标
                    # LabelBinary.transfrom方法将复杂类标签转换为二进制标签
                    label_t = self.spk_enc.transform([target])[0]
                    # np.asarray将python原生列表或元组形式的现有数据来创建numpy数组
                    label_t = np.asarray([label_t])
                    # 取出字典中的文件名与内容
                    for filename, content in d.items():
                        f0 = content['f0']
                        ap = content['ap']
                        # 调用自定义方法处理对应的数据
                        sp_norm_pad = self.pad_coded_sp(content['coded_sp_norm'])
                        
                        convert_result = []
                        for start_idx in range(0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                            one_seg = sp_norm_pad[:, start_idx : start_idx+FRAMES]
                            
                            one_seg = torch.FloatTensor(one_seg).to(self.device)
                            one_seg = one_seg.view(1,1,one_seg.size(0), one_seg.size(1))
                            l = torch.FloatTensor(label_t)
                            one_seg = one_seg.to(self.device)
                            l = l.to(self.device)
                            one_set_return = self.G(one_seg, l).data.cpu().numpy()
                            one_set_return = np.squeeze(one_set_return)
                            one_set_return = norm.backward_process(one_set_return, target)
                            convert_result.append(one_set_return)

                        convert_con = np.concatenate(convert_result, axis=1)
                        convert_con = convert_con[:, 0:content['coded_sp_norm'].shape[1]]
                        contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)   
                        decoded_sp = decode_spectral_envelope(contigu, SAMPLE_RATE, fft_size=FFTSIZE)
                        f0_converted = norm.pitch_conversion(f0, speaker, target)
                        wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                        name = f'{speaker}-{target}_iter{i+1}_{filename}'
                        path = os.path.join(self.sample_dir, name)
                        print(f'[save]:{path}')
                        librosa.output.write_wav(path, wav, SAMPLE_RATE)
                        
            # 保存模型检查点
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.C.state_dict(), C_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # 衰减学习率
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, c_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def gradient_penalty(self, y, x):
        """计算梯度惩罚: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """重置梯度变化缓冲区"""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def restore_model(self, resume_iters):
        """重置训练好的发生器和鉴别器"""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))

    @staticmethod
    def pad_coded_sp(coded_sp_norm):
        f_len = coded_sp_norm.shape[1]
        if  f_len >= FRAMES: 
            pad_length = FRAMES-(f_len - (f_len//FRAMES) * FRAMES)
        elif f_len < FRAMES:
            pad_length = FRAMES - f_len

        sp_norm_pad = np.hstack((coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length))))
        return sp_norm_pad 

    def test(self):
        """用StarGAN处理音频数据"""
        # 加载训练生成器
        self.restore_model(self.test_iters)
        norm = Normalizer()

        # 设置数据加载器
        d, speaker = TestSet(self.test_dir).test_data(self.src_speaker)
        targets = self.trg_speaker
       
        for target in targets:
            print(target)
            assert target in speakers
            label_t = self.spk_enc.transform([target])[0]
            label_t = np.asarray([label_t])
            
            with torch.no_grad():

                for filename, content in d.items():
                    f0 = content['f0']
                    ap = content['ap']
                    sp_norm_pad = self.pad_coded_sp(content['coded_sp_norm'])

                    convert_result = []
                    for start_idx in range(0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                        one_seg = sp_norm_pad[:, start_idx : start_idx+FRAMES]
                        
                        one_seg = torch.FloatTensor(one_seg).to(self.device)
                        one_seg = one_seg.view(1,1,one_seg.size(0), one_seg.size(1))
                        l = torch.FloatTensor(label_t)
                        one_seg = one_seg.to(self.device)
                        l = l.to(self.device)
                        one_set_return = self.G(one_seg, l).data.cpu().numpy()
                        one_set_return = np.squeeze(one_set_return)
                        one_set_return = norm.backward_process(one_set_return, target)
                        convert_result.append(one_set_return)

                    convert_con = np.concatenate(convert_result, axis=1)
                    convert_con = convert_con[:, 0:content['coded_sp_norm'].shape[1]]
                    contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)   
                    decoded_sp = decode_spectral_envelope(contigu, SAMPLE_RATE, fft_size=FFTSIZE)
                    f0_converted = norm.pitch_conversion(f0, speaker, target)
                    wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                    name = f'{speaker}-{target}_iter{self.test_iters}_{filename}'
                    path = os.path.join(self.result_dir, name)
                    print(f'[save]:{path}')
                    librosa.output.write_wav(path, wav, SAMPLE_RATE)            


    
# 如果执行模式为main就跳过这个文件
if __name__ == '__main__':
    pass
