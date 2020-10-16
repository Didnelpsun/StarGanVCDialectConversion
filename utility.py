# 该文件为项目专门的工具类，主要是提供计算方程与解析格式化文件

# npy格式文件是numpy专用的保存数组的二进制文件，npz是对应的压缩文件
import numpy as np
# os库为标准文件处理库
import os

# glob为python标准库之一，可根据Unix终端所用规则找出所有匹配特定模式的路径名，但会按不确定的顺序返回结果，主要用于文件查找
import glob
# librosa主要用于语音处理
import librosa


# 设置单例类
class Singleton(type):
    # __init__为对象初始化方法
    # args参数为一个任意元素个数的元组，kwargs为一个任意元素个数的字典
    def __init__(self, *args, **kwargs):
        # 设置对象的__instance属性为空，这个属性代表对象的实例
        self.__instance = None
        super().__init__(*args, **kwargs)

    # __call__使实例能够像函数一样被调用，同时不影响实例本身的生命周期（__call__()不影响一个实例的构造和析构）。
    # 但是__call__()可以用来改变实例的内部成员的值。
    # 赋值方法
    def __call__(self, *args, **kwargs):
        # 如果对象的实例属性为空，则调用方法构造实例属性
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        # 否就直接返回而不会再新建一个实例
        else:
            return self.__instance


# 普通信息类，参数表明使用元类创建单例模式
class CommonInfo(metaclass=Singleton):
    """一般信息的的说明"""

    # 类初始化方法，参数为数据文件路径
    def __init__(self, datadir: str):
        # super的第一个参数为本身类名，第二个参数为self
        # 调用super父类的初始化方法
        super(CommonInfo, self).__init__()
        self.datadir = datadir

    # @property装饰器就是负责把一个方法变成属性调用
    @property
    # 获取发音者标签组方法
    def speakers(self):
        """ 返回发音者组作为训练源
        如 ['SF2', 'TM1', 'SF1', 'TM2']
        """
        # os.path.join连接两个或更多的路径名组件，如果各组件名首字母不包含’/’，则函数会自动加上
        # 如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
        # 如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾
        # 这里会将数据路径后加上*
        p = os.path.join(self.datadir, "*")

        # glob.glob匹配所有的符合条件的文件，并将其以list的形式返回
        # 调用glob.glob方法查询该目录下的所有文件或文件夹，*代表匹配全部
        all_sub_folder = glob.glob(p)

        # 使用rsplit方法将获取的文件集通过指定分隔符从最后开始对字符串进行分割并返回一个列表
        # maxsplit=1为仅切分一次，并取第二个部分
        # 如../marry/v1文件，就被分为../marry/和v1两个部分，[1]就是取v1这个部分
        # 使用列表生成式保存为列表，提取所有的文件夹的名字作为发音者标签
        all_speaker = [s.rsplit('/', maxsplit=1)[1] for s in all_sub_folder]
        # 将标签组重新排序
        all_speaker.sort()
        return all_speaker


# 调用CommonInfo构造方法将数据地址传入，以属性方式调用speakers方法获取对应标签组
speakers = CommonInfo('data/speakers').speakers


# 个性归一化准化方法，包含多个归一化操作
class Normalizer(object):
    """获取归一化的方法"""
    # 默认的归一化处理对象，路由为etc文件夹
    def __init__(self, statfolderpath: str = './etc'):
        # 定义文件集初始路由，默认为etc文件夹
        self.folderpath = statfolderpath
        # 将自定义的normalizer_dict方法所返回的数据字典作为属性赋值给类
        self.norm_dict = self.normalizer_dict()

    # 向前处理，下面的两个函数都是为了处理数据的相关处理方法
    def forward_process(self, x, speakername):
        # coded_sps_mean为编码平均值，coded_sps_std为编码标准差
        # 获取对应发音者键名的对应值
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        # np.reshape方法将数组根据[行数,列数]的参数变为不同形状矩阵，而这里的-1表示是不知道行数多少
        # 所以这样就可以将mean和std数据同时变为列向量形式的特征向量
        mean = np.reshape(mean, [-1, 1])
        std = np.reshape(std, [-1, 1])
        # 将输入值先减去平均值再除以标准差
        x = (x - mean) / std
        return x

    # 向后处理
    def backward_process(self, x, speakername):
        # 取出对应键名的数据值
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1, 1])
        std = np.reshape(std, [-1, 1])
        # 将输入值先乘上标准差再加上平均值
        x = x * std + mean
        return x

    # 返回一个发音者名对应发音者音频数据的字典
    def normalizer_dict(self):
        """返回所有发音者的归一化后参数"""
        d = {}
        # 迭代处理每个发声者数据，首先遍历speakers类中的所有发音者名
        for one_speaker in speakers:
            # 将数据路由后加上*.npz表示取出该目录默认etc下所有的npz类型文件
            p = os.path.join(self.folderpath, '*.npz')
            try:
                # 根据路径取出对应的文件，如果遍历的子对象在对应的文件集中，就再取出作为文件路由，因为只有一个数据，所以取[0]
                stat_filepath = [fn for fn in glob.glob(p) if one_speaker in fn][0]
            except:
                # 如果没有找到对应的文件就抛出异常
                # 这里就是判断对应的标签etc文件夹下是否有对应的npz文件
                raise Exception('====找不到对应的文件！====')
            # 找到路由对应文件就加载该文件
            t = np.load(stat_filepath)
            # 并以对象的发音者名:处理过的etc数据文件作为存储方式来存储到字典中
            d[one_speaker] = t
        return d

    # 音频转换函数，采用对数方式
    def pitch_conversion(self, f0, source_speaker, target_speaker):
        """基音转换的对数高斯归一化"""

        # 分别取出源发音者与目标发音者的数据集平均值和标准差
        mean_log_src = self.norm_dict[source_speaker]['log_f0s_mean']
        std_log_src = self.norm_dict[source_speaker]['log_f0s_std']

        mean_log_target = self.norm_dict[target_speaker]['log_f0s_mean']
        std_log_target = self.norm_dict[target_speaker]['log_f0s_std']
        # np.exp为e的幂次方，numpy.ma支持数值数组中包含掩码元素，用于处理错误和异常数据

        # 掩码数组是将标准的多维数组numpy.ndarray和掩码相结合。掩码要么是nomask，表示与该掩码有关数组的所有值都是有效的。
        # 要么是一个布尔值数组，用于确定关联数组的每个元素值是否有效。
        # 当掩码中某个元素值为False，那么关联数组的对应元素是有效的，即被认为是未掩码的。
        # 当掩码中某个元素值为True，那么关联数组的对应元素是无效的，即被认为是掩码的

        # exp为幂方法，log就是取对数方法
        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)
        # 返回转换结果
        return f0_converted


# 生成发音者对应npz格式数据字典
class GenerateStatistics(object):
    # 初始化对象，一个参数为folder，表示处理目录
    def __init__(self, folder: str = './data/processed'):
        self.folder = folder
        # 定义保存npz数据文件的字典
        self.include_dict_npz = {}
        # 遍历全局定义的speakers数据遍历，遍历值为所有发音者名字
        for s in speakers:
            # __contains__用于判断某键是否存在该字典中
            # 如果s这个发音者名字不在定义的字典中，就将键加入字典，并赋值为空
            if not self.include_dict_npz.__contains__(s):
                self.include_dict_npz[s] = []
            # os.listdir用于返回指定的文件夹包含的文件或文件夹的名字的列表
            # 循环遍历folder路由下所有文件，这里包含处理过的npz和npy格式文件
            for one_file in os.listdir(folder):
                # startswith用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False，endswith同理
                # 如果这个文件是以发音者名开始，并以npz结尾（即该文件是npz格式文件）
                if one_file.startswith(s) and one_file.endswith('npz'):
                    # 就将这个npz格式文件放入对应发音者键值的字典中
                    self.include_dict_npz[s].append(one_file)

    # @staticmethod注释表示该方法为静态方法，虽然名义由该模块管理，但是可以无视self，即对象，自由调用该方法
    # 与其他编程语言的静态方法一致无需实例化，所以下面的参数中没有带self参数
    @staticmethod
    # 参数为编码频谱包络组，获取参数的平均值与标准差
    def coded_sp_statistics(coded_sps):
        # 包络的形状为（特征维数，采样点数）
        # np.concatenate表示按参数axis为轴合并为一个新数组，axis表示拼接维数
        # 这里表示在第二维，即行方向合并coded_sps数组，即源数据同一列的数组会合并为一个数组
        # 这样会取得同一个发音者所有音频数据基频的对应合并数据组
        coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
        # np.mean按计算axis参数维上的平均值，keepdims=False即不保持维数，可能会降维
        # 这里计算数组行方向上即每一列数据组的每一个平均值，并返回一个一维数组
        coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=False)
        # 与上面方法类似，np.std计算axis参数维上的标准差，在这里计算行方向的标准差，并不保持维数
        coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=False)
        print(coded_sps_mean, coded_sps_std)
        return coded_sps_mean, coded_sps_std

    @staticmethod
    # 参数为基频数据组
    def logf0_statistics(f0s):
        # 按照默认参数0，对f0s数组按列方向合并，即每一行的数据合并为一个数据组
        # 在对结果进行掩码化与对数取值
        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        # 分别取其平均值与标准差，不设置axis会只返回一个实数
        log_f0s_mean = log_f0s_concatenated.mean()
        log_f0s_std = log_f0s_concatenated.std()
        print(log_f0s_mean, log_f0s_std)
        return log_f0s_mean, log_f0s_std

    # 建立一个etc文件夹，用来对每一个发音者的基频数据组与编码频谱包络组进行求平均值和标准差操作并保存为文件
    def generate_stats(self, statfolder: str = 'etc'):
        """
        生成归一化的对于所有用户使用过的统计数据
        输入sp与f0等类似数据
        第一步，生成编码的平均标准差
        第二步，生成f0即生成数据的平均标准差
         """
        # os.path.realpath方法用于返回该脚本的绝对路径
        # 将绝对路径与一个文件夹名路径连接，这个文件夹用来存放一些其他数据
        etc_path = os.path.join(os.path.realpath('.'), statfolder)
        # os.makedirs用于创建目录文件夹，这个方法如果父文件集都不存在则会一同创建
        # exist_ok=True表示当要创建的文件夹已经存在时不会抛出OSError错误
        # 这里会在该项目一级目录下创建一个etc文件夹与data文件夹同级
        print("创建或进入etc文件夹")
        os.makedirs(etc_path, exist_ok=True)
        # .keys()以列表的形式返回一个字典的所有键名
        # 遍历所有的保存npz格式文件的字典include_dict_npz属性的键名
        for one_speaker in self.include_dict_npz.keys():
            # 定义两个变量，都是用来保存一个发音者的所有f0与coded_sps数据
            f0s = []
            coded_sps = []
            # 将键名为onespeaker的npz数据字典设置为arr01中
            arr01 = self.include_dict_npz[one_speaker]
            # 如果这个数据为空数据，即对应键名的npz数据字典中没有数据无法处理，就跳出这个循环进行下一次遍历
            if len(arr01) == 0:
                continue
            # 如果含有数据遍历这个npz数据中的所有数据项
            for one_file in arr01:
                # np.load用来加载npy与npz格式文件
                # 将文件夹processed的路由与这个文件的路由名合并并加载文件
                t = np.load(os.path.join(self.folder, one_file))
                # npz文件中包含着一个音频数据的f0与coded_sp
                # 将这个文件的f0项转换形状，即生成样本项转换为特征列向量
                f0_ = np.reshape(t['f0'], [-1, 1])
                # 把这个生成样本的特征列向量放到列向量组中
                f0s.append(f0_)
                # 并把npz数据文件的coded_sp编码项也放入对应数组中
                coded_sps.append(t['coded_sp'])
            # 调用该类原有方法得到对应的基频组与编码频谱包络组的平均值与标准差
            log_f0s_mean, log_f0s_std = self.logf0_statistics(f0s)
            coded_sps_mean, coded_sps_std = self.coded_sp_statistics(coded_sps)
            # 打印对应数据
            print(f'[{one_speaker}] 对数化基频组平均值为：{log_f0s_mean}，对数化基频组标准差值为：{log_f0s_std}', end='，')
            print(f'编码频谱包络组平均值形状为：{coded_sps_mean.shape}，编码频谱包络组标准差形状为：{coded_sps_std.shape}')
            # 将文件名命名为etc文件夹名/one_speaker变量值-stats.npz
            filename = os.path.join(etc_path, f'{one_speaker}-stats.npz')
            # np.savez保存数组到一个二进制的文件中，保存多个数组到同一个文件中,保存格式是.npz,
            # 其实就是多个前面np.save的保存的npy，再通过打包(未压缩)的方式把这些文件归到一个文件上
            np.savez(filename,
                     log_f0s_mean=log_f0s_mean, log_f0s_std=log_f0s_std,
                     coded_sps_mean=coded_sps_mean, coded_sps_std=coded_sps_std)
            print(f'[保存处理过的基频与包络文件]: {filename}.npz')

    # 对数据集进行归一化操作
    def normalize_dataset(self):
        """运行一次归一化数据集"""
        # 定义一个归一化处理类
        norm = Normalizer()
        # librosa.util.find_files获取目录或目录子树中（音频）文件的排序列表
        # 第一个参数为路径，ext表示要包含在搜索中的文件扩展名或文件扩展名列表
        # 在processed文件夹寻找npy数据集文件
        files = librosa.util.find_files(self.folder, ext='npy')
        # 遍历文件
        print("进入processed文件夹")
        for p in files:
            # os.path.basename截取文件名
            filename = os.path.basename(p)
            # 根据文件命名方式获取speaker名
            speaker = filename.split(sep='_', maxsplit=1)[0]
            # 定义数据为梅尔倒谱系数
            mcep = np.load(p)
            # 调用自定义方法对于mcep文件进行计算，从etc文件中取出对应的数据对原有processed文件夹的npy文件进行计算
            mcep_normed = norm.forward_process(mcep, speaker)
            # os.remove用于删除指定文件，即将原来的npy文件删掉
            os.remove(p)
            # 将格式化好的数据放入原来路径中
            np.save(p, mcep_normed)
            print(f'[归一化频谱包络]：{p}.npy')


# 当调用形式为调用main，就跳过这个文件
if __name__ == "__main__":
    pass
