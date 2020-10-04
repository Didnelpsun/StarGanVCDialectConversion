# 这个文件主要负责的是预处理数据

# shutil提供对文件以及文件集合的高级操作
import shutil

# pyworld是一个操作WORLD开源软件的包，主要进行语音操作与合成
import pyworld

# 导入自定义的utility包
from utility import *

# argparse是python专门的命令行操作包
import argparse

# datetime为时间日期处理模块
from datetime import datetime

# 定义常量
# 特征维数
FEATURE_DIM = 36
# 采样率
SAMPLE_RATE = 16000
# 采样点数
FRAMES = 512
# 傅里叶变换大小
FFTSIZE = 1024
# 发音者个数，这里直接对utility导入的speakers值求长度
SPEAKERS_NUM = len(speakers)
# 数据块大小
CHUNK_SIZE = 1  # 将块大小的音频剪辑合并在一起
# 定义ε参数
EPSILON = 1e-10
MODEL_NAME = 'starganvc_model'


# 加载wav格式音频文件
def load_wavs(dataset: str, sr):
    """
    数据字典中包含所有音频文件路由
    特别是包含所有wav格式的文件
    """
    data = {}
    # os.scandir获取当前工作文件夹内的文件夹或文件
    # 扫描数据集对应路由，并将扫描结果取别名为it
    with os.scandir(dataset) as it:
        # 遍历扫描数组结果
        for entry in it:
            # is_dir()检查路由是否为目录
            if entry.is_dir():
                # 对应的文件夹名就是对应发音者的名字
                # 将文件夹名作为键名保存到字典中，并赋值一个空列表
                data[entry.name] = []
                # print(entry.name, entry.path)
                # 再扫描对应的数据集子文件夹路由，并取别名为it_f
                with os.scandir(entry.path) as it_f:
                    # 再次遍历
                    for onefile in it_f:
                        # 如果这个结果是文件
                        if onefile.is_file():
                            # print(onefile.path)
                            # 那么将这个文件路由保存到对应键名的字典的数组中
                            data[entry.name].append(onefile.path)
    # 打印对应键名
    print(f'loaded keys: {data.keys()}')
    # data like {TM1:[xx,xx,xxx,xxx]}
    resdict = {}
    cnt = 0
    # items()以列表返回可遍历的(键, 值) 元组数组
    for key, value in data.items():
        resdict[key] = {}
        # 将键名赋值给resdict字典，并赋值为空字典
        # 遍历字典的值
        for one_file in value:
            # 取得文件名
            filename = one_file.split('/')[-1].split('.')[0]  # like 100061
            newkey = f'{filename}'
            # librosa.load方法f为文件地址，sr参数为采样率
            # dtype为精度，将返回一个音频时间序列和一个音频采样率
            # 因为音频采样率不被使用，所以以_作为变量名占位保存
            # mono=True将信号转换为单声道
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)
            # librosa.effects.trim从音频信号中消除前导和尾随的静音
            # top_db参数为低于参考值的阈值（分贝）视为静音
            # 返回值为两个，第一个是被切割的信号数据，格式为np.ndarray，形状为(m,)或者(2, m)
            y, _ = librosa.effects.trim(wav, top_db=15)
            wav = np.append(y[0], y[1:] - 0.97 * y[:-1])

            resdict[key][newkey] = wav
            # resdict[key].append(temp_dict) #like TM1:{100062:[xxxxx], .... }
            print('.', end='')
            cnt += 1

    print(f'\nTotal {cnt} aduio files!')
    return resdict


def chunks(iterable, size):
    """从迭代器中生成连续的n大小的数据块"""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def wav_to_mcep_file(dataset: str, sr=SAMPLE_RATE, processed_filepath: str = './data/processed'):
    '''用生成器将wavs转换带有特征的数据'''
    shutil.rmtree(processed_filepath)
    os.makedirs(processed_filepath, exist_ok=True)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'Total {allwavs_cnt} audio files!')

    d = load_wavs(dataset, sr)
    for one_speaker in d.keys():
        values_of_one_speaker = list(d[one_speaker].values())

        for index, one_chunk in enumerate(chunks(values_of_one_speaker, CHUNK_SIZE)):
            wav_concated = []  # 保存一个批次的wavs数据
            temp = one_chunk.copy()

            # 连接wavs数据
            for one in temp:
                wav_concated.extend(one)
            wav_concated = np.array(wav_concated)

            # 处理一个批次的wavs数据
            f0, ap, sp, coded_sp = cal_mcep(wav_concated, sr=sr, dim=FEATURE_DIM)
            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)
            np.savez(file_path_z, f0=f0, coded_sp=coded_sp)
            print(f'[保存]: {file_path_z}')

            # 拆分t0媒体文件
            for start_idx in range(0, coded_sp.shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = coded_sp[:, start_idx: start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{newname}_{start_idx}'
                    filePath = os.path.join(processed_filepath, temp_name)

                    np.save(filePath, one_audio_seg)
                    print(f'[保存]: {filePath}.npy')


def world_features(wav, sr, fft_size, dim):
    f0, timeaxis = pyworld.harvest(wav, sr)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, sr, fft_size=fft_size)
    ap = pyworld.d4c(wav, f0, timeaxis, sr, fft_size=fft_size)
    coded_sp = pyworld.code_spectral_envelope(sp, sr, dim)

    return f0, timeaxis, sp, ap, coded_sp


def cal_mcep(wav, sr=SAMPLE_RATE, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''cal mecp给出wav信号
    帧周期只被用于pad_wav_to_get_fixed_frames
    '''
    f0, timeaxis, sp, ap, coded_sp = world_features(wav, sr, fft_size, dim)
    coded_sp = coded_sp.T  # dim x n

    return f0, ap, sp, coded_sp


if __name__ == "__main__":
    # 首先使用datetime.now()获取当前的时间
    start = datetime.now()
    # 定义一个命令行对象，并添加这个命令行对象的描述
    parser = argparse.ArgumentParser(description='将wav格式文件波形转换为梅尔倒谱系数（MCCs），并计算语音统计特征')
    # 定义输入数据的目录
    input_dir = './data/speakers'
    # 定义处理后的输出的数据目录
    output_dir = './data/processed'
    # 如果需要可以定义输入和输出目录
    parser.add_argument('--input_dir', type=str, help='这个目录包含需要处理的数据',
                        default=input_dir)
    parser.add_argument('--output_dir', type=str, help='这个目录存储处理过的数据', default=output_dir)
    # 获取对应的输入输出参数
    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir
    # 新建输出目录，如果存在不会报错
    os.makedirs(output_dir, exist_ok=True)

    wav_to_mcep_file(input_dir, SAMPLE_RATE, processed_filepath=output_dir)

    # 输入文件夹是训练数据，我们需要计算并保存对应的音频数据
    # 为每个发音者统计特征
    generator = GenerateStatistics(output_dir)
    generator.generate_stats()
    generator.normalize_dataset()
    end = datetime.now()
    print(f"[程序运行时间]: {end - start}")
