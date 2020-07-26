# 这个文件主要负责的是预处理数据
import librosa
import numpy as np
import os
# pyworld是一个操作WORLD开源软件的包，主要进行语音操作与合成
import pyworld
import pyworld as pw
import glob
from utility import *
# argparse是python专门的命令行操作包
import argparse
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
# 发音者个数
SPEAKERS_NUM = len(speakers)
# 数据块大小
CHUNK_SIZE = 1  # 将块大小的音频剪辑合并在一起
# 定义ε
EPSILON = 1e-10
MODEL_NAME = 'starganvc_model'


# 加载wav格式音频文件
def load_wavs(dataset: str, sr):
    """
    数据字典中包含所有音频文件路由
    特别是包含所有wav格式的文件
    """
    data = {}
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
            y, _ = librosa.effects.trim(wav, top_db=15)
            wav = np.append(y[0], y[1:] - 0.97 * y[:-1])

            resdict[key][newkey] = wav
            # resdict[key].append(temp_dict) #like TM1:{100062:[xxxxx], .... }
            print('.', end='')
            cnt += 1

    print(f'\nTotal {cnt} aduio files!')
    return resdict


def chunks(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def wav_to_mcep_file(dataset: str, sr=SAMPLE_RATE, processed_filepath: str = './data/processed'):
    '''convert wavs to mcep feature using image repr'''
    shutil.rmtree(processed_filepath)
    os.makedirs(processed_filepath, exist_ok=True)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'Total {allwavs_cnt} audio files!')

    d = load_wavs(dataset, sr)
    for one_speaker in d.keys():
        values_of_one_speaker = list(d[one_speaker].values())

        for index, one_chunk in enumerate(chunks(values_of_one_speaker, CHUNK_SIZE)):
            wav_concated = []  # preserve one batch of wavs
            temp = one_chunk.copy()

            # concate wavs
            for one in temp:
                wav_concated.extend(one)
            wav_concated = np.array(wav_concated)

            # process one batch of wavs
            f0, ap, sp, coded_sp = cal_mcep(wav_concated, sr=sr, dim=FEATURE_DIM)
            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)
            np.savez(file_path_z, f0=f0, coded_sp=coded_sp)
            print(f'[save]: {file_path_z}')

            # split mcep t0 muliti files
            for start_idx in range(0, coded_sp.shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = coded_sp[:, start_idx: start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{newname}_{start_idx}'
                    filePath = os.path.join(processed_filepath, temp_name)

                    np.save(filePath, one_audio_seg)
                    print(f'[save]: {filePath}.npy')


def world_features(wav, sr, fft_size, dim):
    f0, timeaxis = pyworld.harvest(wav, sr)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, sr, fft_size=fft_size)
    ap = pyworld.d4c(wav, f0, timeaxis, sr, fft_size=fft_size)
    coded_sp = pyworld.code_spectral_envelope(sp, sr, dim)

    return f0, timeaxis, sp, ap, coded_sp


def cal_mcep(wav, sr=SAMPLE_RATE, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''cal mcep given wav singnal
        the frame_period used only for pad_wav_to_get_fixed_frames
    '''
    f0, timeaxis, sp, ap, coded_sp = world_features(wav, sr, fft_size, dim)
    coded_sp = coded_sp.T  # dim x n

    return f0, ap, sp, coded_sp


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser(description='Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics')

    input_dir = './data/speakers'
    output_dir = './data/processed'

    parser.add_argument('--input_dir', type=str, help='the direcotry contains data need to be processed',
                        default=input_dir)
    parser.add_argument('--output_dir', type=str, help='the directory stores the processed data', default=output_dir)

    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir

    os.makedirs(output_dir, exist_ok=True)

    wav_to_mcep_file(input_dir, SAMPLE_RATE, processed_filepath=output_dir)

    # input_dir is train dataset. we need to calculate and save the speech\
    # statistical characteristics for each speaker.
    generator = GenerateStatistics(output_dir)
    generator.generate_stats()
    generator.normalize_dataset()
    end = datetime.now()
    print(f"[Runing Time]: {end - start}")
