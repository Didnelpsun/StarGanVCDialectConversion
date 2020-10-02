# 此文件用来下载对应的数据文件，主要是vcc2016和evaluation_all文件
# os库是Python标准库，包含几百个函数,常用路径操作、进程管理、环境参数等几类。os.path子库以path为入口，用于操作和处理文件路径。
import os
# import random
# urllib库

# urlretrieve() 方法直接将远程数据下载到本地。
# 这个函数可以方便的将网页上的一个文件保存到本地。文件类型可以是网页的html文件、图片、视频等媒体文件。
# 函数原型：urlretrieve(url, filename=None, reporthook=None, data=None)
# 参数 url 指定了要下载的文件的url
# 参数 finename 指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
# 参数 reporthook 是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
# 参数 data 指 post 到服务器的数据，该方法返回一个包含两个元素的(filename, headers)元组，filename 表示保存到本地的路径，header 表示服务器的响应头。
# from urllib.request import urlretrieve
import argparse
# shlex类使得为类似于unixshell的简单语法编写词法分析器变得很容易。
# 这对于编写小型语言（例如，Python应用程序的运行控制文件）或解析引用的字符串通常很有用。
# subprocess类用于子进程管理，生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。
import shlex, subprocess
# zipfile用于处理zip格式压缩文件
import zipfile


# 用于解压zip文件
def unzip(zip_filepath, dest_dir='./data'):
    # 用于读写zip类型的文件
    with zipfile.ZipFile(zip_filepath) as zf:
        # 从归档中提取出所有成员放入当前工作目录。参数为path=None, members=None, pwd=None
        # path 指定一个要提取到的不同目录。
        # members 为可选项且必须为 namelist() 所返回列表的一个子集。
        # pwd 是用于解密文件的密码。
        # 这里将zip_filepath路径的文件放入dest_dir路径中
        zf.extractall(dest_dir)
    print("Extraction complete!")


# 用于下载vcc2016数据集
def download_vcc2016():
    # 数据连接与数据文件
    datalink = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
    data_files = ['vcc2016_training.zip', 'evaluation_all.zip']
    # 如果对应的数据文件存在就不用下载
    if os.path.exists(data_files[0]) or os.path.exists(data_files[1]):
        print("File already exists!")
        return
    # 定义对应训练集与评估数据集
    trainset = f'{datalink}/{data_files[0]}'
    evalset = f'{datalink}/{data_files[1]}'

    train_comm = f'wget {trainset}'
    eval_comm = f'wget {evalset}'
    # 将字符串按照空格切分
    train_comm = shlex.split(train_comm)
    eval_comm = shlex.split(eval_comm)

    print('Start download dataset...')
    # 加入一个子进程开始下载文件
    subprocess.run(train_comm)
    subprocess.run(eval_comm)
    # 下载完成后解压zip格式文件
    unzip(data_files[0])
    unzip(data_files[1])
    print('Finish download dataset...')


def create_dirs(trainset: str = './data/fourspeakers', testset: str = './data/fourspeakers_test'):
    """创建测试与训练数据集"""
    if not os.path.exists(trainset):
        # 如果不存在该目录就创造目录
        print(f'create train set dir {trainset}')
        os.makedirs(trainset, exist_ok=True)

    if not os.path.exists(testset):
        print(f'create test set dir {testset}')
        os.makedirs(testset, exist_ok=True)


# 当以main方式运行时
if __name__ == '__main__':
    # 定义一个命令行操作对象，包含将命令行解析成Python数据类型所需的全部信息。

    # prog - 程序的名称（默认：sys.argv[0]）
    # usage - 描述程序用途的字符串（默认值：从添加到解析器的参数生成）
    # description - 在参数帮助文档之前显示的文本（默认值：无）
    # epilog - 在参数帮助文档之后显示的文本（默认值：无）
    # parents - 一个ArgumentParser对象的列表，他们的参数也应包含在内
    # formatter_class - 用于自定义帮助文档输出格式的类
    # prefix_chars - 可选参数的前缀字符集合（默认值：’-’）
    # fromfile_prefix_chars - 当需要从文件中读取其他参数时，用于标识文件名的前缀字符集合（默认值：None）
    # argument_default - 参数的全局默认值（默认值： None）
    # conflict_handler - 解决冲突选项的策略（通常是不必要的）
    # add_help - 为解析器添加一个 -h/–help 选项（默认值： True）
    # allow_abbrev - 如果缩写是无歧义的，则允许缩写长选项 （默认值：True）

    parser = argparse.ArgumentParser(description='Download  voice conversion datasets.')
    # 定义对应的数据集路由
    datasets_default = 'vcc2016'
    train_dir = './data/fourspeakers'
    test_dir = './data/fourspeakers_test'
    # 调用add方法给命令行对象增加命令参数，即输入对应命令就跳到不同指令处理程序

    # name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
    # action - 当参数在命令行中出现时使用的动作基本类型。
    # nargs - 命令行参数应当消耗的数目。
    # const - 被一些 action 和 nargs 选择所需求的常数。
    # default - 当参数未在命令行中出现时使用的值。
    # type - 命令行参数应当被转换成的类型。
    # choices - 可用的参数的容器。
    # required - 此命令行选项是否可省略 （仅选项可用）。
    # help - 一个此选项作用的简单描述。
    # metavar - 在使用方法消息中使用的参数值示例。
    # dest - 被添加到 parse_args() 所返回对象上的属性名。

    parser.add_argument('--datasets', type=str, help='Datasets available: vcc2016', default=datasets_default)
    parser.add_argument('--train_dir', type=str, help='trainset directory', default=train_dir)
    parser.add_argument('--test_dir', type=str, help='testset directory', default=test_dir)
    # 解析参数并把结果赋值给argv
    argv = parser.parse_args()
    # 得到对应的输入数据集值
    datasets = argv.datasets
    # 并创建对应的训练集与测试集文件夹
    create_dirs(train_dir, test_dir)
    #  如果输入的是vcc2016数据集
    if datasets == 'vcc2016' or datasets == 'VCC2016':
        # 就下载对应数据集
        download_vcc2016()
    # 否则就显示无该数据集
    else:
        print('Dataset not available.')
