# 此文件用来下载对应的数据文件，主要是vcc2016和evaluation_all文件

# os库是Python标准库，包含几百个函数,常用路径操作、进程管理、环境参数等几类。os.path子库以path为入口，用于操作和处理文件路径。
import os

# argparse是python自带的命令行参数解析包，可以用来方便地读取命令行参数。
import argparse

# shlex类使得为类似于unixshell命令行语言的简单语法编写词法分析器变得很容易。
# 这对于编写小型语言（例如，Python应用程序的运行控制文件）或解析引用的字符串通常很有用。
import shlex

# subprocess类用于子进程管理，生成新的进程，连接它们的输入、输出、错误管道，并且获取它们的返回码。
import subprocess

# zipfile用于处理zip格式压缩文件
import zipfile


# 用于解压zip文件，默认解压地址为data文件夹
def unzip(zip_filepath, dest_dir='./data'):
    # with语句用于取代try/catch/finally语句
    # 紧跟with后面的语句会被求值，返回对象的__enter__()方法被调用，这个方法的返回值将被赋值给as关键字后面的变量。
    # 当with后面的代码块全部被执行完之后，将调用前面返回对象的__exit__()方法。
    # with语句最关键的地方在于被求值对象必须有__enter__()和__exit__()这两个方法，那我们就可以通过自己实现这两方法来自定义with语句处理异常。
    # zipfile.ZipFile以path为路径将对应文件创建一个ZipFile对象并转换，表示一个zip文件。
    with zipfile.ZipFile(zip_filepath) as zf:
        # ZipFile对象自有的extractall从方法归档中提取（解压）出所有成员放入当前工作目录。
        # 参数默认为path=None, members=None, pwd=None
        # path指定一个要提取到的不同目录。
        # members为可选项且必须为namelist() 所返回列表的一个子集。
        # pwd是用于解密文件的密码。
        # 这里将zip_filepath路径的文件放入dest_dir路径中
        zf.extractall(dest_dir)
    print("解压对应文件已完成！")


# 用于下载vcc2016数据集
def download_vcc2016():
    # 数据连接与数据文件
    datalink = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
    data_files = ['vcc2016_training.zip', 'evaluation_all.zip']

    # os.path.exists方法用于判断当前目录下是否存在某个文件
    # 如果对应的数据文件存在就不用下载
    if os.path.exists(data_files[0]) or os.path.exists(data_files[1]):
        print("对应的文件已经存在！")
        return

    # 定义对应训练集与评估数据集
    # f''用于字符串格式化
    trainset = f'{datalink}/{data_files[0]}'
    evalset = f'{datalink}/{data_files[1]}'

    # 定义下载对应文件的命令，使用wget下载
    train_comm = f'wget {trainset}'
    eval_comm = f'wget {evalset}'

    # 调用shlex.split方法将字符串按照空格切分
    train_comm = shlex.split(train_comm)
    eval_comm = shlex.split(eval_comm)

    print('开始下载对应的数据集...')
    # 加入一个子进程开始下载文件，下面使用的参数为args，其中默认是不使用shell执行命令，即参数shell=False。
    # 所以这个参数为一个字符串序列而非一整个字符串，这时subprocess.run一般都与shelx.split一起使用。
    # 如果想使用一整个字符串命令，那么必须设置参数shell=True
    subprocess.run(train_comm)
    subprocess.run(eval_comm)
    # 下载完成后调用自定义的unzip方法解压zip格式文件
    unzip(data_files[0])
    unzip(data_files[1])
    print('已经下载完对应的数据集！')


# 创建目录方法，创建训练集与测试集
def create_dirs(trainset: str = './data/fourspeakers', testset: str = './data/fourspeakers_test'):
    """创建测试与训练数据集"""
    if not os.path.exists(trainset):
        # 如果不存在该目录就创造目录
        print(f'create train set dir {trainset}')
        # os.makedirs用于递归创建目录。
        # 如果子目录创建失败或者已经存在，会抛出一个OSError的异常，Windows上Error 183即为目录已经存在的异常错误。
        # 如果第一个参数path只有一级，则和mkdir()函数相同。
        # exist_ok表示是否在目录存在时触发异常。如果exist_ok为False（默认值），则在目标目录已存在的情况下触发FileExistsError异常；
        # 如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常。
        os.makedirs(trainset, exist_ok=True)

    if not os.path.exists(testset):
        print(f'create test set dir {testset}')
        os.makedirs(testset, exist_ok=True)


# 当以main方式运行时
if __name__ == '__main__':

    # 定义一个命令行操作对象parse，调用argparse包的ArgumentParser构造函数。

    # 构造函数中的参数有：
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
    parser = argparse.ArgumentParser(description='下载语音转换数据集')

    # 定义对应的数据集路由
    # 默认路由
    datasets_default = 'vcc2016'
    # 训练集路由
    train_dir = './data/fourspeakers'
    # 测试集路由
    test_dir = './data/fourspeakers_test'

    # 调用add_argument方法给命令行对象增加命令参数，即输入对应命令就跳到不同指令处理程序。默认为python/python3 download.py

    # name or flags - 一个命名或者一个选项字符串的列表，这是第一个参数，可以直接添加而不用指明名称，例如 foo 或 -f, --foo。
    # 命名前加上--就表示是可选参数不必输入的，而没有则是必须的。
    # 如果两个及以上可选参数而同一种处理方式，直接加上所有的参数，如：'--v','--version'。
    # action - 当参数在命令行中出现时使用的动作基本类型。
    # nargs - 命令行参数应当消耗的数目。
    # const - 被一些 action 和 nargs 选择所需求的常数。
    # default - 当参数未在命令行中出现时使用的值。
    # type - 命令行参数应当被转换成的类型。
    # choices - 可用的参数的值域。
    # required - 此命令行选项是否可省略 （仅选项可用）。
    # help - 一个此选项作用的简单描述。
    # metavar - 在使用方法消息中使用的参数值示例。
    # dest - 被添加到 parse_args() 所返回对象上的属性名。

    parser.add_argument('--datasets', type=str, help='可使用的数据集为：vcc2016', default=datasets_default)
    parser.add_argument('--train_dir', type=str, help='模型训练的目录', default=train_dir)
    parser.add_argument('--test_dir', type=str, help='模型测试的目录', default=test_dir)

    # parse_args方法解析参数并把结果赋值给argv
    argv = parser.parse_args()
    # 得到对应的输入数据集值
    datasets = argv.datasets
    # 并调用自定义的create_dirs方法创建对应的训练集与测试集文件夹
    create_dirs(train_dir, test_dir)
    #  如果输入的是vcc2016数据集
    if datasets == 'vcc2016' or datasets == 'VCC2016':
        # 就下载对应数据集
        download_vcc2016()
    # 否则就显示无该数据集
    else:
        print('找不到对应的数据集')

# 经过运行download.py后会在data文件夹下出现四个文件夹，fourspeakers为训练文件夹，fourspeakers_test为测试文件夹。
# vcc2016_training和evaluation_all为对应下载文件的解压文件夹。
