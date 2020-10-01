## [基于StarGAN的语音转换模型](https://github.com/hujinsen/pytorch-StarGAN-VC)

这是一个Pythorch实现的论文: [StarGAN VC：星型生成对抗网络下的非并行多对多语音转换](https://arxiv.org/abs/1806.02169).

**转换后的语音示例位于*samples*和*results_2019-06-10*目录中**

## [依赖](https://github.com/hujinsen/pytorch-StarGAN-VC)

- Python 3.6+
- pytorch 1.0
- librosa
- pyworld
- tensorboardX
- scikit-learn

## [使用方式](https://github.com/hujinsen/pytorch-StarGAN-VC)

### 下载数据集

将vcc 2016数据集下载到当前目录。

```shell
python download.py
```

下载的zip文件解压到`./data/vcc2016_training`和`./data/evaluation_all`两个目录。

1. **训练集：** 在本文中，作者从目录`./data/vcc2016_training`选用**四个说话人**。所以我们将对应的文件夹（比如SF1,SF2,TM1,TM2）到`./data/speakers`.
2. **测试集：** 在本文中，作者从目录`./data/evaluation_all`选用**四个说话人**。所以我们将对应的文件夹（比如SF1,SF2,TM1,TM2）到`./data/speakers_test`.

那么数据目录会变成这样：

```null
data
├── speakers  (训练集)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── speakers_test (测试集)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── vcc2016_training (vcc 2016训练集)
│   ├── ...
├── evaluation_all (vcc 2016评价集，作为测试集合)
│   ├── ...
```

### 预处理

从每个语音片段中提取特征（mcep、f0、ap）。这些特性存储为npy文件。我们还计算了每个说话人的统计特征。

```shell
python preprocess.py
```

这个预处理很可能花几分钟！

### 训练

```shell
python main.py
```

### 转换

```shell
python main.py --mode test --test_iters 200000 --src_speaker TM1 --trg_speaker "['TM1','SF1']"
```

## [网络结构](https://github.com/hujinsen/pytorch-StarGAN-VC)

![Snip20181102_2](https://github.com/hujinsen/StarGAN-Voice-Conversion/raw/master/imgs/Snip20181102_2.png)

注：我们的实现遵循了原论文的网络结构，而[pytorch-StarGAN](https://github.com/liusongxiang/StarGAN-Voice-Conversion)的VC代码使用StarGAN的网络。两者都有可以产生良好的音质。

## [参考](https://github.com/hujinsen/pytorch-StarGAN-VC)

[tensorflow StarGAN-VC代码](https://github.com/hujinsen/StarGAN-Voice-Conversion)

[StarGAN代码](https://github.com/taki0112/StarGAN-Tensorflow)

[CycleGAN-VC代码](https://github.com/leimao/Voice_Converter_CycleGAN)

[pytorch-StarGAN-VC代码](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN-VC论文](https://arxiv.org/abs/1806.02169)

[StarGAN论文](https://arxiv.org/abs/1806.02169)

[CycleGAN论文](https://arxiv.org/abs/1703.10593v4)

## 更新于2019/06/10

原实现的网络结构是原论文的网络结构，但为了达到更好的转换效果，本次更新做了如下修改：

- 无训练问题的分类器改进
- 更新损失函数
- 将鉴别器激活函数修改为tanh（双曲正切函数）

---

如果你觉得这个回购是好的，请**点星**！

你的鼓励是我最大的动力！

## [StarGAN-VC](https://github.com/hujinsen/pytorch-StarGAN-VC)

This is a pytorch implementation of the paper: [StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks](https://arxiv.org/abs/1806.02169).

**The converted voice examples are in *samples* and *results_2019-06-10* directory**

## [Dependencies](https://github.com/hujinsen/pytorch-StarGAN-VC)

- Python 3.6+
- pytorch 1.0
- librosa 
- pyworld 
- tensorboardX
- scikit-learn

## [Usage](https://github.com/hujinsen/pytorch-StarGAN-VC)

### Download dataset

Download the vcc 2016 dataset to the current directory 

```
python download.py 
```

The downloaded zip files are extracted to `./data/vcc2016_training` and `./data/evaluation_all`.

1. **training set:** In the paper, the author choose **four speakers** from `./data/vcc2016_training`. So we  move the corresponding folder(eg. SF1,SF2,TM1,TM2 ) to `./data/speakers`.
2. **testing set** In the paper, the author choose **four speakers** from `./data/evaluation_all`. So we  move the corresponding folder(eg. SF1,SF2,TM1,TM2 ) to `./data/speakers_test`.

The data directory now looks like this:

```
data
├── speakers  (training set)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── speakers_test (testing set)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── vcc2016_training (vcc 2016 training set)
│   ├── ...
├── evaluation_all (vcc 2016 evaluation set, we use it as testing set)
│   ├── ...
```

### Preprocess

Extract features (mcep, f0, ap) from each speech clip.  The features are stored as npy files. We also calculate the statistical characteristics for each speaker.

```
python preprocess.py
```

This process may take minutes !


### Train

```
python main.py
```



### Convert



```
python main.py --mode test --test_iters 200000 --src_speaker TM1 --trg_speaker "['TM1','SF1']"
```


## [Network structure](https://github.com/hujinsen/pytorch-StarGAN-VC)

![Snip20181102_2](https://github.com/hujinsen/StarGAN-Voice-Conversion/raw/master/imgs/Snip20181102_2.png)



 Note: Our implementation follows the original paper’s network structure, while [pytorch StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion) use StarGAN's network.Both can generate good audio quality. 

## [Reference](https://github.com/hujinsen/pytorch-StarGAN-VC)
[tensorflow StarGAN-VC code](https://github.com/hujinsen/StarGAN-Voice-Conversion)

[StarGAN code](https://github.com/taki0112/StarGAN-Tensorflow)

[CycleGAN-VC code](https://github.com/leimao/Voice_Converter_CycleGAN)


[pytorch-StarGAN-VC code](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN-VC paper](https://arxiv.org/abs/1806.02169)

[StarGAN paper](https://arxiv.org/abs/1806.02169)

[CycleGAN paper](https://arxiv.org/abs/1703.10593v4)

## Update 2019/06/10

The former implementation's network structure is the network of the original paper, but in order to achieve better conversion result, the following modifications are made in this update:

- Modification of classifier without training problem
- Update loss function
- Modify the discriminator activation function to tanh

---

If you feel this repo is good, please  **star**  ! 

Your encouragement is my biggest motivation!
