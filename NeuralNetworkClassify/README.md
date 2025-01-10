# 神经网络分类

[MaaFW Pipeline NeuralNetworkClassify](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#neuralnetworkclassify).

神经网络分类，用于判断图像中的**固定位置**的“类别”。

要求尺寸、位置确定，否则请使用 [神经网络检测](../NeuralNetworkDetect/)。但相对的，分类模型比较简单，训练和推理速度都会更快。

本文默认你有一定的 ~~基本厨艺~~ Python 基础。如果你熟练使用其他某个编程语言，但不会 Python，也是可以的，我们不涉及很复杂的语法。否则请先自行去学习一下 Python。

## 准备炊具

如果你有一块 Nvidia GPU

```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

否则

```bash
# CPU
pip install torch torchvision
```

更多其他版本请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

finally, 安装其他依赖：

```bash
pip install -r ./requirements.txt
```

## 准备食材

我们需要准备大量的需要分类的图片（数据集）。

例如原始图像中有一块 32x32 的区域只可能是 猫、狗 或者 鸟，而我们想要区分它，则需要先准备大量的该区域的截图（32x32），并做好分类。

32x32 只是举例，实际训练中尺寸可以自己确定，但所有数据集和后续识别 ROI 需为同一个尺寸，否则会强制图片拉伸影响效果。

而我们需要区分猫、狗、鸟这三个类别，这样的任务被称为“三分类”，同理也有“二分类”、“四分类”等等。

本食谱中，我们调用 `download.py` 去下载开源数据集 `CIFAR-10`，它里面包含了猫、狗、鸟、飞机、汽车等 10 类 * 6000 张图片。我们只取猫、狗、鸟三种图片各 300 张以演示三分类任务，有兴趣可自行尝试十分类或其他任务。实际应用中，这些图片需要开发者自行截图并整理。

```bash
# 若下载较慢请尝试修改上网方式
python ./download.py
```

## 起锅开火

这里我们使用 Jupyter 文档（.ipynb），它是一种可以将代码、运行结果、文档 等结合的格式，更便于展示。

请使用 VSCode 安装 Jupyter 插件或下载其他 Jupyter 软件，然后打开 [train.ipynb](./train.ipynb)。
