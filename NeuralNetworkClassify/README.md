# 神经网络分类

判断图像中的**固定位置**是否为预期的“类别”。

[MaaFW Pipeline](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#neuralnetworkclassify).

本文默认你有一定的 ~~基本厨艺~~ Python 基础。如果你熟练使用其他某个编程语言，但不会 Python，也是可以的，我们不涉及很复杂的语法。否则请先自行去学习一下 Python。

## 准备食材

我们需要准备大量的需要分类的图片（数据集）。

例如原始图像中有一块 100x100 的区域可能是 猫、狗 或者 老鼠，而我们想要识别出该 ROI 内可能是什么，则需要先准备大量的该区域的截图（100x100），并将它们分好类。

比如 cat 文件夹里放 1000 张 100x100 的该区域是猫的截图；dog 文件夹中放了 1300 张 100x100 的该区域是狗的截图，等等。

100x100 只是举例，实际训练中尺寸可以自己确定，但所有数据集和后续识别 ROI 需为同一个尺寸，否则会强制图片拉伸影响效果。

这也就是我们所说的“分类”，即判断图像中的**固定位置**是否为预期的类别。若大小、位置等是动态的，请参考 [神经网络检测](../NeuralNetworkDetect/)。

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