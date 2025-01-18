# 神经网络分类

[MaaFW Pipeline NeuralNetworkClassify](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#neuralnetworkclassify).

神经网络分类，用于判断图像中的**固定位置**的“类别”。

要求 ROI 固定，否则请使用 [神经网络检测](../NeuralNetworkDetect/)。但相对的，分类模型比较简单，训练和推理速度都会更快。

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

本食谱中，我们调用 `download.py` 去下载开源数据集 `CIFAR-10`，它里面包含了猫、狗、鸟、飞机、汽车等 10 类 * 6000 张图片。我们只取猫、狗、鸟三种图片以演示三分类任务，有兴趣可自行尝试十分类或其他任务。

实际应用中，这些图片需要开发者自行截图并整理。

```bash
# 若下载较慢请尝试修改上网方式
python ./download.py
```

## 开始烹饪

 打开 `train.py`，**根据实际情况修改前面几项大写的变量值**，然后再运行：

```bash
python ./train.py
```

稍后会输出

```plaintext
====== New best is epoch 1, Loss: 0.00105209, Accuracy: 44.3667 ======
```

每次输出这项时，说明训练得到了更优的模型，并保存了模型文件到 `data/model/GoogleNet_best.pt`。

Loss 表示损失，期望尽可能低；Acc 表示准确率，期望尽可能高。

若训练速度较慢，且感觉 Acc 已经差不多了，也可以提前结束训练。

## 出锅装盘

打开 `export.py`，**根据实际情况修改前面几项大写的变量值**，然后再运行：

```bash
python ./export.py
```

最终得到 onnx 格式的模型文件：`data/model/GoogleNet_best.onnx`，即是我们 [MaaFW Pipeline](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#neuralnetworkclassify) 所需要的模型文件。
