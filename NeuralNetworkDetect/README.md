# 神经网络检测

[MaaFW Pipeline NeuralNetworkDetect](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#neuralnetworkdetect).

神经网络检测，基于深度学习的“找图”。与分类器相比，支持动态的尺寸、位置。

MaaFW 使用 YOLO 标准的输入输出模型，若您有 YOLO 训练经验，可直接复用，然后将 pt 模型导出为 onnx 模型即可。  
同时网上的 YOLO 训练教程非常多，若您对本食谱有不理解的，也可查阅 [官方文档](https://docs.ultralytics.com/) 或其他教程（当然也欢迎向我们提问或提供修改建议）。

*相较分类模型，检测对训练设备性能要求较高，虽然理论上 CPU 也能跑，但还是非常推荐你有一块 Nvidia GPU。*

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

TODO

## 起锅开火

开始训练

```bash
python ./train.py
```

## 出锅装盘

导出 ONNX 模型

```bash
python ./export.py
```
