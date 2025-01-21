# 神经网络检测

[MaaFW Pipeline NeuralNetworkDetect](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#neuralnetworkdetect).

神经网络检测，基于深度学习的“找图”。

简单举例：

- 这张图是猫，狗，还是鸟？答：是猫。—— 这是分类。
- 这张图是猫，狗，还是鸟？答：[100, 100, 20, 80] 这个位置有一只猫，[200, 200, 50, 30] 这里还有一只猫，[300, 300, 20, 50] 这里有一只狗，但是没看到哪里有鸟。—— 这是检测。

请根据您的实际需求判断需要分类还是检测。相对的，检测模型更加强大、输出信息更多，但也意味着需要更多的训练时间和数据集，实际运行也会更慢。

MaaFW 使用 YOLO 标准的输入输出格式，若您有 YOLO 训练经验，可直接复用，然后将 pt 模型导出为 onnx 模型即可。  
同时网上的 YOLO 训练教程非常多，若您对本食谱有不理解的，也可查阅 [官方文档](https://docs.ultralytics.com/) 或其他教程（当然也欢迎向我们提问或提供修改建议）。

## 准备炊具

*相较分类，训练检测模型对设备性能要求较高，虽然理论上 CPU 也能跑，但还是非常推荐你有一块 Nvidia GPU。*

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

## 开始烹饪

```bash
python ./train.py
```

## 出锅装盘

导出 ONNX 模型

```bash
python ./export.py
```
