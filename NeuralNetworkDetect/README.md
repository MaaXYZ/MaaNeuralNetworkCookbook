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

### 获取食材

首先，您需要明确要检测的目标，该目标在不同的背景/场景中应当具有相同/相似的特征。

一般而言，每一类目标的数量推荐至少大于200。这意味着如果一张图中只有一个目标，你需要准备超过200张图片以供训练。

图片的分辨率不需要全都一样(一样最好)，图片中除了目标也应当包含一些背景信息，这有助于提升模型的鲁棒性，如下所示

![image-20250311143632840](https://img.ravi.top/img/48c0511cac8d25e1071b17a823bf317b.png)![image-20250311143945576](https://img.ravi.top/img/12cb1190cd163fc65c62165ecc194859.png)

### 食材调味

> 获取食材之后就要调调味了，不然没法煮

对于图片标注，推荐使用 [roboflow](https://roboflow.com/)

注册登录之后，点击 `NewProject`

![image-20250311150240483](https://img.ravi.top/img/f2fd050161e79d7fdc3659cab6e58ee1.png)

然后上传图片，也就是数据集

![image-20250311150643405](https://img.ravi.top/img/3f62fc40e2462ce516f258e2c5dd3d73.png)

![image-20250311150802290](https://img.ravi.top/img/6aa51620d427b32516b2ed24565cc187.png)

上传完成后点击 `Start Manual Labeling` ，选择 `Assign to myself`，再点击 `Start Annotating` 开始标注

![image-20250311151543822](https://img.ravi.top/img/64e8d062842ad8fab2e2d880ab697806.png)

全部标注完成后返回，点击 `Add images to Dataset`

再进入左侧的 `Versions`，点击 `Rebalance` 调整数据集的比例，一般建议为7:3:1

![image-20250311152009121](https://img.ravi.top/img/1517013eb27003e981e74f8cdb6517bd.png)

然后在Preprocessing中，将除了 `Auto-Orient` 的其他选项都删掉，Augmentation不用动，最后点击 `Create` 即可

![image-20250311152559800](https://img.ravi.top/img/320ae0e4a5a9a9ea4d7bd8653f6b303b.png)

稍等一会就下载好了

## 开始烹饪

将前文中下载的压缩包解压到dataset文件夹

```bash
yolo detect train data=./dataset/data.yaml model=yolo11n.pt epochs=500 imgsz=640 batch=0.8 cos_lr=True patience=100
```

参数解释：

- imgsz: 训练时使用的图片尺寸，必须为32的倍数，yolo会使用letterbox自动缩放到指定尺寸，尺寸越大，显存需求量也越大，如640就是缩放到640x640

  **注意：train的imgsz不建议与export的imgsz差距过大，同时分辨率越大推理速度和训练速度相应也会下降**

- model: 选择预训练模型，一般而言模型规模越大，精度越高，速度越慢，一般选n或s即可，各尺寸模型对比详见 [ultralytics](https://docs.ultralytics.com/models/yolo11/#__tabbed_1_1) 

  **注意：模型规模越大，最后导出的onnx模型也越大，其大小约为预训练模型的2~3倍**

- epochs: 训练轮次，可以适当调大些
- patience: 在多少轮训练之后，如果mA等指标无明显提升则提前中止训练
- cos_lr: 使用余弦学习率调度器，可以有效提高模型的收敛效果
- batch: 每次加载多少图片到显卡中，填写小数会根据显存自动决定，如0.8会占用80%的显存。多卡训练时不能填小数

更多参数参见 https://docs.ultralytics.com/modes/train/#train-settings

您可以根据自己的理解尝试调整一下各种参数，可能会有别样的风味?

## 品尝佳肴

烹饪结束后，您可以打开 `F1_curve.png` 和 `PR_curve.png` 以评估训练效果

对于这两张图，简单而言，曲线越靠近右上角说明训练效果越好

![image-20250311172441872](https://img.ravi.top/img/ccb823e40614a1f883701429f694b390.png)

如上图所示，`video_small` 这一类的训练效果就相对较差

图中标签后的小数为mAP50分数，您可以简单地理解为检测的准确率，即该值越大越好

![image-20250311192410125](https://img.ravi.top/img/2e35b54512a23ae9cefacbb62b31aa2b.png)

F1是对准确率和召回率的调和平均数，您可以通过该图决定置信度(Confidence)阈值设置为多少比较合适

图中的 `all classes 0.95 at 0.721` 意味着当置信度阈值为0.721时，F1分数取得最大值0.95

如果您对味道满意的话就可以出锅装盘啦，不满意就回锅重做吧~~

## 出锅装盘

进入weight文件夹，导出 ONNX 模型

```bash
yolo export model=best.pt format=onnx imgsz=320,640
```

参数解释:

- model: 准备导出的pt模型
- format: 要导出的格式
- imgsz: 输入图片的尺寸，格式为 `height, width`

更多参数参见 https://docs.ultralytics.com/modes/export/#arguments

**注意：imgsz需要符合在pipeline中设置的ROI大小**

## 改进配方

TODO
