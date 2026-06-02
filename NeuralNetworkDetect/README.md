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

*相较分类，训练检测模型对设备性能要求较高，虽然理论上 CPU 也能跑，但还是非常推荐使用独立显卡（NVIDIA CUDA 或 AMD ROCm）。*

### NVIDIA 显卡

```bash
# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

更多版本请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

### AMD 显卡（Windows + ROCm）

Windows 上可使用 [AMD 官方 ROCm PyTorch](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html)（如 RX 6000/7000/9000）。需 **Python 3.12** 与较新驱动；**训练、验证、导出与 NVIDIA / CPU 相同**，见下文「开始烹饪」「出锅装盘」。

<details>
<summary><b>ROCm 环境部署（Windows）</b> — 点击展开</summary>

适用显卡见 [ROCm Windows 支持列表](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)。

**前置要求：** Windows 11（推荐 22H2+）、[AMD 显卡驱动](https://www.amd.com/en/support/download/drivers.html)（建议 26.1.1+）、[Miniconda](https://docs.conda.io/en/latest/miniconda.html) / Anaconda、Python **3.12**。

#### 1. 创建 Conda 环境

```powershell
conda create -n yolov8-maa python=3.12 -y
conda activate yolov8-maa
```

环境名可自定。

#### 2. 安装 ROCm SDK

按顺序安装（体积较大，需耐心等待）：

```powershell
$base = "https://repo.radeon.com/rocm/windows/.rocm-rel-7.2_a"

python -m pip install --no-cache-dir "$base/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "$base/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "$base/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "$base/rocm-7.2.0.dev0.tar.gz"
```

也可使用 `https://repo.radeon.com/rocm/windows/rocm-rel-7.2/`，包名相同。

#### 3. 安装 PyTorch（ROCm）

**RX 7900 / 6800 等（通用）：**

```powershell
$base = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2"

python -m pip install --no-cache-dir "$base/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
python -m pip install --no-cache-dir "$base/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
python -m pip install --no-cache-dir "$base/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
```

`torchaudio` 对 YOLO 非必需，可跳过。

<details>
<summary><b>RX 6700 XT / 6750 XT（gfx1031）</b> — 标准包可能无法初始化 GPU，点击展开</summary>

```powershell
python -m pip uninstall -y torch torchvision

python -m pip install --no-cache-dir `
  --index-url https://rocm.nightlies.amd.com/whl-staging-multi-arch/ `
  "torch[device-gfx1031]" "torchvision[device-gfx1031]"
```

可用 `offload-arch`（conda 环境 `Scripts` 目录）确认架构，例如输出 `gfx1031`。

</details>

#### 4. 验证环境

```powershell
python -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

输出 `GPU: True` 且能打印显卡名称即可。然后继续执行下文「安装其他依赖」。

#### 常见问题

| 现象 | 处理 |
|------|------|
| `torch.cuda.is_available()` 崩溃 | gfx1031 见上一小节；多显卡可设 `$env:HIP_VISIBLE_DEVICES=0` |
| MIOpen / xnack 警告 | 一般可忽略 |

</details>

### CPU（无独显或仅作测试）

```bash
pip install torch torchvision
```

### 安装其他依赖

```bash
pip install -r ./requirements.txt
```

## 准备食材

### 获取食材

首先，您需要明确要检测的目标，该目标在不同的背景/场景中应当具有相同/相似的特征。

一般而言，每一类目标的数量推荐至少大于 200。这意味着如果一张图中只有一个目标，你需要准备超过 200 张图片以供训练。

图片的分辨率不需要全都一样（一样最好），图片中除了目标也应当包含一些背景信息，这有助于提升模型的鲁棒性，如下所示

![image](https://github.com/user-attachments/assets/c202b058-7c70-44ca-8a78-929792dd6bfe)
![image](https://github.com/user-attachments/assets/48a2d313-0498-4867-a4ac-d202f983cc12)


### 食材调味

> 获取食材之后就要调调味了，不然没法煮

对于图片标注，推荐使用 [roboflow](https://roboflow.com/)

注册登录之后，点击 `NewProject`

![image](https://github.com/user-attachments/assets/29910c38-6ddf-4f48-b305-fd3e5b90920e)


然后上传图片，也就是数据集

![image](https://github.com/user-attachments/assets/d5cc5ff2-0005-4983-8b6a-048c0487cfe5)


![image](https://github.com/user-attachments/assets/e644c156-5e14-41e6-987e-8ed698c889d5)


上传完成后点击 `Start Manual Labeling` ，选择 `Assign to myself`，再点击 `Start Annotating` 开始标注

![image](https://github.com/user-attachments/assets/d9151b3c-555a-4d7b-83d1-0ad905c149e6)


全部标注完成后返回，点击 `Add images to Dataset`

再进入左侧的 `Versions`，点击 `Rebalance` 调整数据集的比例，一般建议为 7:2:1

![image](https://github.com/user-attachments/assets/6b6acbb4-d260-4b0c-86f6-f7b48bad1583)


然后在 Preprocessing 中，将除了 `Auto-Orient` 的其他选项都删掉，Augmentation 不用动，最后点击 `Create` 即可

![image](https://github.com/user-attachments/assets/6feecc7e-661c-4e5f-8f6e-d3c11b1f4c85)


稍等一会就下载好了

## 开始烹饪

将前文中下载的压缩包解压到dataset文件夹

```bash
yolo detect train data=./dataset/data.yaml model=yolo11n.pt epochs=500 imgsz=640 batch=-1 cos_lr=True patience=100
```

参数解释：

- imgsz: 训练时使用的图片尺寸，必须为 32 的倍数，yolo 会使用 letterbox 自动缩放到指定尺寸，尺寸越大，显存需求量也越大，如 640 就是缩放到 640x640

  **注意：train 的 imgsz 不建议与 export 的 imgsz 差距过大，同时分辨率越大推理速度和训练速度相应也会下降**

- model: 选择预训练模型，一般而言模型规模越大，精度越高，速度越慢，一般选 n 或 s 即可，各尺寸模型对比详见 [ultralytics](https://docs.ultralytics.com/models/yolo11/#__tabbed_1_1) 

  **注意：模型规模越大，最后导出的 onnx 模型也越大，其大小约为预训练模型的 2 ~ 3 倍**

- epochs: 训练轮次，可以适当调大些
- patience: 在多少轮训练之后，如果mA等指标无明显提升则提前中止训练
- cos_lr: 使用余弦学习率调度器，可以有效提高模型的收敛效果
- batch: 每次加载多少图片到显卡中，填写小数会根据显存自动决定，如 0.8 会占用 80% 的显存。多卡训练时不能填小数

更多参数参见 [train-settings](https://docs.ultralytics.com/modes/train/#train-settings)

您可以根据自己的理解尝试调整一下各种参数，可能会有别样的风味？

## 品尝佳肴

烹饪结束后，您可以打开 `F1_curve.png` 和 `PR_curve.png` 以评估训练效果

对于这两张图，简单而言，曲线越靠近右上角说明训练效果越好

![image](https://github.com/user-attachments/assets/d0f4731a-0ec7-4877-823c-f41cf56f2d39)


如上图所示，`video_small` 这一类的训练效果就相对较差

图中标签后的小数为 mAP50 分数，您可以简单地理解为检测的准确率，即该值越大越好

![image](https://github.com/user-attachments/assets/10975bd0-7936-48c4-aedc-62ae424610a1)


F1是对准确率和召回率的调和平均数，您可以通过该图决定置信度（Confidence）阈值设置为多少比较合适

图中的 `all classes 0.95 at 0.721` 意味着当置信度阈值为 0.721 时，F1 分数取得最大值 0.95

如果您对味道满意的话就可以出锅装盘啦，不满意就回锅重做吧~

## 出锅装盘

进入 weight 文件夹，导出 ONNX 模型

```bash
yolo export model=best.pt format=onnx imgsz=640 opset=17
```

参数解释:

- model: 准备导出的 pt 模型
- format: 要导出的格式
- imgsz: 输入图片的尺寸，格式为 `height, width`

更多参数参见 [arguments](https://docs.ultralytics.com/modes/export/#arguments)


## 改进配方

TODO
