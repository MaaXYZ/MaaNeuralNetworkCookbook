# AMD 显卡 ROCm 环境部署（Windows）

本文档仅说明 **Windows + AMD 显卡** 下如何安装 ROCm 与 PyTorch，以便运行本目录的 Ultralytics YOLO 训练。  
**训练、评估、导出 ONNX 的步骤请以 [README.md](./README.md) 为准**，与 NVIDIA CUDA、CPU 一致，无需单独命令。

适用显卡见 [ROCm Windows 支持列表](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)（如 RX 6000/7000/9000 系列）。

> **前置要求**
>
> - Windows 11（推荐 22H2 及以上）
> - [AMD 显卡驱动](https://www.amd.com/en/support/download/drivers.html)（ROCm 7.2 PyTorch 官方建议 **26.1.1** 及以上）
> - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
> - Python **3.12**（ROCm Windows 轮子为 cp312）

## 1. 创建 Conda 环境

```powershell
conda create -n yolov8-maa python=3.12 -y
conda activate yolov8-maa
```

环境名可自定。

## 2. 安装 ROCm SDK

按顺序安装（体积较大，需耐心等待）：

```powershell
$base = "https://repo.radeon.com/rocm/windows/.rocm-rel-7.2_a"

python -m pip install --no-cache-dir "$base/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "$base/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "$base/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl"
python -m pip install --no-cache-dir "$base/rocm-7.2.0.dev0.tar.gz"
```

也可使用官方路径：`https://repo.radeon.com/rocm/windows/rocm-rel-7.2/`，包名相同。

## 3. 安装 PyTorch（ROCm）

### 3.1 通用（如 RX 7900 / 6800 等）

```powershell
$base = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2"

python -m pip install --no-cache-dir "$base/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
python -m pip install --no-cache-dir "$base/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
python -m pip install --no-cache-dir "$base/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
```

`torchaudio` 对 YOLO 训练非必需，可跳过；若 404 请从 `rocm-rel-7.2` 目录选用 `torchaudio-2.9.1+rocmsdk20260116` 包。

### 3.2 RX 6700 XT / 6750 XT（gfx1031）

标准 ROCm PyTorch 包下，`torch.cuda.is_available()` 可能崩溃。需改用多架构 nightly：

```powershell
python -m pip uninstall -y torch torchvision

python -m pip install --no-cache-dir `
  --index-url https://rocm.nightlies.amd.com/whl-staging-multi-arch/ `
  "torch[device-gfx1031]" "torchvision[device-gfx1031]"
```

可用 `offload-arch`（conda 环境 `Scripts` 目录）确认架构，例如输出 `gfx1031`。

## 4. 安装项目依赖

在 `NeuralNetworkDetect` 目录下：

```powershell
pip install -r requirements.txt
```

## 5. 验证环境

```powershell
python -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
```

输出中 `GPU: True` 且能打印显卡名称即可。若 `cuda.is_available()` 崩溃，见 [3.2 节](#32-rx-6700-xt--6750-xtgfx1031)。

## 6. 常见问题（仅部署相关）

| 现象 | 处理 |
|------|------|
| `torch.cuda.is_available()` 进程崩溃 | gfx1031 见 [3.2](#32-rx-6700-xt--6750-xtgfx1031)；多显卡时可设 `$env:HIP_VISIBLE_DEVICES=0` |
| MIOpen / xnack 警告 | 安装或训练日志中常见，一般可忽略 |

---

环境就绪后，请回到 **[README.md → 开始烹饪](./README.md#开始烹饪)** 进行训练。

## 参考链接

- [AMD：Windows 上通过 pip 安装 PyTorch（ROCm 7.2）](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html)
- [ROCm Windows 支持的 GPU 列表](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)
