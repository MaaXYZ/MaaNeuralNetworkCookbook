from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torch
import time

################
# 这部分需要根据项目实际情况修改

# 数据集所在目录
cwd = Path(__file__).parent
TRAIN_PATH = cwd / "data" / "train"
TEST_PATH = cwd / "data" / "test"
OUTPUT_PATH = cwd / "data" / "model"

# 数据有哪些，这里的顺序也对应最终输出的顺序
LABELS = ["dog", "cat", "bird"]

# 一般来说越大越快，但太大会爆显存/内存
BATCH_SIZE = 1024

# 线性层输入，这个要计算的，有点复杂，先不用管，后面会报错，然后它会告诉你正确的是多少
# 例如报错：RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x2200 and 1000x3)
# 需要改成这里的 mat1.shape[1] 即 2200
GOOGLENET_LINEAR = 2200

# 再后面的代码有兴趣可以看看，不改也行
################

class MyDataset(Dataset):

    def __init__(self, path: Path):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),  # 高斯模糊
                transforms.RandomPosterize(3),  # 随机压缩
                transforms.RandomAdjustSharpness(3),  # 随机锐化
                transforms.RandomAutocontrast(),  # 自动对比度调整
                # ...... 更多变换请根据实际应用场景添加
                # https://pytorch.ac.cn/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
                transforms.ToTensor(),  # 这个是一定要的
            ]
        )

        self.data = []
        self.labels = defaultdict(int)
        for f in path.glob("*.png"):
            # 如果你的文件名不是这种格式，请自行修改此处代码
            label = f.stem.split("-")[0]
            if label not in LABELS:
                print(f"Unknown label: {label}")

            self.labels[label] += 1

            # pytorch 实际会调用 __len__ 和 __getitem__
            # 如果内存不够，改为在 __getitem__ 再读文件即可
            image = Image.open(f).convert("RGB")

            label = LABELS.index(label)

            self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label


train_data = MyDataset(TRAIN_PATH)
test_data = MyDataset(TEST_PATH)
print(f"train len: {len(train_data)}, {dict(train_data.labels)}")
print(f"test len: {len(test_data)}, {dict(test_data.labels)}")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 本次演示我们使用 GoogleNet，导入本目录下的 google_net.py
from src.google_net import GoogleNet

use_cuda = torch.cuda.is_available()
device_type = "cuda" if use_cuda else "cpu"
device = torch.device(device_type)
if not use_cuda:
    print("WARNING: CPU will be used for training.")

# channels - 输入图像的通道数。我们是 RGB 三通道
# linear - 线性层输入。这个要计算的，有点复杂，不用管，后面会报错，然后它会告诉你正确的是多少
# cls_count - 分类数。我们是 猫、狗、鸟 三分类
model = GoogleNet(channels=3, linear=GOOGLENET_LINEAR, cls_count=len(train_data.labels))

# 将数据移到 GPU/CPU
model = model.to(device)

from torch.amp import GradScaler, autocast

# 这里优化器用的 Adam，可以搜一下不同优化器的区别，挑一种合适的
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学习率可以自己调一下
criterion = torch.nn.CrossEntropyLoss().to(device)
scaler = GradScaler()


def train(epoch: int):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # 将数据移到 GPU/CPU
        data, target = data.to(device), target.to(device)

        # 梯度归零
        optimizer.zero_grad()

        # 前馈
        # 自动混合精度
        with autocast(device_type):
            ## 计算 y_pred
            output = model(data)
            ## 计算损失
            loss = criterion(output, target)

        # 反馈
        ## 反向传播
        scaler.scale(loss).backward()

        # 更新
        scaler.step(optimizer)
        scaler.update()

        print(
            f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx * len(data) / len(train_loader.dataset):.0f}%)]\tLoss: {loss.item():.6f}"
        )


def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    # 测试不需要计算梯度
    with torch.no_grad():
        for data, target in test_loader:

            # 将数据移到 GPU/CPU
            data, target = data.to(device), target.to(device)

            # 前馈
            # 自动混合精度
            with autocast(device_type):
                output = model(data)
                test_loss += criterion(output, target).item()

            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)

    print(
        f"Test: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)"
    )

    return test_loss, acc


output_dir = OUTPUT_PATH
output_dir.mkdir(exist_ok=True, parents=True)

model_name = model.__class__.__name__
best_model_path = output_dir / f"{model_name}_best.pt"

best_epoch = 0
best_loss = 100.0
best_acc = 0.0


def pipeline(start_epoch=0):
    global best_epoch, best_loss, best_acc

    for epoch in range(start_epoch, 1000):
        starting = time.time()

        train(epoch)
        loss, acc = test()

        cost = int(time.time() - starting)
        print(f"Epoch: {epoch}, Cost: {cost}s\n")

        # torch.save(model, output_dir / f"{model_name}_{epoch}.pt")

        if loss > best_loss:
            if epoch - best_epoch > 100:
                print("No improvement for a long time, Early stop!")
                break
            else:
                continue
            
        best_epoch = epoch
        best_loss = loss
        best_acc = acc
        print(
            f"====== New best is epoch {best_epoch}, Loss: {best_loss:.8f}, Accuracy: {best_acc:.4f} ======\n"
        )
        torch.save(model, output_dir / f"{model_name}_{epoch}.pt")
        torch.save(model, output_dir / f"{model_name}_best.pt")


if use_cuda:
    torch.cuda.empty_cache()

pipeline()
