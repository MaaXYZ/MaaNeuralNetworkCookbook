from torchvision import datasets
from pathlib import Path
from collections import defaultdict

CIFAR10_CLS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
REQUIRED_CLS = {"bird", "cat", "dog"}


cwd = Path(__file__).parent
raw = cwd / "data" / "raw"

# 下载 CIFAR-10
train_data = datasets.CIFAR10(root=raw, train=True, download=True)
test_data = datasets.CIFAR10(root=raw, train=False, download=True)


def save_image(data, path):

    count = defaultdict(int)
    for image, label in data:
        label = CIFAR10_CLS[label]
        if label not in REQUIRED_CLS:
            continue
        
        image.save(path / f"{label}-{count[label]}.png")
        count[label] += 1


train_path = cwd / "data" / "train"
train_path.mkdir(parents=True, exist_ok=True)
test_path = cwd / "data" / "test"
test_path.mkdir(parents=True, exist_ok=True)

save_image(train_data, train_path)
save_image(test_data, test_path)
