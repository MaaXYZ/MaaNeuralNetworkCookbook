import torch

"""
这个代码实现了一个简单的GoogleNet模型，其中包含了两个Inception模块。

Inception模块是GoogleNet中的重要组成部分，通过使用不同大小和类型的卷积核来并行计算特征图，从而提高了模型的精度和效率。

GoogleNet是经典的深度神经网络，可以用于图像分类、目标检测和语义分割等任务。
"""


class InceptionA(torch.nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.branch_1x1 = torch.nn.Conv2d(in_ch, 16, kernel_size=1)

        self.branch_5x5_1 = torch.nn.Conv2d(in_ch, 16, kernel_size=1)
        self.branch_5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch_3x3_1 = torch.nn.Conv2d(in_ch, 16, kernel_size=1)
        self.branch_3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_ch, 24, kernel_size=1)

    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)

        branch_5x5 = self.branch_5x5_1(x)
        branch_5x5 = self.branch_5x5_2(branch_5x5)

        branch_3x3 = self.branch_3x3_1(x)
        branch_3x3 = self.branch_3x3_2(branch_3x3)
        branch_3x3 = self.branch_3x3_3(branch_3x3)

        branch_pool = torch.nn.functional.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch_1x1, branch_5x5, branch_3x3, branch_pool]  # 16 + 24 + 24 + 24
        return torch.cat(outputs, 1)


class GoogleNet(torch.nn.Module):
    def __init__(self, channels, linear, cls_count) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(10)
        self.incep2 = InceptionA(20)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(linear, cls_count)

    def forward(self, x):
        in_size = x.size(0)
        x = torch.nn.functional.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = torch.nn.functional.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
