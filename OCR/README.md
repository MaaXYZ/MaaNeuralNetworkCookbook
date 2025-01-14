# 光学文字识别（OCR）

OCR fine-tuning 对设备要求极高，且速度极慢，非常不推荐没有深度学习经验/没有高性能 GPU 设备的朋友入坑。

一般的，使用 4090 训练 100k 张图片，需要一周左右的时间。

有兴趣可参考 [MAA AI](https://github.com/MaaAssistantArknights/MaaAI/tree/main/common/OCR)，它通过解包获取游戏文本，同时使用对应的字体，自动生成数据集。
