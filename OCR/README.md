# 光学文字识别（OCR）

MaaFW 提供了基于 PPOCR 官方推理模型转换的 [ONNX 模型](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/1.1-%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.md#%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6)，可以直接使用，满足绝大部分项目的需求。对于少量的识别错误，也可通过 [replace 字段](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/zh_cn/3.1-%E4%BB%BB%E5%8A%A1%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8D%8F%E8%AE%AE.md#ocr) 进行替换。

OCR fine-tuning 对训练设备要求极高、耗时极长，非常不推荐没有深度学习经验的朋友入坑（一般的，使用 RTX 4090 训练 100k 张图片，需要满载运行一周左右的时间），建议仅在有大量无法替换的的识别错误，才进行操作。

可参考 [MAA AI](https://github.com/MaaAssistantArknights/MaaAI/tree/main/common/OCR)，它通过解包获取游戏文本，同时使用对应的字体，自动生成数据集。
