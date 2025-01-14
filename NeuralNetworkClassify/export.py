from pathlib import Path
import torch.onnx
import torch

cwd = Path(__file__).parent
MODEL_PATH = cwd / "data" / "model" / "GoogleNet_best.pt"
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

def convert_onnx(path: Path):
    model = torch.load(path, map_location=torch.device("cpu"))
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    torch.onnx.export(
        model,
        dummy_input,
        path.with_suffix(".onnx"),
        input_names=["input"],
        output_names=["output"],
    )


convert_onnx(MODEL_PATH)