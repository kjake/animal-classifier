import os
import json
import platform
import shutil
from pathlib import Path

import torch
from torchvision import models
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

with open("classmap.json", "r") as f:
    idx_to_class = json.load(f)

model = models.resnet50(weights="DEFAULT")
num_classes = len(idx_to_class)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load("checkpoints/best_model.pth", map_location=device)
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    # Strip 'module.' prefix if present (from DataParallel)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)  # remove only first 'module.'
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

label_name_map = idx_to_class

input_shape = (1, 3, 224, 224)

img = np.zeros(input_shape, dtype=np.float32)

stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean = np.array(stats[0], dtype=np.float32).reshape(1, -1, 1, 1)
std = np.array(stats[1], dtype=np.float32).reshape(1, -1, 1, 1)
img = ((img - mean) / std).astype(np.float32)

example_input = torch.from_numpy(img).to(device)

# Quick sanity-check inference (optional)
with torch.no_grad():
    output_data = model(example_input)
    o = output_data[0].softmax(dim=0)
    result = torch.max(o, dim=0)
    label = label_name_map.get(str(result.indices.item()), str(result.indices.item()))
    print(f"Predicted label: {label}")

model_config = {
    "input_shape": input_shape,
    "model": "resnet",
    "mean": stats[0],
    "std": stats[1],
    "files": [],
    "labels": label_name_map,
}

def _rm_rf(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)

def _mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _cpu_model_and_input():
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = device

    model_cpu = model.to("cpu").eval()
    example_cpu = torch.from_numpy(img).to("cpu")

    def _restore():
        # Move back to original device for any subsequent work
        try:
            model.to(orig_device).eval()
        except Exception:
            # If restore fails (rare), at least leave model on CPU
            model.to("cpu").eval()

    return model_cpu, example_cpu, _restore

def export_onnx(opset: int = 13):
    path = "models/onnx"
    _rm_rf(path)
    _mkdir(path)

    model_cpu, example_cpu, restore = _cpu_model_and_input()
    try:
        torch.onnx.export(
            model_cpu,
            example_cpu,
            f"{path}/model.onnx",
            verbose=False,
            input_names=["input"],
            opset_version=opset,
        )
    finally:
        restore()

    model_config["files"] = ["model.onnx"]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_openvino():
    import openvino as ov

    onnx_path = "models/onnx/model.onnx"
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Run export_onnx() first.")

    path = "models/openvino"
    _rm_rf(path)
    _mkdir(path)

    # If you need to force the input shape/type, the following is the most explicit:
    # ov_model = ov.convert_model(onnx_path, input={"input": input_shape})
    ov_model = ov.convert_model(onnx_path)

    # On macOS CPU, keeping FP32 weights can be safer during bring-up; you can switch back to FP16 later.
    compress_fp16 = False if platform.system() == "Darwin" else True
    ov.save_model(ov_model, f"{path}/model.xml", compress_to_fp16=compress_fp16)

    model_config["files"] = ["model.xml", "model.bin"]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_coreml():
    import sys

    supported_python_min = (3, 10)
    supported_python_max = (3, 12)
    supported_torch_minors = {(2, 1), (2, 2), (2, 3)}

    if not (supported_python_min <= sys.version_info[:2] <= supported_python_max):
        raise RuntimeError(
            "Core ML export requires Python 3.10–3.12. "
            "Please use a compatible environment (e.g., Python 3.11 with coremltools 7.x)."
        )

    torch_version_parts = torch.__version__.split("+", maxsplit=1)[0].split(".")
    torch_version = tuple(int(part) for part in torch_version_parts[:2] if part.isdigit())
    if torch_version not in supported_torch_minors:
        raise RuntimeError(
            "Core ML export requires a supported PyTorch version. "
            "Known-good: PyTorch 2.1–2.3 with coremltools 7.x."
        )

    import coremltools as ct

    # CoreML conversion is most reliable from a CPU TorchScript trace.
    model_cpu, example_cpu, restore = _cpu_model_and_input()
    try:
        traced_model_cpu = torch.jit.trace(model_cpu.eval(), example_cpu, strict=True)
        traced_model_cpu.eval()
    finally:
        restore()

    path = "models/coreml"
    _rm_rf(path)
    _mkdir(path)

    mlmodel = ct.convert(
        traced_model_cpu,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_shape)],
    )

    mlmodel.save(path + "/model.mlpackage")

    model_config["files"] = [
        "model.mlpackage/Manifest.json",
        "model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_ncnn():
    path = "models/ncnn"
    _rm_rf(path)
    _mkdir(path)

    model_cpu, example_cpu, restore = _cpu_model_and_input()
    try:
        traced_model_cpu = torch.jit.trace(model_cpu.eval(), example_cpu, strict=True)
        traced_model_cpu.eval()
        traced_model_cpu.save(f"{path}/model.pt")
    finally:
        restore()

    input_shape_str = json.dumps(input_shape)
    os.system(f"pnnx {path}/model.pt 'inputshape={input_shape_str}'")

    model_config["files"] = [
        "model.ncnn.param",
        "model.ncnn.bin",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

# comment/uncomment the ones you want to export.
# some exports may not work depending on the model or host operating system.
# openvino may require intel system.
# ncnn has limited model/op support.
export_onnx()
export_openvino()
export_coreml()
export_ncnn()
