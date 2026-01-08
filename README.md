# Animal Classification Model

This animal classification model is built from the Animals dataset hosted on Hugging Face:

https://huggingface.co/datasets/Fr0styKn1ght/Animals

## Class set (reduced)

This repo trains and exports a **reduced classification set** (a curated subset of the Hugging Face dataset). Any animals not in the list below are excluded during dataset preparation/training, and the resulting `classmap.json` is generated/ordered accordingly.
<details>
  <summary>Included classes (31)</summary>
- badger
- bat
- bear
- boar
- cat
- cow
- coyote
- deer
- dog
- donkey
- fox
- goat
- hamster
- hare
- hedgehog
- horse
- kangaroo
- koala
- lizard
- mouse
- pig
- possum
- raccoon
- rat
- sheep
- snake
- squirrel
- turkey
- turtle
- wolf
- wombat
</details>

If you want to train on a different label set, update the allow-list used during dataset preparation, regenerate the dataset on disk, and retrain. Ensure you also regenerate `classmap.json` so inference/export uses the correct labels.

## Scripts

* animal_dataset.py - Download the dataset from Hugging Face and export it to an ImageFolder layout for PyTorch.
* train.py - ResNet50 trainer with optional resume, mixup/cutmix, and metrics export.
* infer.py - Run inference on a single image or a directory of images.
* export.py - Export a model to ONNX, OpenVINO, Core ML, and NCNN.

## Quickstart

0. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Create the dataset on disk:

```bash
python animal_dataset.py --output-dir animal-dataset --test-size 0.2
```

2. Train the model:

```bash
python train.py --data-dir animal-dataset
```

3. Run inference:

```bash
python infer.py /path/to/image.jpg --checkpoint checkpoints/best_model.pth
python infer.py /path/to/images --checkpoint checkpoints/resnet50_epoch_10.pth --topk 3 --json
```

4. Export model formats:

```bash
python export.py --quiet
python export.py --formats onnx openvino --verbose
```

## Training details

The trainer expects an ImageFolder layout with `train/` and `test/` splits. It writes
checkpoints to `checkpoints/` and emits `classmap.json` in the repo root for inference/export.

Key training options:

* Resume with `--resume checkpoints/resnet50_epoch_10.pth`
* Adjust input size with `--img-size 224`
* Enable mixup/cutmix with `--mixup-alpha`, `--cutmix-alpha`, or the `--aug` preset
  (for example `--aug mixup0.2+cutmix1.0@p0.6`)
* Enable eval-time TTA with `--tta`
* Export metrics and confusion stats with `--write-metrics --metrics-out checkpoints/metrics.json`

## Inference details

`infer.py` accepts either a single image path or a directory. It automatically chooses a
device (`cpu`, `cuda`, or `mps`) unless you override it with `--device`. By default it
looks for `classmap.json` next to the checkpoint, falling back to the repo root.

Examples:

```bash
python infer.py /path/to/image.jpg --checkpoint checkpoints/best_model.pth --topk 5
python infer.py /path/to/images --checkpoint checkpoints/resnet50_epoch_10.pth --json --device cpu
```

## Export details

`export.py` reads a checkpoint (raw state_dict or a dict with `model_state_dict`) and
`classmap.json` to produce format-specific subfolders under `models/`.

Useful flags:

* `--formats onnx openvino coreml ncnn`
* `--opset 13` and `--input-size 224`
* `--openvino-fp16` / `--openvino-fp32` to control OpenVINO precision
* `--smoke-test` to run a quick forward pass and log the top-1 label

## Notes

- The training script expects an ImageFolder layout with `train/` and `test/` splits. If the Hugging Face dataset only provides a train split, the dataset/export tooling will create a test split automatically.
- The training script writes a `classmap.json` file in the repo root for use with inference. With the reduced class set, `classmap.json` will include **only** the classes listed above, ordered to match the model output indices.
- If you change the class allow-list, you should delete/regenerate any previously created dataset folders and retrain to avoid stale labels.

## Core ML export requirements

Core ML export relies on Core ML Tools and currently works best with:

- Python 3.10–3.12
- PyTorch 2.1–2.3
- coremltools 7.x

Known-good install example:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
