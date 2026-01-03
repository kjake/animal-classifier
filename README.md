# Animal Classification Model

This animal classification model uses the Animals dataset hosted on Hugging Face.

https://huggingface.co/datasets/Fr0styKn1ght/Animals

* animal_dataset.py - Download the dataset from Hugging Face and export it to an ImageFolder layout for PyTorch.
* train.py - Simple ResNet50 trainer. Supports resume.
* infer.py - Infer a single image given a checkpoint.
* export.py - Export a model to OpenVINO, CoreML, ONNX, and NCNN.

## Quickstart

0. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Create the dataset on disk:

```bash
python animal_dataset.py --output-dir animal-dataset
```

2. Train the model:

```bash
python train.py --data-dir animal-dataset
```

3. Run inference:

```bash
python infer.py path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

## Notes

The training script expects an ImageFolder layout with `train/` and `test/` splits. If the Hugging Face
Dataset only provides a train split, the export script will create a test split automatically.

The training script writes a `classmap.json` file in the repo root for use with `infer.py`.

## Core ML export requirements

Core ML export relies on Core ML Tools and currently works best with:

* Python 3.10–3.12
* PyTorch 2.1–2.3
* coremltools 7.x

Known-good install example:

```bash
python -m pip install -r requirements.txt
```
