
# Building

## Install dependencies

### Conda environment

All Python scripts in this project require a dedicated conda environment.
Create and activate it with:

```shell
conda create -n torch-pascal python=3.12
conda activate torch-pascal
```

Then install the required packages:

```shell
# Install PyTorch (see pytorch.org for CUDA/cpu variants)
conda install pytorch torchvision torchaudio -c pytorch

# Install mamba-ssm and causal-conv1d
pip install mamba-ssm causal-conv1d
```

> **Note:** Remember to run `conda activate torch-pascal` before executing any Python scripts in this project.

This project also needs the following additional dependencies:

- espup (rust toolchain for ESP devices)

## Train the model

```shell
python -m train.main
```

This exports the model to onnx and places it in the src/model directory

## Quantize model for ESP-DL

```shell
python -m train.quantize --model $MODEL --dataset $DATASET
```

Generates a quantized `.espdl` model from the ONNX export for use on ESP targets.

## Run on microcontroller

Make sure don activate the ESP-IDF environment first:

```shell
source ~/.espressif/tools/activate_idf_v6.0.1.sh
```

Then build and flash the ESP-DL project onto an ESP32-S3 or ESP32-P4:

```shell
cd esp-dl
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

This compiles the ESP-DL inference example and flashes it to the device.

## Environment variables

`MODEL`: Select which model to use. Choices from the following models: `mamba-1` `mamba-3`.
`DATASET`: Select which model to use. Choices from the following models: `kws` `har`.
