
# Building

## Install dependencies

This project needs the following dependencies:

- espup (rust toolchain for ESP devices)
- pytorch
- mamba-ssm

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

## Run ESP-DL

Make sure to activate the ESP-IDF environment first:

```shell
source ~/.espressif/tools/activate_idf_v6.0.1.sh
```

Then build and flash the ESP-DL project onto an ESP32-S3 or ESP32-P4:

```shell
cd esp-dl
idf.py set-target <target>   # e.g. esp32s3 or esp32p4
idf.py build
idf.py flash monitor
```

This compiles the ESP-DL inference example and flashes it to the device.

## Run rust project (going away)

```shell
cargo run --release
```

Does the same as ESP-DL.

### Create test vectors (for rust project)

```shell
python -m train.har_to_burn_tensor
python -m train.kws_to_burn_tensor
```

## Environment variables

`MODEL`: Select which model to use. Choices from the following models: `mamba-1` `mamba-3`.
`DATASET`: Select which model to use. Choices from the following models: `kws` `har`.
