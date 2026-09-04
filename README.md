
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

## Run all experiments

To run the complete experiment pipeline — architecture search followed by
quantization, MCU inference and result collection for every experiment — use
`run-all-experiments.sh`:

```shell
./run-all-experiments.sh
```

The script runs two steps on all configurations in `config/har/` and
`config/kws/`:

1. `python -m train.arch_search` — Optuna hyperparameter search for each
   experiment (HAR and KWS, single-direction and bidirectional models),
   storing trials in the `mamba_hpo.db` Optuna database.
2. `python -m train.top_models` — select the top models from each study,
   quantize them to ESP-DL `.espdl` format, flash and run them on the
   ESP32-S3 (via `run-esp.sh`), and store the results in
   `experiments/<study>/`.

> **Note:** Step 2 requires the ESP32-S3 to be connected on `/dev/ttyACM0`.
> Run `conda activate torch-pascal` first (or `conda run -n torch-pascal
> ./run-all-experiments.sh`) so the Python scripts run in the right
> environment.

## Create all plots

To regenerate every figure used in the thesis from the experiment results in
`experiments/`, run:

```shell
./create-all-plots.sh
```

The script removes the previous `.png` and `.pdf` figures first and then
creates all plots used in the paper.

Figures are written to `figures/` as `.png` (plus `.svg` for plot types
that opt in) and to `figures/pdf/` as `.pdf`. Requires the experiment results
to exist first — run `./run-all-experiments.sh` beforehand if they are
missing.

## Environment variables

`MODEL`: Select which model to use. Choices from the following models: `mamba-1` `mamba-3`.
`DATASET`: Select which model to use. Choices from the following models: `kws` `har`.
