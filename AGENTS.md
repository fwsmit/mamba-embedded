# Mamba Embedded — Mamba on ESP32-S3

This project trains **Mamba state-space models (SSMs)** and deploys them on **ESP32-S3** microcontrollers via **ESP-DL** (int8 quantized).

## Repository Structure

| Path | Purpose |
|------|---------|
| `train/` | PyTorch training, ONNX export, quantization |
| `esp-dl/` | ESP-IDF project for ESP32-S3 inference |
| `tools/` | ESP-DL operator compatibility checker |
| `~/Models/` | Exported ONNX models |
| `data/` | Datasets (HAR, SpeechCommands) |
| `run-esp.sh` | Copy an .espdl model, build, flash, and monitor the ESP-DL project |

## Quick Start: Build and Run on ESP32-S3

To build the ESP-DL firmware with a given quantized model, flash it to the device over USB, and monitor the serial output:

```bash
./run-esp.sh path/to/model.espdl
```

This script:
1. Copies the specified `.espdl` file to `esp-dl/main/model/model.espdl`
2. Copies `dataset.bin` from the same directory if present
3. Sources the ESP-IDF v6.0.1 environment
4. Runs `idf.py build` in `esp-dl/`
5. Runs `idf.py flash` on `/dev/ttyACM0`
6. Flashes `dataset.bin` to the `dataset` partition if present
7. Opens a serial monitor and waits for an inference result
8. Returns exit code 0 on `INFERENCE_OK`, 1 on failure, 2 on crash

## Training a Model

```bash
# Activate conda environment first
conda activate torch-pascal

# Train and export to ONNX
python -m train.main

# Quantize the ONNX model to int8 ESP-DL format
python -m train.quantize --model mamba-1 --dataset har
```

Set `MODEL` (e.g., `mamba-1`, `mamba-3`) and `DATASET` (e.g., `har`, `kws`) as environment variables.

## Architecture Search

Run an Optuna-based hyperparameter search with a pre-defined configuration using Hydra:

```bash
conda activate torch-pascal
python -m train.arch_search config/arch-mamba1-kws.yaml
```

The positional argument is the path to a config YAML file. Available configs in `config/`:

| Config file | Model | Dataset | Multi-layer |
|---|---|---|---|
| `arch-mamba1-kws.yaml` | mamba-1 | kws | fixed (1 layer) |
| `arch-mamba1-har.yaml` | mamba-1 | har | fixed (1 layer) |
| `arch-mamba3-kws.yaml` | mamba-3 | kws | fixed (1 layer) |
| `arch-mamba3-har.yaml` | mamba-3 | har | fixed (1 layer) |
| `arch-mamba1-kws-multi.yaml` | mamba-1 | kws | searched over n_layers |

To add a new search configuration, create a new YAML file in `config/` with these fields:

```yaml
BATCHSIZE: 128
EPOCHS: 2
MODEL: mamba-1          # "mamba-1" or "mamba-3"
DATASET: kws            # "kws" or "har"
EXPERIMENT_NAME: "v2"   # distinguishes this experiment in the Optuna study name

SEARCH_SPACE:
  d_model:
    low: 8
    high: 32
  d_state:
    low: 8
    high: 16
  d_conv:
    low: 2
    high: 4
  expand:
    low: 1
    high: 4
  n_layers:
    low: 1
    high: 1   # set low=high to fix at a single value (no suggest_int call)
```

The `SEARCH_SPACE` section defines the Optuna `suggest_*` ranges for each model parameter. For integer parameters without a step, only `low`/`high` are needed. For categorical parameters, use `choices` (e.g. `nheads: {choices: [1, 2, 4, 8]}`). For parameters with a step, add `step` (e.g. `d_model: {low: 8, high: 32, step: 4}`). Only the parameters relevant to the chosen model type are used (`mamba-1` ignores `nheads`).

Results are stored in an Optuna SQLite database (`mamba_hpo.db`) and ONNX files in `~/Models/<MODEL>-<DATASET>-<EXPERIMENT_NAME>/`.

## Model Pipeline

1. **Train** → exports `~/Models/<model>.onnx`
2. **Check op compatibility** → `python tools/check_espdl_ops.py ~/Models/<model>.onnx`
3. **Quantize** → generates `.espdl` file placed in `esp-dl/main/model/`
4. **Build & flash** → `./run-esp.sh ~/Models/<model>.espdl`

## Dataset Partition

A `dataset` partition (type `data`, subtype `undefined`, 2 MB at offset `0x7e0000`) is defined in `partitions.csv` for storing a dataset binary on the ESP32-S3 flash.

To include a dataset:

1. Place `dataset.bin` next to your `.espdl` model file (same directory)
2. Run `./run-esp.sh` as normal — it automatically copies and flashes the dataset to its partition
3. The firmware logs the partition info at startup via `load_dataset()` in `app_main.cpp`

The `load_dataset()` function mmaps the partition and parses the 8-byte header (`uint32 num_samples`, `uint32 elements_per_sample`) followed by the quantized int8 sample data. The firmware then runs inference on every sample by assigning each one to the model's input tensor via `TensorBase::assign()` before calling `model->run()`.

## Known Issues

- The build script expects the ESP32-S3 on `/dev/ttyACM0` and resets via RTS.

## Resolved Issues

- **`model->test()` crash**: Fixed by re-quantizing after fixing the `train/onnx.py` shadowing bug (renamed to `train/onnx_utils.py`). The current esp-ppq version correctly handles scalar-index Gather, so `fix_gather_output_shapes` fixes 0 ops — no longer needed.
- **Garbage output scores**: The quantized model outputs `DATA_TYPE_INT8`. Fixed `app_main.cpp` to check `get_dtype()` and dequantize via `DL_SCALE(exponent)`.

## Performance

- **HAR inference latency**: ~11.4 ms on ESP32-S3 @ 240 MHz (measured via `esp_timer_get_time()`).

## Agent Instructions

These instructions apply to any AI agent working on this repository.

- **Minimal changes**: Make the smallest possible set of edits to satisfy the task. Do not refactor, reorganise, or beautify code beyond what is strictly required.
- **Avoid comments**: Do not add code comments unless the logic is genuinely non-obvious and a comment is more maintainable than clearer code.
- **Keep docs current**: If you modify the project in a way that makes any part of `AGENTS.md` or `README.md` inaccurate, update the affected files to reflect the new state of the world.