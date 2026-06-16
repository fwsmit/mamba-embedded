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
| `build-esp-dl.sh` | Build, flash, and monitor the ESP-DL project |

## Quick Start: Build and Run on ESP32-S3

To build the ESP-DL firmware, flash it to the device over USB, and monitor the serial output:

```bash
./build-esp-dl.sh
```

This script:
1. Sources the ESP-IDF v6.0.1 environment
2. Runs `idf.py build` in `esp-dl/`
3. Runs `idf.py flash` on `/dev/ttyACM0`
4. Opens a serial monitor and waits for an inference result
5. Returns exit code 0 on `INFERENCE_OK`, 1 on failure, 2 on crash

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

Run an Optuna-based hyperparameter search with a pre-defined configuration:

```bash
conda activate torch-pascal
python -m train.arch_search --config config/arch-mamba1-kws.yaml
```

The `--config` flag is **required** and selects which configuration file to use. Available configs in `config/`:

| Config file | Model | Dataset | Multi-layer |
|---|---|---|---|
| `arch-mamba1-kws.yaml` | mamba-1 | kws | no |
| `arch-mamba1-har.yaml` | mamba-1 | har | no |
| `arch-mamba3-kws.yaml` | mamba-3 | kws | no |
| `arch-mamba3-har.yaml` | mamba-3 | har | no |
| `arch-mamba1-kws-multi.yaml` | mamba-1 | kws | yes |

To add a new search configuration, create a new YAML file in `config/` with these fields:

```yaml
BATCHSIZE: 128
EPOCHS: 2
MODEL: mamba-1          # "mamba-1" or "mamba-3"
DATASET: kws            # "kws" or "har"
MULTI_LAYER: false      # search over number of layers
EXPERIMENT_NAME: "v2"   # distinguishes this experiment in the Optuna study name
```

Results are stored in an Optuna SQLite database (`mamba_hpo.db`) and ONNX files in `~/Models/<MODEL>-<DATASET>-<EXPERIMENT_NAME>/`.

## Model Pipeline

1. **Train** → exports `~/Models/<model>.onnx`
2. **Check op compatibility** → `python tools/check_espdl_ops.py ~/Models/<model>.onnx`
3. **Quantize** → generates `.espdl` file placed in `esp-dl/main/model/`
4. **Build & flash** → `./build-esp-dl.sh`

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