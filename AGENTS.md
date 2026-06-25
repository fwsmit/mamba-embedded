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
./run-esp.sh -v path/to/model.espdl   # verbose: real-time build, flash & serial output
```

This script:
1. Copies the specified `.espdl` file to `esp-dl/main/model/model.espdl`
2. Copies the matching `dataset-trial-<N>.bin` (inferred from the model filename) as `dataset.bin` — exits with an error if not found
3. Sources the ESP-IDF v6.0.1 environment
4. Runs `idf.py build` in `esp-dl/`
5. Runs `idf.py flash` on `/dev/ttyACM0` (dataset partition is flashed automatically by the build system)
6. Opens a serial monitor and waits for an inference result
7. Returns exit code 0 on `INFERENCE_OK`, 1 on failure, 2 on crash. Add `-v` before the model path to see all build, flash, and serial output in real time (default is non-verbose, which only prints build/flash output on failure and buffers serial output until a sentinel is hit).

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
plot_description: "Mamba-1 baseline"  # optional; label used in Pareto front plots

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

The optional `plot_description` field sets a custom label for this config in Pareto front comparison plots (see `train/plot_arch_search.py`). If omitted, the plot falls back to the auto-generated study name (`{MODEL}-{DATASET}-{EXPERIMENT_NAME}`).

Results are stored in an Optuna SQLite database (`mamba_hpo.db`) and ONNX files in `~/Models/<MODEL>-<DATASET>-<EXPERIMENT_NAME>/`.

## Model Pipeline

1. **Train** → exports `~/Models/<model>.onnx`
2. **Check op compatibility** → `python tools/check_espdl_ops.py ~/Models/<model>.onnx`
3. **Quantize** → generates `.espdl` file placed in `esp-dl/main/model/`
4. **Build & flash** → `./run-esp.sh ~/Models/<model>.espdl`

## Dataset Partition

A `dataset` partition (type `data`, subtype `undefined`, 9 MB at offset `0x210000`) is defined in `partitions.csv` for storing a dataset binary on the ESP32-S3 flash.

The dataset partition is integrated into the ESP-IDF build system: if `dataset.bin` exists in `esp-dl/main/model/`, the CMake build automatically registers it as a flash image for the `dataset` partition. This means `idf.py flash` handles everything in one step.

To include a dataset:

1. Place `dataset-trial-<N>.bin` next to your `*-trial-<N>.espdl` model file (same directory)
2. Run `./run-esp.sh` as normal — the script automatically matches the trial number and copies it as `dataset.bin`; the build system handles flashing both the firmware and the dataset

For manual testing (without `run-esp.sh`), just copy both files:
```bash
cp path/to/model.espdl      esp-dl/main/model/
cp path/to/dataset-trial-N.bin  esp-dl/main/model/dataset.bin
cd esp-dl
idf.py build && idf.py -p /dev/ttyACM0 flash
```

The firmware logs the partition info at startup via `load_dataset()` in `app_main.cpp`. That function mmaps the partition and parses the 8-byte header (`uint32 num_samples`, `uint32 elements_per_sample`) followed by the quantized int8 sample data. The firmware then runs inference on every sample by assigning each one to the model's input tensor via `TensorBase::assign()` before calling `model->run()`.

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