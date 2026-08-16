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
n_workers: 3                        # number of parallel workers (multiprocessing).
                                    # Set to 1 for single-process execution.
                                    # Each worker runs trials in its own process
                                    # with a separate CUDA context.
quantization_precision: [8]       # bit-width(s) for quantizing models in top_models.py
                                    # scalar (e.g. 8) or list (e.g. [8, 16]) accepted
quantization_methods: [standard]  # optional; PTQ methods to compare in top_models.py.
                                    # "standard" = default esp-ppq PTQ (32 calib samples);
                                    # "strat-kl-tqt" = best-performing 8-bit config
                                    # (KL calibration over 256 stratified samples + TQT
                                    # block_size=512, steps=3000, lr=2e-4; see
                                    # TQT_FINAL_REPORT.md). Each method is stored under
                                    # its own suffixed results keys (e.g. *_strat). If TQT
                                    # diverges (NaN/Inf scales, e.g. on some bidirectional
                                    # models), it automatically falls back to
                                    # calibration-only PTQ so the pipeline still completes.
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

## Performance Optimizations

### Batched reference kernel (mamba-3)

The Mamba-3 reference kernel (`mamba3_siso_fwd_ref_batched` in `train/mamba_cpu_funcs.py`) processes the entire batch in one go instead of looping over individual sequences. This reduces GPU kernel launch overhead significantly.

Measured speedup on CPU (B=128, T=10, d_model=20): **~27×** over the original per-sequence loop. On GPU the speedup is expected to be even larger since the original loop launches 128 tiny CUDA kernels per forward pass.

### Multiprocessing

The architecture search now supports multi-worker execution via the `n_workers` config field. Workers run in separate processes with their own CUDA contexts. Each worker pulls the next available trial from the shared Optuna SQLite database.

Set `n_workers` in the config YAML file. Recommended values:
- Quadro P1000 (4 GB): 3 workers
- Higher-end GPUs: 4-8 workers
- CPU-only: `os.cpu_count()` workers

## Known Issues

- The build script expects the ESP32-S3 on `/dev/ttyACM0` and resets via RTS.

## Resolved Issues

- **`model->test()` crash**: Fixed by re-quantizing after fixing the `train/onnx.py` shadowing bug (renamed to `train/onnx_utils.py`). The current esp-ppq version correctly handles scalar-index Gather, so `fix_gather_output_shapes` fixes 0 ops — no longer needed.
- **Garbage output scores**: The quantized model outputs `DATA_TYPE_INT8`. Fixed `app_main.cpp` to check `get_dtype()` and dequantize via `DL_SCALE(exponent)`.

## Experiments Directory

After training and quantizing models, MCU inference results are stored in `experiments/<STUDY_NAME>/`.

Each study directory contains:

| File | Purpose |
|------|---------|
| `results.json` | Aggregated results of all MCU-tested trials |
| `<study>-trial-<N>.json` | Per-trial detailed inference output |
| `predictions_trial_<N>.json` | Per-trial per-class predictions |

The `results.json` file is a JSON array of objects, each with:

| Field | Description |
|-------|-------------|
| `trial_number` | Optuna trial number (links to study DB) |
| `float_accuracy` | Accuracy of the original float PyTorch model on validation set (%) |
| `quantized_accuracy` | Accuracy after int8 quantization (on PC) on validation set (%) |
| `quantized_accuracy_int16` | Accuracy after int16 quantization (on PC) on validation set (%) — only present when 16-bit is requested |
| `nr_parameters` | Number of trainable parameters counted from the float ONNX model (sum of elements across all float-dtype initializers, excluding integer shape/index constants) |
| `param_size_bytes` | Parameter size after int8 quantization (bytes) |
| `param_size_bytes_int16` | Parameter size after int16 quantization (bytes) — only present when 16-bit is requested |
| `test_float_accuracy` | Accuracy of the float model on the test set (%) |
| `test_quantized_accuracy` | Accuracy after int8 quantization on the test set (%) |
| `test_quantized_accuracy_int16` | Accuracy after int16 quantization on the test set (%) — only present when 16-bit is requested |
| `mcu_accuracy` | Accuracy on the validation set measured on ESP32-S3 (%) |
| `mcu_latency_ms` | Average inference latency on ESP32-S3 (ms) |
| `mcu_profiling` | Dict of operator-level profiling breakdown (count and total latency in ms per op type) |

When `quantization_methods` includes `strat-kl-tqt`, its metrics are stored under `_strat`-suffixed keys (e.g. `quantized_accuracy_strat`, `test_quantized_accuracy_strat`, `param_size_bytes_strat`), mirroring the `_int16` pattern, so methods can be compared for the same trials.

## Best Models

To list the n most parameter-efficient models (ranked by validation accuracy / nr_parameters) across one or more studies, for a given quantization precision and method:

```bash
conda activate torch-pascal
python -m train.best_models config/kws/arch-mamba1-kws-2.yaml config/har/arch-mamba1-har.yaml --n 10 --bits 8 --strat ptq
```

`--bits` selects 8 or 16, `--strat` selects `ptq` or `tqt`. It reads each study's `experiments/<study>/results.json` and prints a table with trial number, validation accuracy, nr parameters, test accuracy, and accuracy/nr_parameters for both validation and test sets.

## Visualisation

Pareto front comparison and accuracy plots are generated with `train/plot_arch_search.py`.

```bash
conda activate torch-pascal
python -m train.plot_arch_search --plot pareto config/har/arch-mamba1-har.yaml config/har/arch-mamba1-har-bidir.yaml
python -m train.plot_arch_search --plot accuracy config/har/arch-mamba1-har.yaml
python -m train.plot_arch_search --plot accuracy_grid config/kws/arch-mamba1-kws-2.yaml config/kws/arch-mamba1-kws-bidir.yaml config/kws/arch-mamba1-kws-bidir-mul.yaml config/har/arch-mamba1-har.yaml config/har/arch-mamba1-har-bidir.yaml config/har/arch-mamba1-har-bidir-mul.yaml
python -m train.plot_arch_search --plot mcu_pareto config/har/arch-mamba1-har.yaml
```

Four plot types are available:

| `--plot` value | Description |
|----------------|-------------|
| `pareto` | Compares Pareto fronts of multiple experiments on PC latency vs accuracy |
| `accuracy` | Scatter plot of quantized accuracy (y) vs float accuracy (x), with distinct points per quantization method (int8, and int16 / strat-kl-tqt when present) and a y=x dotted reference line |
| `accuracy_grid` | Combined scatter plot: one panel per study in a shared grid with common axis limits/labels, a single shared legend and a y=x reference line in every panel |
| `mcu_pareto` | Two-panel figure: (left) PC Pareto front with ★ markers for MCU-tested trials; (right) MCU accuracy vs MCU latency for those models, annotated with trial numbers |
| `param_accuracy` | Scatter plot of nr of parameters (x, logarithmic axis) vs accuracy (y) from `results.json`, with the N most parameter-efficient models selected from the combined Pareto front (max accuracy, min parameters) across ALL studies highlighted as diamonds and annotated with trial numbers. The models are selected using the **validation-set** accuracy but plotted at their **test-set** accuracy. Use `--n-models N` (default 10) to control how many models are selected in total (not per study). Use `--min-val-acc N` to restrict selection to models scoring strictly above N% on the validation set first. Use `--size` (32 float / 16 / 8 bits, default 8) with `--quantization` (`no` / `percent` / `tqt`, default `percent`) to pick the accuracy metric: `32+no` → float, `16+percent` → int16 PTQ, `8+percent` → int8 PTQ, `8+tqt` → int8 KL-TQT. For HAR studies, reference points from the literature (`har-numbers-literature.md`, see `HAR_LITERATURE_POINTS` in `train/plot_types/param_accuracy.py`) are overlaid as black markers — accuracy values as crosses, F1-scores as pentagons — with the F1/ compute-cost / dataset-average nuances annotated next to each point. |
| `stacked` | Stacked bar chart of MCU operator latency across ALL MCU-tested trials (one bar per trial, segment per operator, sorted by total latency). Normalised to 100% by default; use `--absolute` for summed ms. Total latency annotated above each bar |
| `quantization_loss` | Paper-ready single panel comparing quantization loss per strategy (8-bit PTQ, 8-bit KL-TQT, 16-bit PTQ). Architecture variants (bidirectional add / mul / single direction) are pooled: each chart shows exactly one column per strategy (n=42 for HAR, n=30 for KWS), each with a light semi-transparent raw-point strip (colour + shape per strategy — circle/square/triangle survive grayscale printing) over a semi-transparent boxplot (its internal line marks the median); a black diamond marks the mean (value suffixed `*` when pulled by outliers), with its numeric label placed beside/above the box so it never collides with box edges. A symlog y-axis (linear near 0) keeps the dense near-zero bulk and large outliers readable, with a dashed zero line as the "no change" reference (points below = quantized model more accurate); a horizontal legend sits below the plot. The per-column n= count is merged into the x tick labels and a per-chart title ("HAR:"/"KWS: Quantization Loss by Strategy", inferred from the study title) identifies the dataset. Single-column figure size, exported at 300 DPI PNG plus PDF and SVG vectors. Pass `--ylim LOW HIGH` to fix the y-range (e.g. identical values for KWS and HAR) so companion figures share a consistent axis |

Pass `--bar` together with `--plot accuracy` to draw a grouped bar chart (one bar group per trial, one bar per quantization method) instead of the scatter plot. Without `--bar`, the accuracy plot is unchanged.

All figures are saved to `figures/` as `.png` and `.pdf`.

## Performance

- **HAR inference latency**: ~11.4 ms on ESP32-S3 @ 240 MHz (measured via `esp_timer_get_time()`).

## Agent Instructions

These instructions apply to any AI agent working on this repository.

- **Minimal changes**: Make the smallest possible set of edits to satisfy the task. Do not refactor, reorganise, or beautify code beyond what is strictly required.
- **Avoid comments**: Do not add code comments unless the logic is genuinely non-obvious and a comment is more maintainable than clearer code.
- **Keep docs current**: If you modify the project in a way that makes any part of `AGENTS.md` or `README.md` inaccurate, update the affected files to reflect the new state of the world.

