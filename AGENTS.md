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
# Run each command inside the torch-pascal conda environment (conda run keeps
# the env active per-command, so it works across separate shell invocations)
conda run -n torch-pascal python -m train.main

# Quantize the ONNX model to int8 ESP-DL format
conda run -n torch-pascal python -m train.quantize --model mamba-1 --dataset har
```

Set `MODEL` (e.g., `mamba-1`, `mamba-3`) and `DATASET` (e.g., `har`, `kws`) as environment variables.

### Mamba-Lite micro comparison models

Train the 7 fixed-architecture Mamba-1 models used to compare latency with the
Mamba-Lite micro paper. Each model is a `MambaWrapper` with every Mamba
hyperparameter at its library default (d_state=16, d_conv=4, expand=2) and only
`d_model=64` customised. Input shapes match the paper (KWS: 40 features × 100 frames,
10 classes; HAR: 57 × 10, 6 classes).

```bash
conda run -n torch-pascal python -m train.train_lite [--epochs N] [--model NAME]
```

Exports the 7 ONNX models to `mambalite-micro/` in the repo root. When no CUDA
device is available the CUDA-only Mamba kernels are patched out with CPU
reference implementations so the models can be trained on CPU.

## Architecture Search

Run an Optuna-based hyperparameter search with a pre-defined configuration using Hydra:

```bash
conda run -n torch-pascal python -m train.arch_search config/arch-mamba1-kws.yaml
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

To list the n best models (ranked by a validation-based efficiency metric) across one or more studies, for a given quantization precision and method:

```bash
conda run -n torch-pascal python -m train.best_models config/kws/arch-mamba1-kws-2.yaml config/har/arch-mamba1-har.yaml --n 10 --bits 8 --strat ptq
conda run -n torch-pascal python -m train.best_models config/kws/arch-mamba1-kws-2.yaml config/har/arch-mamba1-har.yaml --n 10 --bits 8 --strat ptq --sort latency
```

`--bits` selects 8 or 16, `--strat` selects `ptq` or `tqt`, and `--sort` selects the ranking metric: `param` (default) ranks by validation accuracy / nr_parameters, `latency` ranks by validation accuracy / mcu_latency_ms (only trials with MCU latency data are considered). It reads each study's `experiments/<study>/results.json` and prints a table with trial number, validation accuracy, cost (nr parameters or MCU latency in ms), test accuracy, and accuracy/cost for both validation and test sets. Selection (accuracy filter and ranking) always uses validation results; test accuracy is reported for reference only.

## Visualisation

Pareto front comparison and accuracy plots are generated with `train/plot_arch_search.py`.

```bash
conda run -n torch-pascal python -m train.plot_arch_search --plot pareto config/har/arch-mamba1-har.yaml config/har/arch-mamba1-har-bidir.yaml
conda run -n torch-pascal python -m train.plot_arch_search --plot accuracy config/har/arch-mamba1-har.yaml
conda run -n torch-pascal python -m train.plot_arch_search --plot accuracy_grid config/kws/arch-mamba1-kws-2.yaml config/kws/arch-mamba1-kws-bidir.yaml config/kws/arch-mamba1-kws-bidir-mul.yaml config/har/arch-mamba1-har.yaml config/har/arch-mamba1-har-bidir.yaml config/har/arch-mamba1-har-bidir-mul.yaml
conda run -n torch-pascal python -m train.plot_arch_search --plot mcu_pareto config/har/arch-mamba1-har.yaml
conda run -n torch-pascal python -m train.plot_arch_search --plot mcu_pareto --size 8 --quantization tqt config/har/arch-mamba1-har.yaml
conda run -n torch-pascal python -m train.plot_arch_search --plot mambalite --title "Mamba-Lite micro comparison" config/har/*
conda run -n torch-pascal python -m train.plot_arch_search --plot val_test_gap --title "Float validation vs test accuracy gap" config/har/* config/kws/*
```

Four plot types are available:

| `--plot` value | Description |
|----------------|-------------|
| `pareto` | Compares Pareto fronts of multiple experiments on PC latency vs accuracy |
| `accuracy` | Scatter plot of quantized accuracy (y) vs float accuracy (x), with distinct points per quantization method (int8, and int16 / strat-kl-tqt when present) and a y=x dotted reference line |
| `accuracy_grid` | Combined scatter plot: one panel per study in a shared grid with common axis limits/labels, a single shared legend and a y=x reference line in every panel |
| `mcu_pareto` | Two-panel figure: (left) PC Pareto front with ★ markers for MCU-tested trials; (right) MCU accuracy vs MCU latency for those models, annotated with trial numbers. Use `--size` (32 / 16 / 8) with `--quantization` (`no` / `percent` / `tqt`, default `percent`) to pick which accuracy variant is plotted in the right panel (float / int16 PTQ / int8 PTQ / int8 KL-TQT), exactly as for `param_accuracy` |
| `param_accuracy` | Scatter plot of nr of parameters (x, logarithmic axis) vs accuracy (y) from `results.json`, with the N most parameter-efficient models selected from the combined Pareto front (max accuracy, min parameters) across ALL studies highlighted as diamonds and annotated with trial numbers. The models are selected using the **validation-set** accuracy but plotted at their **test-set** accuracy. Use `--n-models N` (default 10) to control how many models are selected in total (not per study). Use `--min-val-acc N` to restrict selection to models scoring strictly above N% on the validation set first. Use `--size` (32 float / 16 / 8 bits, default 8) with `--quantization` (`no` / `percent` / `tqt`, default `percent`) to pick the accuracy metric: `32+no` → float, `16+percent` → int16 PTQ, `8+percent` → int8 PTQ, `8+tqt` → int8 KL-TQT. For HAR and KWS studies, reference points from the literature (`har-numbers-literature.md` / `kws-numbers-literature.md`, see `HAR_LITERATURE_POINTS` / `KWS_LITERATURE_POINTS` in `train/plot_types/param_accuracy.py`) are overlaid as black markers — accuracy values as crosses, F1-scores as pentagons — with the F1-score / dataset-average nuances annotated next to each point (compute cost in MACs/FLOPs is not shown). The KWS LMU models report their size in kbits, which is converted to parameters assuming int8 storage (kbits / 8) and noted as "model size, int8 params". |
| `stacked` | Stacked bar chart of MCU operator latency across ALL MCU-tested trials (one bar per trial, segment per operator, sorted by total latency). Normalised to 100% by default; use `--absolute` for summed ms. Total latency annotated above each bar |
| `importance` | Two heatmaps of hyperparameter importance (one per objective: Accuracy/Latency): rows are the studies, columns are the hyperparameters, cells coloured by importance with value labels. Importance values are computed with optuna's PedAnova evaluator on the raw objective value — the same computation as optuna-dashboard — so the numbers match the dashboard. (PedAnova measures importance for *low* target values, so the Accuracy heatmap shows which parameters drive accuracy *down*, e.g. lr causing diverged training.) Rows are ordered by dataset (HAR first) then architecture variant (single → bidir add → bidir mul); the 0–1 colour scale is shared across both heatmaps. Pass all configs at once (e.g. `config/har/* config/kws/*`) to get the full 6-study grid. |
| `quantization_loss` | Paper-ready single panel comparing quantization loss per strategy (8-bit PTQ, 8-bit KL-TQT, 16-bit PTQ). Architecture variants (bidirectional add / mul / single direction) are pooled: each chart shows exactly one column per strategy (n=42 for HAR, n=30 for KWS), each with a light semi-transparent raw-point strip (colour + shape per strategy — circle/square/triangle survive grayscale printing) over a semi-transparent boxplot (its internal line marks the median); a black diamond marks the mean (value suffixed `*` when pulled by outliers), with its numeric label placed beside/above the box so it never collides with box edges. A symlog y-axis (logarithmic with a linear band near zero) keeps the dense near-zero bulk and large outliers readable, with plain round-number tick labels (e.g. 0, 5, 10, 20, 40, 80) instead of log-decade numbers, and a dashed zero line as the "no change" reference (points below = quantized model more accurate); a horizontal legend sits below the plot. The per-column n= count is merged into the x tick labels and a per-chart title ("HAR:"/"KWS: Quantization Loss by Strategy", inferred from the study title) identifies the dataset. Single-column figure size, exported at 300 DPI PNG plus PDF and SVG vectors. Pass `--ylim LOW HIGH` to fix the y-range (e.g. identical values for KWS and HAR) so companion figures share a consistent axis |
| `val_test_gap` | Overfitting / val–test distribution-shift check for the **float** model: one beeswarm + box column per experiment (har-single, har-bidir-add, har-bidir-mul, kws-single, kws-bidir-add, kws-bidir-mul) of the validation-minus-test accuracy gap in percentage points (`float_accuracy - test_float_accuracy`), reusing the quantization-loss figure's raw-point strip + boxplot + mean-diamond styling (shared helpers in `plot_types/common.py`). Positive = validation accuracy above test accuracy; a dashed zero line is the "no gap" reference. Columns are ordered by dataset (HAR first) then variant (single → bidir add → bidir mul), coloured per dataset (blue = HAR, orange = KWS) with per-variant markers; the per-column n= count is merged into the x tick labels. Pass `config/har/* config/kws/*` for the full 6-experiment figure; `--ylim LOW HIGH` fixes the y-range. Exported at 300 DPI PNG plus PDF and SVG |
| `val_contamination` | Bar chart of per-subject contamination of the HAR validation set. Each bar shows, for one of the 30 UCI subjects, the percentage of that subject's validation windows overlapped by a train window (adjacent same-subject windows sharing 50% raw data that were split across the random train/val boundary). It reproduces `load_har_data()`'s 80/20 seed-42 random split (torch. `random_split`) but reads the raw UCI HAR dataset directly; the config path is used only for the title. Only pairs that genuinely share identical raw readings (all 9 inertial axes) count — adjacent pairs at recording boundaries do not share data and are excluded — so contamination is never over-counted. Subjects held out to the UCI test partition (which have no validation windows) are drawn hatched at 0%, with a dashed dataset-mean reference line. Raster PNG at 300 DPI plus PDF |
| `mambalite` | Bar charts comparing this work's Mamba-Lite micro models against the published Mamba-Lite micro reference, per dataset (HAR and KWS). Three metrics are compared, each as one bar chart per dataset: average single-inference latency (ms), peak memory usage (total KB; this work's bars are stacked and coloured by RAM kind — blue internal-RAM on the bottom, purple PSRAM on top — while the Mamba-Lite reference bar is a single internal-RAM bar since it uses only internal RAM) and flash model storage (bytes; converted from the KB flash total). Successful runs are parsed directly from `experiments/mambalite-micro/*.output` (a run counts if the log has an `INFERENCE_OK` line, a latency figure and the memory-summary total row); failed runs are skipped. The Mamba-Lite reference bar is hatched. Raster PNG at 300 DPI plus PDF |

Pass `--bar` together with `--plot accuracy` to draw a grouped bar chart (one bar group per trial, one bar per quantization method) instead of the scatter plot. Without `--bar`, the accuracy plot is unchanged.

All figures are saved to `figures/` as `.png` and `.pdf`.

## Performance

- **HAR inference latency**: ~11.4 ms on ESP32-S3 @ 240 MHz (measured via `esp_timer_get_time()`).

## Agent Instructions

These instructions apply to any AI agent working on this repository.

- **Minimal changes**: Make the smallest possible set of edits to satisfy the task. Do not refactor, reorganise, or beautify code beyond what is strictly required.
- **Avoid comments**: Do not add code comments unless the logic is genuinely non-obvious and a comment is more maintainable than clearer code.
- **Keep docs current**: If you modify the project in a way that makes any part of `AGENTS.md` or `README.md` inaccurate, update the affected files to reflect the new state of the world.

