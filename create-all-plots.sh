#!/bin/bash
set -e

# Ensure we're in the repo root (the directory this script lives in)
cd "$(dirname "$0")"

# Remove old plots before generating new ones
rm -f figures/*.png figures/pdf/*.pdf

# Investigate quantization loss
python -m train.plot_arch_search --plot scatter --x-field param_size_bytes --y-field quantization_loss_int8 --x-label "Parameter Size (bytes, int8 quantized)" --y-label "Quantization loss (int8)" --title "Quantization loss vs parameter size (HAR)" config/har/*
python -m train.plot_arch_search --plot scatter --x-field param_size_bytes --y-field quantization_loss_int8 --x-label "Parameter Size (bytes, int8 quantized)" --y-label "Quantization loss (int8)" --title "Quantization loss vs parameter size (KWS)" config/kws/*
python -m train.plot_arch_search --plot scatter --x-field lr --y-field quantization_loss_int8 --x-label "Learning rate" --y-label "Quantization loss (int8)" --title "Quantization loss vs learning rate" config/har/arch-mamba1-har.yaml

# Scatter: parameter size vs MCU latency
python -m train.plot_arch_search --plot scatter --x-field param_size_bytes --y-field mcu_latency_ms --x-label "Parameter Size (bytes, int8 quantized)" --y-label "Latency on MCU (ms)" --title "Parameter size vs MCU latency (HAR)" config/har/*
python -m train.plot_arch_search --plot scatter --x-field param_size_bytes --y-field mcu_latency_ms --x-label "Parameter Size (bytes, int8 quantized)" --y-label "Latency on MCU (ms)" --title "Parameter size vs MCU latency (KWS)" config/kws/*

# Scatter: parameter size vs MCU accuracy
python -m train.plot_arch_search --plot scatter --x-field param_size_bytes --y-field test_quantized_accuracy --x-label "Parameter Size (bytes, int8 quantized)" --y-label "Accuracy on MCU (%)" --title "Parameter size vs MCU accuracy (HAR)" config/har/*
python -m train.plot_arch_search --plot scatter --x-field param_size_bytes --y-field test_quantized_accuracy --x-label "Parameter Size (bytes, int8 quantized)" --y-label "Accuracy on MCU (%)" --title "Parameter size vs MCU accuracy (KWS)" config/kws/*

# Scatter: nr parameters vs accuracy
python -m train.plot_arch_search --plot scatter --x-field nr_parameters --y-field test_quantized_accuracy --x-label "Parameters (K)" --y-label "8-bit quantized accuracy (%)" --title "Parameters vs accuracy (KWS)" config/kws/*
python -m train.plot_arch_search --plot scatter --x-field nr_parameters --y-field test_quantized_accuracy --x-label "Parameters (K)" --y-label "8-bit quantized accuracy (%)" --title "Parameters vs accuracy (HAR)" config/har/*

# Scatter: overfitting check (validation minus test accuracy vs trial number)
python -m train.plot_arch_search --plot scatter --x-field trial_number --y-field val_test_float_gap --x-label "Trial number" --y-label "Validation minus test accuracy (percentage points)" --y-zero-line --title "Overfitting across HPO trials" config/kws/* config/har/*

# Quantization loss (shared y-limits keep the zero line at the same height
# across both plots; negative LOW keeps zero-loss-improved trials visible)
python -m train.plot_arch_search --plot quantization_loss --ylim -1 100 --title "Quantization loss per strategy (KWS)" config/kws/*
python -m train.plot_arch_search --plot quantization_loss --ylim -1 100 --title "Quantization loss per strategy (HAR)" config/har/*

# Profiling plot
python -m train.plot_arch_search --plot profiling --trial 18 config/har/arch-mamba1-har-bidir-mul.yaml

# Confusion matrix of a specific trial's MCU predictions (validation set)
python -m train.plot_arch_search --plot confusion --trial 18 config/har/arch-mamba1-har-bidir-mul.yaml
python -m train.plot_arch_search --plot confusion --trial 8 config/kws/arch-mamba1-kws-bidir.yaml

# python -m train.plot_arch_search --plot latency config/kws/* --title "PC latency vs MCU latency (KWS)"
# python -m train.plot_arch_search --plot latency config/har/* --title "PC latency vs MCU latency (HAR)"
python -m train.plot_arch_search --plot latency config/kws/arch-mamba1-kws-2.yaml
python -m train.plot_arch_search --plot latency config/kws/arch-mamba1-kws-bidir.yaml
python -m train.plot_arch_search --plot latency config/kws/arch-mamba1-kws-bidir-mul.yaml
python -m train.plot_arch_search --plot latency config/har/arch-mamba1-har.yaml
python -m train.plot_arch_search --plot latency config/har/arch-mamba1-har-bidir.yaml
python -m train.plot_arch_search --plot latency config/har/arch-mamba1-har-bidir-mul.yaml

python -m train.plot_arch_search --plot latency --use-param-size config/kws/arch-mamba1-kws-2.yaml
python -m train.plot_arch_search --plot latency --use-param-size config/kws/arch-mamba1-kws-bidir.yaml
python -m train.plot_arch_search --plot latency --use-param-size config/kws/arch-mamba1-kws-bidir-mul.yaml
python -m train.plot_arch_search --plot latency --use-param-size config/har/arch-mamba1-har.yaml
python -m train.plot_arch_search --plot latency --use-param-size config/har/arch-mamba1-har-bidir.yaml
python -m train.plot_arch_search --plot latency --use-param-size config/har/arch-mamba1-har-bidir-mul.yaml

python -m train.plot_arch_search --plot mcu_pareto config/kws/* --size 16 --quantization percent
python -m train.plot_arch_search --plot mcu_pareto config/har/* --size 8 --quantization tqt
python -m train.plot_arch_search --plot accuracy --bar --title "Quantization Mamba single direction (KWS)" config/kws/arch-mamba1-kws-2.yaml
python -m train.plot_arch_search --plot accuracy --bar --title "Quantization Mamba bidirectional (add) (KWS)" config/kws/arch-mamba1-kws-bidir.yaml
python -m train.plot_arch_search --plot accuracy --bar --title "Quantization Mamba bidirectional (mul) (KWS)" config/kws/arch-mamba1-kws-bidir-mul.yaml
python -m train.plot_arch_search --plot accuracy --bar --title "Quantization Mamba single direction (HAR)" config/har/arch-mamba1-har.yaml
python -m train.plot_arch_search --plot accuracy --bar --title "Quantization Mamba bidirectional (add) (HAR)" config/har/arch-mamba1-har-bidir.yaml
python -m train.plot_arch_search --plot accuracy --bar --title "Quantization Mamba bidirectional (mul) (HAR)" config/har/arch-mamba1-har-bidir-mul.yaml
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (HAR)" config/har/*
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (KWS)" --par-acc-top auto config/kws/* config/kws-multi-layer/*

python -m train.plot_arch_search --plot pareto --title "Pareto front comparison multi-layer (KWS)" --ylim 90 97 config/kws/arch-mamba1-kws-bidir-mul.yaml config/kws-multi-layer/* --show
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison Mamba-3 (KWS)" --ylim 88 96 --xlim 0 1250 config/kws/arch-mamba1-kws-2.yaml config/mamba-3/arch-mamba3-kws-bidir-mul.yaml config/kws/arch-mamba1-kws-bidir-mul.yaml

# Per-subject contamination of the HAR validation split (reads the raw
# UCI HAR dataset directly; config only resolves the title)
python -m train.plot_arch_search --plot val_contamination --title "HAR validation contamination per subject" config/har/arch-mamba1-har.yaml

# Compare parameter count with other studies
python -m train.plot_arch_search --plot param_accuracy --size 8 --quantization tqt --n-models 5 --min-val-acc 85 config/har/*
python -m train.plot_arch_search --plot param_accuracy --size 16 --quantization percent --n-models 5 --min-val-acc 85 config/kws/*
# python -m train.plot_arch_search --plot param_accuracy --size 8 --quantization tqt --n-models 5 --min-val-acc 85 config/kws/*

# Hyperparameter importance
python -m train.plot_arch_search --plot importance config/har/* config/kws/* config/kws-multi-layer/*

# Mamba-Lite micro comparison (latency / peak RAM / flash vs the paper's
# reference; parsed from experiments/mambalite-micro/*.output)
python -m train.plot_arch_search --plot mambalite --title "Mamba-Lite micro comparison" config/har/*

# Overfitting test
# Float validation minus test accuracy per experiment (overfitting /
# val-test distribution shift check, one beeswarm + box column per study)
python -m train.plot_arch_search --plot val_test_gap --title "Float validation vs test accuracy gap" config/har/* config/kws/*

# python -m train.plot_arch_search --plot scatter \
#   --x-field number --y-field val_test_float_gap \
#   --x-label "Trial number" --y-label "Validation - test accuracy (pp, float32)" \
#   --title "Overfitting check: val - test accuracy by trial (KWS, float32)" \
#   --y-zero-line --show \
#   config/kws/*
#
# python -m train.plot_arch_search --plot scatter \
#   --x-field number --y-field val_test_float_gap \
#   --x-label "Trial number" --y-label "Validation - test accuracy (pp, float32)" \
#   --title "Overfitting check: val - test accuracy by trial (HAR, float32)" \
#   --y-zero-line --show \
#   config/har/*
#
# python -m train.plot_arch_search --plot scatter \
#   --x-field number --y-field val_test_float_gap \
#   --x-label "Trial number" --y-label "Validation - test accuracy (pp, float32)" \
#   --title "Overfitting check: val - test accuracy by trial (HAR bidir-mul, float32)" \
#   --y-zero-line --show \
#   config/har/arch-mamba1-har-bidir-mul.yaml
#
# python -m train.plot_arch_search --plot scatter \
#   --x-field number --y-field val_test_float_gap \
#   --x-label "Trial number" --y-label "Validation - test accuracy (pp, float32)" \
#   --title "Overfitting check: val - test accuracy by trial (all, float32)" \
#   --y-zero-line --show \
#   config/kws/* config/har/*
