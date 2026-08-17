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
python -m train.plot_arch_search --view --plot scatter --x-field nr_parameters --y-field test_quantized_accuracy --x-label "Parameters (K)" --y-label "8-bit quantized accuracy (%)" --title "Parameters vs accuracy (KWS)" config/kws/* --show
python -m train.plot_arch_search --view --plot scatter --x-field nr_parameters --y-field test_quantized_accuracy --x-label "Parameters (K)" --y-label "8-bit quantized accuracy (%)" --title "Parameters vs accuracy (HAR)" config/har/* --show

# Quantization loss
python -m train.plot_arch_search --plot quantization_loss --title "Quantization loss per strategy (KWS)" config/kws/*
python -m train.plot_arch_search --plot quantization_loss --title "Quantization loss per strategy (HAR)" config/har/*

# Profiling plot
python -m train.plot_arch_search --plot profiling --trial 18 config/har/arch-mamba1-har-bidir-mul.yaml

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
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (KWS)" config/kws/*

# Compare parameter count with other studies
python -m train.plot_arch_search --plot param_accuracy --size 8 --quantization tqt --n-models 5 --min-val-acc 85 config/har/*
python -m train.plot_arch_search --plot param_accuracy --size 8 --quantization tqt --n-models 5 --min-val-acc 85 config/kws/*
