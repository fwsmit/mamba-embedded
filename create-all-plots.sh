#!/bin/bash
set -e

# Ensure we're in the repo root (the directory this script lives in)
cd "$(dirname "$0")"

# Remove old plots before generating new ones
rm -f figures/*.png figures/pdf/*.pdf

# Quantization loss
python -m train.plot_arch_search --plot quantization_loss config/kws/*
python -m train.plot_arch_search --plot quantization_loss config/har/*

# Profiling plot
python train/plot_arch_search.py --plot profiling --trial 18 config/har/arch-mamba1-har-bidir-mul.yaml

python -m train.plot_arch_search --plot latency config/kws/arch-mamba1-kws-2.yaml
# python -m train.plot_arch_search --plot latency config/kws/arch-mamba1-kws-bidir.yaml
# python -m train.plot_arch_search --plot latency config/kws/arch-mamba1-kws-bidir-mul.yaml
# python -m train.plot_arch_search --plot latency config/har/arch-mamba1-har.yaml
# python -m train.plot_arch_search --plot latency config/har/arch-mamba1-har-bidir.yaml
# python -m train.plot_arch_search --plot latency config/har/arch-mamba1-har-bidir-mul.yaml

python -m train.plot_arch_search --plot mcu_pareto config/kws/*
python -m train.plot_arch_search --plot mcu_pareto config/har/*
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba single direction (KWS)" config/kws/arch-mamba1-kws-2.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (add) (KWS)" config/kws/arch-mamba1-kws-bidir.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (mul) (KWS)" config/kws/arch-mamba1-kws-bidir-mul.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba single direction (HAR)" config/har/arch-mamba1-har.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (add) (HAR)" config/har/arch-mamba1-har-bidir.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (mul) (HAR)" config/har/arch-mamba1-har-bidir-mul.yaml
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (HAR)" config/har/*
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (KWS)" config/kws/*
