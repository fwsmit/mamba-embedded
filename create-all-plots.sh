#!/bin/bash
set -e

# Ensure we're in the repo root (the directory this script lives in)
cd "$(dirname "$0")"

# Remove old plots before generating new ones
rm -f figures/*.png figures/*.pdf

python -m train.plot_arch_search --plot mcu_pareto config/kws/*
python -m train.plot_arch_search --plot mcu_pareto config/har/*
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba single direction (KWS)" config/kws/arch-mamba1-kws-2.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (add) (KWS)" config/kws/arch-mamba1-kws-bidir.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (mul) (KWS)" config/kws/arch-mamba1-kws-bidir-mul.yaml
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (HAR)" config/har/*
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (KWS)" config/kws/*
