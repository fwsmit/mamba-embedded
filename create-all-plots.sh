#!/bin/bash

python -m train.plot_arch_search --plot mcu_pareto config/har/arch-mamba1-har.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba single direction (KWS)" config/kws/arch-mamba1-kws-2.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (add) (KWS)" config/kws/arch-mamba1-kws-bidir.yaml
python -m train.plot_arch_search --plot accuracy --title "Quantization mamba bidirectional (mul) (KWS)" config/kws/arch-mamba1-kws-bidir-mul.yaml
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (HAR)" config/har/*
python -m train.plot_arch_search --plot pareto --title "Pareto front comparison (KWS)" config/kws/*
