
# Building

## Install dependencies

This project needs the following dependencies:

- espup (rust toolchain for ESP devices)
- pytorch
- mamba-ssm

## Train the model

```shell
python -m train.main
```

This exports the model to onnx and places it in the src/model directory

## Create test vectors

```shell
python -m train.har_to_burn_tensor
python -m train.kws_to_burn_tensor
```

This exports the test

## Run rust project

```shell
cargo run --release
```

# Environment variables

`MODEL`: Select which model to use. Choices from the following models: `mamba-1` `mamba-3`.
`DATASET`: Select which model to use. Choices from the following models: `kws` `har`.
