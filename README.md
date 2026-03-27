
# Building

## Install dependencies

This project needs the following dependencies:

- espup (rust toolchain for ESP devices)
- pytorch
- mamba-ssm

## Train the model

```shell
python model/train.py
```

This exports the model to onnx and places it in the src/model directory

## Create test vector

```shell
python utils/mnist_to_burn_tensor.py
```

This exports the test

## Run rust project

```shell
cargo run --release
```

# Environment variables

`MODEL`: Select which model to use. Choices from the following models: `mamba-1` `mamba-5`.
