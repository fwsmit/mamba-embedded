#!/bin/bash

python -m train.arch_search config/har/* config/kws/*
python -m train.top_models config/har/* config/kws/*
