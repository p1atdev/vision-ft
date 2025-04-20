#!/bin/bash

source .venv/bin/activate

accelerate launch train/sdxl/style_tokenizer.py $@
