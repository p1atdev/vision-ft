#!/bin/bash

source .venv/bin/activate

accelerate launch ./tools/quantize_model.py $@