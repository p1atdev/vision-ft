#!/bin/bash

source .venv/bin/activate

accelerate launch ./tools/inference_cli.py $@