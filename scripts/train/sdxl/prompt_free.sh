#!/bin/bash

source .venv/bin/activate

accelerate launch train/sdxl/prompt_free.py $@
