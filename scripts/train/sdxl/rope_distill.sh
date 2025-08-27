#!/bin/bash

source .venv/bin/activate

accelerate launch train/sdxl/rope_distill.py $@
