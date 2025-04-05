#!/bin/bash

source .venv/bin/activate

accelerate launch train/sdxl/text_to_image.py $@
