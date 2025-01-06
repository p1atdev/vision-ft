#!/bin/bash

source .venv/bin/activate

accelerate launch train/text_to_image.py $@

