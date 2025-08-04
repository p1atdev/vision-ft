#!/bin/bash

source .venv/bin/activate

accelerate launch train/lumina2/text_to_image.py $@
