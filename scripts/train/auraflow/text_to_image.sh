#!/bin/bash

source .venv/bin/activate

accelerate launch train/auraflow/text_to_image.py $@
