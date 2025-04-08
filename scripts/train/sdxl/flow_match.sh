#!/bin/bash

source .venv/bin/activate

accelerate launch train/sdxl/flow_match.py $@
