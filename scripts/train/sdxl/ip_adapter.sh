#!/bin/bash

source .venv/bin/activate

accelerate launch train/sdxl/ip_adapter.py $@
