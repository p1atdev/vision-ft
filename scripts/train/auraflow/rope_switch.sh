#!/bin/bash

source .venv/bin/activate

accelerate launch train/auraflow/rope_switch.py $@
