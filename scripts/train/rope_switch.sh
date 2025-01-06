#!/bin/bash

source .venv/bin/activate

accelerate launch train/rope_switch.py $@

