#!/bin/bash

source .venv/bin/activate

accelerate launch train/auraflow/shortcut.py $@
