#!/bin/bash

source .venv/bin/activate

accelerate launch main.py $@

