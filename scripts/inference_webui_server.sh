#!/bin/bash

source .venv/bin/activate

CHECKPOINT_PATH=./models/aura_flow_0.3.bnb_nf4.safetensors

python ./tools/inference_server.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --port 8123 
