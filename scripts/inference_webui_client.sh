#!/bin/bash

source .venv/bin/activate

python ./tools/inference_client.py \
    --server http://localhost:8123 \
    --host 0.0.0.0 

wait