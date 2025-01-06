#!/bin/bash

source .venv/bin/activate

CONFIG_PATH=./projects/rope/rope_migration.yml

python ./tools/inference_server.py \
    --config_path $CONFIG_PATH \
    --port 8123 \
    $@
