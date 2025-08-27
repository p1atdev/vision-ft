#!/bin/bash

source .venv/bin/activate

python ./tools/data/dl_safebooru.py \
    --output "./data/sfw_0.1k" \
    --limit 100