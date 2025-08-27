#!/bin/bash

source .venv/bin/activate

python ./tools/data/dl_konachan.py \
    $@
