#!/bin/bash

MODEL=eva02-large
BATCH_SIZE=32
FORMAT=json

tagger v3 \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --format $FORMAT \
    $@


