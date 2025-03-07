#!/bin/bash

source .venv/bin/activate

accelerate launch train/auraflow/vae_encode_migration.py $@
