#!/bin/bash

source .venv/bin/activate

accelerate launch train/rope_migration.py $@

