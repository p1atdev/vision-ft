#!/bin/bash

source .venv/bin/activate

text_encoders=("bf16" "fp8_e4m3fn" "bnb_int8" "bnb_fp4" "bnb_nf4" "quanto_int4" "quanto_int8" "ao_nf4" "ao_fp8")
denoisers=("bf16" "fp8_e4m3fn" "bnb_int8" "bnb_fp4" "bnb_nf4" "quanto_int4" "quanto_int8" "ao_nf4" "ao_fp8")
skip_offload=("false" "true")

for te in "${text_encoders[@]}"; do
    for dn in "${denoisers[@]}"; do
        for skip in "${skip_offload[@]}"; do
            echo "Running with text_encoder=${te}, denoiser=${dn}, skip_offload=${skip}"
            if [ "$skip" == "true" ]; then
                python ./tools/bench/sdxl_quant.py --text_encoder "${te}" --denoiser "${dn}" --skip_offload $@
            else
                python ./tools/bench/sdxl_quant.py --text_encoder "${te}" --denoiser "${dn}" $@
            fi
        done
    done
done
