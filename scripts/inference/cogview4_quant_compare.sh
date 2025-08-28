#!/bin/bash

source .venv/bin/activate

# 使用する quant_type の種類を定義（必要に応じて追加・変更してください）
text_encoders=("fp8_e4m3fn" "bnb_int8" "bnb_fp4" "bnb_nf4" "quanto_int4" "quanto_int8" "ao_nf4" "ao_fp8")
denoisers=("quanto_int4")
skip_offload=("false" "true")

# 各組み合わせで Python スクリプトを実行
for te in "${text_encoders[@]}"; do
  for dn in "${denoisers[@]}"; do
    for skip in "${skip_offload[@]}"; do
      echo "Running with text_encoder=${te}, denoiser=${dn}, skip_offload=${skip}"
      if [ "$skip" == "true" ]; then
        python ./tools/cogview4_quant_compare.py --text_encoder "${te}" --denoiser "${dn}" --skip_offload
      else
        python ./tools/cogview4_quant_compare.py --text_encoder "${te}" --denoiser "${dn}"
      fi
    done
  done
done