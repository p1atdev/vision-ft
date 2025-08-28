#!/bin/bash

source .venv/bin/activate

# 使用する quant_type の種類を定義（必要に応じて追加・変更してください）
text_encoders=("bf16" "fp8_e4m3fn" "bnb_int8" "bnb_fp4" "bnb_nf4" "quanto_int4" "quanto_int8" "ao_nf4" "ao_fp8")
denoisers=("bf16" "fp8_e4m3fn" "bnb_int8" "bnb_fp4" "bnb_nf4" "quanto_int4" "quanto_int8" "ao_nf4" "ao_fp8")

# 各組み合わせで Python スクリプトを実行
for te in "${text_encoders[@]}"; do
  for dn in "${denoisers[@]}"; do
    echo "===== text_encoder=${te}, denoiser=${dn}"
    echo "- offload=False"
    python ./tools/snapshot_max_memory.py "./output/text-encoder-${te}_denoiser-${dn}_offload-False.pickle"
    echo "- offload=True"
    python ./tools/snapshot_max_memory.py "./output/text-encoder-${te}_denoiser-${dn}_offload-True.pickle"
  done
done