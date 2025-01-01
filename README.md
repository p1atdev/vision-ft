# AuraFlow finetuning scripts

WIP

## Features

(planned)

- [NF4 model](https://huggingface.co/p1atdev/AuraFlow-v0.3-bnb-nf4) loading 
- QLoRA with bitsandbytes
- Aspect ratio bucketing

## Setup

```bash
uv sync
```

## Train

Distributed training is not tested yet.

Simplest example:

```bash
accelerate launch \
    ./main.py \
    --config ./configs/lora.yaml
```

> [!WARNING]
> This may be changed in the future.


## References

- https://github.com/kohya-ss/sd-scripts
  - Heavily inspired by this repository.

- https://github.com/cloneofsimo/minRF
  - Model implementation

- https://github.com/microsoft/LoRA
- https://github.com/huggingface/peft
  - Peft logic and implementation

- https://github.com/huggingface/diffusers
- https://github.com/Lightning-AI/pytorch-lightning
  - Traning cycle and API design

- https://github.com/NovelAI/novelai-aspect-ratio-bucketing
  - Aspect ratio bucketing

- https://github.com/bitsandbytes-foundation/bitsandbytes
  - Quantization


