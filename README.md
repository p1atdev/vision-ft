# Vision model finetuning scripts

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/p1atdev/vision-ft)

WIP

## Features

- NF4 model loading 
- QLoRA with bitsandbytes
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) support
- Aspect ratio bucketing

### Models

- [x] [SDXL](./src/models/sdxl)
- [x] [AuraFlow](./src/models/auraflow)
- [x] [Lumina Image 2.0](./src/models/lumina2)
- [ ] Flux & Flex (WIP)
- [ ] CogView4 (WIP)
- [ ] FractalGen (TODO)
- [ ] Wan 2.1 (TODO)

## Setup

```bash
uv sync --extra build
uv sync --all-extras
# uv sync --extra compile --extra flash-attn --extra xformers --extra triton
# at least compile is required for training
```


## References

- https://github.com/kohya-ss/sd-scripts
  - Heavily inspired by this repository.

- https://github.com/cloneofsimo/minRF
- https://github.com/huggingface/diffusers
- https://github.com/Stability-AI/sd3-ref
- https://github.com/black-forest-labs/flux
- https://github.com/Alpha-VLLM/Lumina-Image-2.0
  - Model implementation

- https://github.com/microsoft/LoRA
- https://github.com/huggingface/peft
  - Peft logic and implementation

- https://github.com/huggingface/diffusers
- https://github.com/Lightning-AI/pytorch-lightning
- https://github.com/ostris/ai-toolkit
  - Traning cycle and API design

- https://github.com/NovelAI/novelai-aspect-ratio-bucketing
  - Aspect ratio bucketing

- https://github.com/bitsandbytes-foundation/bitsandbytes
  - Quantization


