# pytorch training template

Key frameworks:

- [PyTorch](https://pytorch.org/)
- [Lightning Fabric](https://lightning.ai/docs/fabric/2.4.0/)
- [HuggingFace Hub](https://huggingface.co/)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [Wandb](https://wandb.ai/)

## Setup

```bash
uv sync
```

### Extra libraries

- `optimizers`: some popular optimizers (e.g. `schedulefree`)
- `quant`: quantization tools (e.g. `bitsandbytes`)

To use all:

```bash
uv sync --extra optimizers --extra quant
```

## Train

Distributed training is not tested yet.

Simplest example:

```bash
fabric run \
    ./main.py \
    --config ./configs/mnist.yaml
```

With options:

```bash
fabric run \
    --accelerator cuda \
    --precision bf16-mixed \
    ./main.py \
    --config ./configs/mnist.yaml
```


