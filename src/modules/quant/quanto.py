from typing import Literal

import torch
import torch.nn as nn

import torchao as ao
from torchao.float8 import float8_linear as ao_fp8
import bitsandbytes as bnb
import optimum.quanto as quanto


class QuantoLinear(quanto.nn.QLinear):
    pass
