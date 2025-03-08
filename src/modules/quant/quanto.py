import torch
import torch.nn as nn

import optimum.quanto as quanto


# just wrap the original QLinear class
class QuantoLinear(quanto.nn.QLinear):
    pass
