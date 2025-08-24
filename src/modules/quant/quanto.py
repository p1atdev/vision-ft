import torch
import torch.nn as nn

try:
    from optimum.quanto.nn import QLinear
except ImportError:
    QLinear = nn.Linear


# just wrap the original QLinear class
class QuantoLinear(QLinear):
    pass
