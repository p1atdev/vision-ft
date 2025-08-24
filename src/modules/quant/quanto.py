import torch
import torch.nn as nn

try:
    import optimum.quanto as quanto
except ImportError:
    quanto = None


# just wrap the original QLinear class
class QuantoLinear(quanto.nn.QLinear):
    pass
