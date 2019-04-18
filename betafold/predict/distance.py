import torch
from torch import nn
from torch.nn import functional as func

from betafold.predict.residual import ResidualStack2d

class DistanceNetwork(nn.Module):
  def __init__(self, in_size, out_size, size=128, depth=55):
    super(DistanceNetwork, self).__init__()
    self.preprocessor = nn.Conv2d(in_size, size, 1)
    self.stack = ResidualStack2d(
      size=size, hidden_size=size // 2, depth=depth
    )
    self.postprocessor = nn.Conv2d(size, out_size, 1)

  def forward(self, inputs):
    out = self.preprocessor(inputs)
    out = self.stack(out)
    distance = self.postprocessor(out)
    return distance, out
