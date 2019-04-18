import torch
from torch import nn
from torch.nn import functional as func

class DilatedResidualBlockNd(nn.Module):
  def __init__(self, size, hidden_size, N=1,
                dilation=1, activation=func.elu_):
    super(DilatedResidualBlockNd, self).__init__()
    batch_norm = eval(f"nn.BatchNorm{N}d")
    conv = eval(f"nn.Conv{N}d")
    self.bn = nn.ModuleList([
      batch_norm(size),
      batch_norm(hidden_size),
      batch_norm(hidden_size)
    ])
    self.blocks = nn.ModuleList([
      conv(size, hidden_size, 1),
      conv(hidden_size, hidden_size, 3,
            dilation=dilation, padding=dilation),
      conv(hidden_size, size, 1)
    ])
    self.activation = activation

  def forward(self, inputs):
    out = inputs
    for bn, block in zip(self.bn, self.blocks):
      out = block(bn(self.activation(out)))
    return out + inputs

class DilatedResidualBlock1d(DilatedResidualBlockNd):
  def __init__(self, size, hidden_size, dilation=1, activation=func.elu_):
    super(DilatedResidualBlock1d, self).__init__(
      size, hidden_size, N=1, dilation=dilation, activation=activation
    )

class DilatedResidualBlock2d(DilatedResidualBlockNd):
  def __init__(self, size, hidden_size, dilation=1, activation=func.elu_):
    super(DilatedResidualBlock2d, self).__init__(
      size, hidden_size, N=2, dilation=dilation, activation=activation
    )

class ResidualStackNd(nn.Module):
  def __init__(self, size=128, hidden_size=64, N=1,
                depth=55, dilations=[1, 2, 4, 8]):
    super(ResidualStackNd, self).__init__()
    res_block = eval(f"DilatedResidualBlock{N}d")
    self.blocks = nn.ModuleList([
      res_block(
        size, hidden_size,
        dilations[idx % len(dilations)]
      )
      for idx in range(depth * len(dilations))
    ])

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out

class ResidualStack1d(ResidualStackNd):
  def __init__(self, size=128, hidden_size=64, N=1,
               depth=55, dilations=[1, 2, 4, 8]):
    super(ResidualStack1d, self).__init__(
      size=size, hidden_size=hidden_size, N=1, depth=depth, dilations=dilations
    )

class ResidualStack2d(ResidualStackNd):
  def __init__(self, size=128, hidden_size=64, N=1,
               depth=55, dilations=[1, 2, 4, 8]):
    super(ResidualStack2d, self).__init__(
      size=size, hidden_size=hidden_size, N=1, depth=depth, dilations=dilations
    )
