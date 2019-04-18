import torch
from torch import nn
from torch.nn import functional as func

from betafold.predict.residual import ResidualStack1d

class SecondaryMLP(nn.Module):
  def __init__(self, in_size, hidden_sizes=[64, 32, 16], activation=func.elu_):
    super(SecondaryMLP, self).__init__()
    sizes = [in_size] + hidden_sizes + [8]
    self.activation = activation
    self.bn = nn.ModuleList([
      nn.BatchNorm1d(sizes[idx])
      for idx in range(len(sizes) - 1)
    ])
    self.projections = nn.ModuleList([
      nn.Conv1d(sizes[idx], sizes[idx + 1], 1)
      for idx in range(len(sizes) - 1)
    ])

  def forward(self, inputs):
    out = inputs
    for bn, block in zip(self.bn, self.projections):
      out = block(bn(self.activation(out)))
    return out

class SecondaryStack(nn.Module):
  def __init__(self, in_size, hidden_size=64, depth=2):
    super(SecondaryStack, self).__init__()
    self.stack = ResidualStack1d(
      size=in_size, hidden_size=hidden_size, depth=depth
    )

  def forward(self, inputs):
    return self.stack(inputs)

class SecondaryNetwork(nn.Module):
  def __init__(self, in_size, mode='mlp', depth=2,
               hidden_sizes=[64, 32, 16],
               activation=func.elu_):
    super(SecondaryNetwork, self).__init__()
    assert mode in ('mlp', 'stack')
    self.preprocess = lambda x: x
    if mode == 'stack':
      self.preprocess = SecondaryStack(
        in_size, hidden_size=in_size // 2, depth=depth
      )
    self.predict = SecondaryMLP(
      in_size, hidden_sizes=hidden_sizes,
      activation=activation
    )

  def forward(self, inputs):
    return self.predict(self.preprocess(inputs))
