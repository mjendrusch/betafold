import torch
from torch import nn
from torch.nn import functional as func

from betafold.predict.residual import ResidualStack1d

class TorsionMLP(nn.Module):
  def __init__(self, in_size, hidden_sizes=[256, 512, 1024],
               phi_bins=36, psi_bins=36, activation=func.elu_):
    super(TorsionMLP, self).__init__()
    self.phi = phi_bins
    self.psi = psi_bins
    sizes = [in_size] + hidden_sizes + [phi_bins * psi_bins]
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

class TorsionStack(nn.Module):
  def __init__(self, in_size, hidden_size=64, depth=2):
    super(TorsionStack, self).__init__()
    self.stack = ResidualStack1d(
      size=in_size, hidden_size=hidden_size, depth=depth
    )

  def forward(self, inputs):
    return self.stack(inputs)

class TorsionNetwork(nn.Module):
  def __init__(self, in_size, mode='mlp', depth=2, phi_bins=36, psi_bins=36,
               hidden_sizes=[256, 512, 1024], activation=func.elu_):
    super(TorsionNetwork, self).__init__()
    assert mode in ('mlp', 'stack')
    self.preprocess = lambda x: x
    if mode == 'stack':
      self.preprocess = TorsionStack(
        in_size, hidden_size=in_size // 2, depth=depth
      )
    self.predict = TorsionMLP(
      in_size, hidden_sizes=hidden_sizes,
      phi_bins=phi_bins, psi_bins=psi_bins,
      activation=activation
    )

  def forward(self, inputs):
    out = self.predict(self.preprocess(inputs))
    out.reshape(
      inputs.size(0),
      self.predict.phi,
      self.predict.psi,
      inputs.size(-1)
    )
    return out
