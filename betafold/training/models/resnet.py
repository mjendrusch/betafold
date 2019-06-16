import torch
import torch.nn as nn
import torch.nn.functional as func

from betafold.predict.residual import ResidualStack2d

from betafold.predict.distance import DistanceNetwork
from betafold.predict.angles import TorsionMLP

class JointNetwork(nn.Module):
  def __init__(self, in_size, out_size, angle_size, size=128, depth=55):
    super(JointNetwork, self).__init__()
    self.preprocessor = nn.Conv2d(in_size, size, 1)
    self.stack = ResidualStack2d(
      size=size, hidden_size=size // 2, depth=depth
    )
    self.postprocessor = nn.Conv2d(size, out_size, 1)
    self.angle_processor = nn.Conv1d(size, angle_size * angle_size, 1)

  def forward(self, inputs):
    out = self.preprocessor(inputs)
    out = self.stack(out)
    distance = self.postprocessor(out)
    angle_inputs_x = out.mean(dim=3)
    angle_inputs_y = out.mean(dim=2)
    angles_x = self.angle_processor(angle_inputs_x)
    angles_y = self.angle_processor(angle_inputs_y)
    return distance, angles_x, angles_y, out
