import sys
import random
import os

import numpy as np

import torch
import torch.nn as nn

from torchsupport.training.training import MaskedSupervisedTraining
from protsupport.data.proteinnet import DistogramSlice
from protsupport.data.embedding import one_hot_aa

from betafold.predict.distance import DistanceNetwork
from betafold.training.models.resnet import JointNetwork

class DistogramData(DistogramSlice):
  def __getitem__(self, index):
    data = super(DistogramData, self).__getitem__(index)
    distogram = data["distogram"].squeeze()
    dmas = data["mask"].squeeze()
    rama_x, rama_y = data["rama"]
    rama_x = rama_x[0] * self.torsion_bins + rama_x[1]
    rama_y = rama_y[0] * self.torsion_bins + rama_y[1]
    amas_x = dmas[:, 0]
    amas_y = dmas[0, :]
    return data["features"], (
      (distogram, dmas),
      (rama_x, amas_x),
      (rama_y, amas_y)
    )

class MaskedLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss(reduction='none')

  def forward(self, value, target, mask):
    return (self.loss(value, target) * mask.to(torch.float)).sum() / float(mask.sum())

if __name__ == "__main__":
  from pyrosetta import init
  init("-mute all")

  path = sys.argv[1]
  valid_path = sys.argv[2]
  training_args = {
    "device": "cuda:0",
    "batch_size": 32,
    "max_epochs": 1000,
    "network_name": "resnet-220"
  }

  net = nn.DataParallel(JointNetwork(84, 65, 10, depth=55))

  data = DistogramData(
    path, size=64
  )
  valid_data = DistogramData(
    valid_path, size=64
  )

  training = MaskedSupervisedTraining(
    net, data, valid_data, [
      MaskedLoss(),
      MaskedLoss(),
      MaskedLoss()
    ], **training_args,
    valid_callback=lambda x, y, z: x
  )

  net = training.train()
