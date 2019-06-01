import sys
import random
import os

import numpy as np

import torch

from torchsupport.training.vae import IndependentConditionalVAETraining, ConditionalVAETraining
from protsupport.data.data import PDBFragment, PDBNumeric
from protsupport.data.embedding import one_hot_aa

from betafold.training.models.fragment_vae import create_cvae_mlp, create_cvae_conv
from betafold.training.models.fragment_draw import create_draw_mlp

class SimpleCVAEData(PDBFragment):
  def __getitem__(self, index):
    grab_result = super(SimpleCVAEData, self).__getitem__(index)
    phi, psi, sequence, _ = grab_result
    data = torch.cat((
      torch.Tensor(phi).unsqueeze(0),
      torch.Tensor(psi).unsqueeze(0)
    ), dim=0)
    condition = one_hot_aa(sequence)
    data = data / 180 * np.pi
    return (
      (torch.cat((torch.sin(data), torch.cos(data)), dim=0) + 1) / 2,
      condition
    )

class PreLoadedCVAEData(PDBNumeric):
  def __init__(self, path, size=64, **kwargs):
    super(PreLoadedCVAEData, self).__init__(
      path, **kwargs
    )
    self.size = size

    if not os.path.isfile(os.path.join(path, "pdb_index.npy")):
      preloaded_data = []
      preloaded_condition = []
      dump_index = []
      count = 0
      for idx in range(len(self)):
        print(idx / len(self))
        _, _, _, phi, psi, sequence, _ = \
          super(PreLoadedCVAEData, self).__getitem__(idx)
        data = torch.cat((
          torch.Tensor(phi).unsqueeze(0),
          torch.Tensor(psi).unsqueeze(0)
        ), dim=0)
        condition = one_hot_aa(sequence)
        data = data / 180 * np.pi
        data = (torch.cat((torch.sin(data), torch.cos(data)), dim=0) + 1) / 2
        preloaded_data.append(data)
        preloaded_condition.append(condition)

        dump_index.append(count)
        count += data.size(1)
      self.pre_index = np.array(dump_index + [count])
      self.pre_data = torch.cat(preloaded_data, dim=1).numpy()
      self.pre_condition = torch.cat(preloaded_condition, dim=1).numpy()

      np.save(os.path.join(path, "pdb_index.npy"), self.pre_index)
      np.save(os.path.join(path, "pdb_data.npy"), self.pre_data)
      np.save(os.path.join(path, "pdb_condition.npy"), self.pre_condition)
    else:
      self.pre_index = np.load(os.path.join(path, "pdb_index.npy"))
      self.pre_data = np.load(os.path.join(path, "pdb_data.npy"))
      self.pre_condition = np.load(os.path.join(path, "pdb_condition.npy"))

  def __getitem__(self, index):
    start, stop = self.pre_index[index:index + 2]
    print(self.pre_index[:10])
    data = self.pre_data[:, start:stop]
    condition = self.pre_condition[:, start:stop]
    while data.shape[1] < self.size:
      index = (index + 1) % self.size
      start, stop = self.pre_index[index:index + 2]
      data = self.pre_data[:, start:stop]
      condition = self.pre_condition[:, start:stop]
    data = torch.Tensor(data)
    condition = torch.Tensor(condition)
    offset = random.randint(0, data.shape[1] - self.size)
    return (
      data[:, offset:offset + self.size],
      condition[:, offset:offset + self.size]  
    )

class CorrectiveConditionalVAETraining(IndependentConditionalVAETraining):
  def __init__(self, encoder, decoder, data, **kwargs):
    super(CorrectiveConditionalVAETraining, self).__init__(
      encoder, decoder, data, **kwargs
    )

  def loss(self, mean, logvar, reconstruction, guess, target):
    loss_val = super().loss(mean, logvar, reconstruction, target)
    guess_loss = torch.nn.functional.binary_cross_entropy_with_logits(
      guess, target, reduction="sum"
    ) / target.size(0)

    self.current_losses["guess"] = float(guess_loss)

    return loss_val + guess_loss

  def run_networks(self, data):
    target, condition = data
    _, mean, logvar = self.encoder(target, condition)
    sample = self.sample(mean, logvar)
    reconstruction, guess = self.decoder(sample, condition)

    return mean, logvar, reconstruction, guess, target


if __name__ == "__main__":
  from pyrosetta import init
  init("-mute all")

  path = sys.argv[1]
  training_args = {
    "device": "cuda",
    "batch_size": 128,
    "verbose": True,
    "max_epochs": 500,
    "network_name": "draw-small"
  }
  
  encoder, decoder, prior = create_cvae_mlp(
    feature_size=26, out_size=256, latent_size=128,
    sequence_size=32, depth=2, hidden_size=128
  )

  data = PreLoadedCVAEData(
    path, size=32
  )

  training = IndependentConditionalVAETraining(
    encoder, decoder, data, **training_args
  )

  encoder, decoder = training.train()
