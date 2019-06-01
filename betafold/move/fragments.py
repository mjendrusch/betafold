import random

import torch

from pyrosetta import *

from protsupport.data.embedding import one_hot_aa

class GenerativeFragmentMover(rosetta.protocols.moves.Mover):
  def __init__(self, fragment_iterations=100, 
               fragment_size=9, num_fragments=100,
               input_size=32):
    rosetta.protocols.moves.Mover.__init__(self)
    self.fragment_iterations = fragment_iterations
    self.fragment_size = fragment_size
    self.input_size = input_size
    self.num_fragments = num_fragments
    self.cache_sequence = None

  def get_name(self):
    return self.__class__.__name__

  def compute_features(self, sequence):
    raise NotImplementedError("Abstract.")

  def tile_features(self, features):
    offset = self.input_size - self.fragment_size
    length = features.size(-1)
    n_steps = length // offset + 1
    tiles = []
    for idx in range(n_steps):
      tile = features[:, idx * offset:idx * offset + self.input_size]
      tiles.append(tile[None])
    if (n_steps - 1) * offset < length:
      last_tile = features[:, -self.input_size:]
      tiles.append(last_tile[None])
    tiles = torch.cat(tiles, dim=0)
    return tiles

  def chop_samples(self, samples):
    offset = self.input_size - self.fragment_size
    chopped = []
    for idx in range(offset):
      chop = samples[:, :, idx:idx + self.fragment_size]
      chopped.append(chop)
    rebuilt = []
    for idx in range(samples.size(0)):
      for item in chopped:
        rebuilt.append(item[idx].unsqueeze(0))
    rebuilt = torch.cat(rebuilt, dim=0)
    return rebuilt

  def generate(self, features):
    raise NotImplementedError("Abstract.")

  def fragment_insert(self, pose, position):
    phi, psi = random.choice(self.fragments)[position]
    for idx in range(self.fragment_size):
      pose.set_phi(idx + position, phi[idx])
      pose.set_psi(idx + position, psi[idx])

  def apply(self, pose):
    pose = pose.get()
    sequence = pose.sequence()
    if sequence != self.cache_sequence:
      self.cache_sequence = sequence
      self.cache_features = self.compute_features(sequence)
      batch = self.tile_features(self.cache_features)
      fragments = []
      for idx in range(self.num_fragments):
        samples = self.generate(batch)
        fragments += self.chop_samples(samples)
      self.fragments = fragments
    position = random.choice(range(len(sequence)))
    self.fragment_insert(pose, position)

class VAEFragmentMover(GenerativeFragmentMover):
  def __init__(self, prior, generator,
               fragment_iterations=100,
               fragment_size=9,
               num_fragments=100,
               input_size=32):
    super().__init__(
      fragment_iterations=fragment_iterations,
      fragment_size=fragment_size,
      num_fragments=num_fragments,
      input_size=input_size
    )
    self.generator = generator
    self.prior = prior

  def compute_features(self, sequence):
    return one_hot_aa(sequence)

  def generate(self, features):
    _, mu, logvar = self.prior(features)
    sample = torch.randn_like(mu)
    decoded = self.generator(sample, features)
    fde = decoded.sigmoid() * 2 - 1
    phi = torch.atan2(fde[0, 0], fde[0, 2]) 
    psi = torch.atan2(fde[0, 1], fde[0, 3])
    return torch.cat((phi, psi), dim=1)
