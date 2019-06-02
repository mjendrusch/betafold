import random

import torch

from pyrosetta import *
from rosetta.protocols.minimization_packing import MinMover

class LangevinPipeline():
  def __init__(self, initializer, score, max_iter=1000, pymol=None):
    self.initializer = initializer
    self.score = score
    self.move_map = MoveMap()
    self.move_map.set_bb(True)
    self.min_mover = MinMover()
    self.min_mover.movemap(self.move_map)
    self.max_iter = max_iter
    self.pymol = pymol

  def step(self, pose):
    target = Pose()
    target.assign(pose)
    self.min_mover.apply(target)
    score = self.score(target)
    return score, target

  def corrupt(self, pose, sigma=0.1):
    target = Pose()
    target.assign(pose)
    phi_corruption = torch.randn(target.total_residue())
    psi_corruption = torch.randn(target.total_residue())
    for idx in range(target.total_residue()):
      target.set_phi(idx + 1, target.phi(idx + 1) + sigma * phi_corruption[idx])
      target.set_psi(idx + 1, target.psi(idx + 1) + sigma * psi_corruption[idx])
    return target

  def apply(self, pose):
    target = Pose()
    target.assign(pose)
    self.initializer.initialize(target)
    best = (self.score(target), target)
    for idx in range(self.max_iter):
      score, target = self.step(self.corrupt(best[1]))
      if score > best[0]:
        best = (score, target)
        if self.pymol:
          self.pymol.apply(target)
    return best
