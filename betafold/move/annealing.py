import random

import torch

from pyrosetta import *
from rosetta.protocol.simple_moves import SwitchResidueTypeSetMover

from protsupport.data.embedding import one_hot_aa

# TODO: WIP implementation of simulated annealing workers.

class AnnealingPipeline():
  def __init__(self, score, mover, pymol=None,
               coarse_steps=10, fine_steps=10):
    self.score = score
    self.mover = mover
    self.relax = rosetta.protocols.relax.FastRelax(score)
    self.pymol = pymol
    self.to_centroid = SwitchResidueTypeSetMover("centroid")
    self.to_full_atom = SwitchResidueTypeSetMover("fullatom")

    self.coarse_steps = coarse_steps
    self.fine_steps = fine_steps

    self.best_coarse

  def coarse_move(self, pose):
    target = Pose()
    target.assign(pose)
    self.mover(target)
    if self.pymol:
      self.pymol.apply(target)
    return target

  def fine_move(self, pose):
    full_atom = Pose()
    full_atom.assign(pose)
    self.to_full_atom.apply(full_atom)
    self.relax.apply(full_atom)
    if self.pymol:
      self.pymol.apply(full_atom)
    return full_atom

  def refragment(self, pose):
    phi = torch.Tensor([
      self.pose.phi(idx)
      for idx in range(1, pose.total_residue() + 1)
    ])
    psi = torch.Tensor([
      self.pose.psi(idx)
      for idx in range(1, pose.total_residue() + 1)
    ])
    combined = torch.cat((phi.unsqueeze(1), psi.unsqueeze(1)), dim=0)
    fragments = []
    for idx in range(pose.total_residue() - self.mover.fragment_size):
      chop = combined[:, idx:idx + self.mover.fragment_size]
      fragments.append(chop[None])
    fragments = torch.cat(fragments, dim=0)
    self.mover.add_fragments(fragments)

  def apply(self, pose):
    best_coarse_results = []
    for idx in range(self.coarse_steps):
      coarse_result = self.coarse_move(pose)
      best_coarse_results.append(coarse_result)

    best_fine_results = []
    for idx in range(self.fine_steps):
      fine_result = self.fine_move(pose)
      best_fine_results.append(fine_result)

class RefragmentWorker():
  def __init__(self, address):
    self.address = address
    self.database = ... # TODO

  def refragment(self, pose):
    phi = torch.Tensor([
      self.pose.phi(idx)
      for idx in range(1, pose.total_residue() + 1)
    ])
    psi = torch.Tensor([
      self.pose.psi(idx)
      for idx in range(1, pose.total_residue() + 1)
    ])
    combined = torch.cat((phi.unsqueeze(1), psi.unsqueeze(1)), dim=0)
    fragments = []
    for idx in range(pose.total_residue() - self.mover.fragment_size):
      chop = combined[:, idx:idx + self.mover.fragment_size]
      fragments.append(chop[None])
    fragments = torch.cat(fragments, dim=0)
    self.database.save_fragments(fragments)

  def run(self):
    while True:
      fine_pose = self.database.load_fine_pose()
      self.refragment(fine_pose)

class CoarseAnnealingWorker():
  def __init__(self, score, mover, address, pymol=None):
    self.score = score
    self.mover = mover
    self.pymol = pymol
    self.address = address
    self.database = ... # TODO

  def coarse_move(self, pose):
    target = Pose()
    target.assign(pose)
    self.mover(target)
    if self.pymol:
      self.pymol.apply(target)
    return target

  def load_fragments(self):
    fragments = self.database.load_fragments()
    return fragments

  def load_pose(self):
    pose = self.database.load_pose()
    return pose

  def apply(self, pose):
    fragments = self.load_fragments()
    self.mover.fragments = fragments
    pose = self.database.load_pose()
    pose = self.coarse_move(pose)
    score = self.score.score(pose)
    self.database.save_pose(pose, score)
  
  def run(self):
    while True:
      pose = self.load_pose()
      self.apply(pose)

class RefinementWorker():
  def __init__(self, score, mover, address, pymol=None):
    self.score = score
    self.mover = mover
    self.pymol = pymol
    self.address = address
    self.database = ... # TODO
    self.relax = rosetta.protocols.relax.FastRelax(score)
    self.to_fullatom = SwitchResidueTypeSetMover("fullatom")

  def load_fragments(self):
    fragments = self.database.load_fragments()
    return fragments

  def load_pose(self):
    pose = self.database.load_pose()
    return pose

  def fine_move(self, pose):
    full_atom = Pose()
    full_atom.assign(pose)
    self.to_full_atom.apply(full_atom)
    self.relax.apply(full_atom)
    if self.pymol:
      self.pymol.apply(full_atom)
    return full_atom

  def apply(self, pose):
    pose = self.fine_move(pose)
    score = self.score.score(pose)
    self.database.save_fine_pose(pose, score)

  def run(self):
    while True:
      pose = self.load_pose()
      self.apply(pose)
