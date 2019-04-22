import numpy as np
import torch

from pyrosetta import *
from rosetta.core.scoring.methods import \
  ContextIndependentOneBodyEnergy, LongRangeTwoBodyEnergy
from rosetta.core.scoring import ScoreType

from betafold.distributions.von_mises import TorsionDistribution

@rosetta.EnergyMethod
class DistanceTerm(LongRangeTwoBodyEnergy):
  def __init__(self, distogram, reference_distogram,
               min_distance=2, max_distance=22, bins=64):
    LongRangeTwoBodyEnergy.__init__(self, self.creator())
    self.distogram = distogram
    self.reference = reference_distogram
    self.min_distance = min_distance
    self.max_distance = max_distance
    self.bins = bins

  def _bin_distance(self, distance):
    numerator = distance - self.min_distance + 1e-6
    denominator = self.max_distance - self.min_distance + 1e-6
    return int(numerator / denominator * self.bins)

  def residue_pair_energy(self, res1, res2, pose, sf, emap):
    cb1 = res1.atom(res1.atom_index("CB")).xyz
    cb2 = res2.atom(res2.atom_index("CB")).xyz
    pos1 = res1.seqpos() - 1
    pos2 = res2.seqpos() - 1
    distance = (cb2 - cb1).norm()
    logit_bin = self._bin_distance(distance)
    numerator = self.reference[pos1, pos2, logit_bin]
    denominator = self.distogram[pos1, pos2, logit_bin]
    score = numerator - denominator
    emap.get().set(self.scoreType, score)

@rosetta.EnergyMethod
class TorsionTerm(ContextIndependentOneBodyEnergy):
  def __init__(self, torsion, reference_torsion, bins=36):
    ContextIndependentOneBodyEnergy.__init__(self, self.creator())
    self.torsion = torsion
    self.reference = None
    self.has_ref = True
    if reference_torsion is None:
      self.has_ref = False
    else:
      self.reference = reference_torsion
    self.bins = bins

  def _bin_angle(self, angle):
    angle_bin = int((angle + 180) / 360 * self.bins)
    return angle_bin

  def residue_energy(self, res, pose, emap):
    pos = res.seqpos()
    phi = pose.phi(pos)
    psi = pose.psi(pos)
    phi_bin = self._bin_angle(phi)
    psi_bin = self._bin_angle(psi)
    if self.has_ref:
      numerator = self.reference[pos, phi_bin, psi_bin]
    else:
      numerator = 0.0
    denominator = self.torsion[pos, phi_bin, psi_bin]
    score = numerator - denominator
    emap.get().set(self.scoreType, score)

@rosetta.EnergyMethod
class SmoothTorsionTerm(TorsionTerm):
  def __init__(self, torsion, reference_torsion, bins=36,
               concentration=1 / ((np.pi / 18) ** 2)):
    TorsionTerm.__init__(self, torsion, reference_torsion, bins=bins)
    space = np.linspace(-180, 180, bins)
    self.mu = np.repeat(space, bins)
    self.nu = np.repeat(space.reshape(bins, 1), bins, axis=1).reshape(-1)
    self.weights = torch.softmax(
      self.torsion.reshape(self.torsion.size(0), -1),
      dim=1
    )
    self.k = concentration

  def residue_energy(self, res, pose, emap):
    pos = res.seqpos()
    phi = pose.phi(pos)
    psi = pose.psi(pos)
    distribution = TorsionDistribution(
      self.mu, self.nu, self.weights[pos],
      k1=self.k, k2=self.k
    )
    return distribution.log_density(phi, psi)

@rosetta.EnergyMethod
class SmoothArgmaxTorsionTerm(TorsionTerm):
  def __init__(self, torsion, reference_torsion, bins=36,
               concentration=1 / ((np.pi / 18) ** 2)):
    TorsionTerm.__init__(self, torsion, reference_torsion, bins=bins)
    self.k = concentration
    torsion_max = self.torsion.reshape(
      self.torsion.size(0), -1
    ).argmax(dim=1)
    self.mu = torsion_max % bins
    self.nu = torsion_max // bins

  def residue_energy(self, res, pose, emap):
    pos = res.seqpos()
    phi = pose.phi(pos)
    psi = pose.psi(pos)
    distribution = TorsionDistribution(
      self.mu[pos], self.nu[pos],
      k1=self.k, k2=self.k
    )
    return distribution.log_density(phi, psi)

@rosetta.EnergyMethod
class InterpolatedTorsionTerm(TorsionTerm):
  def __init__(self, torsion, reference_torsion, bins=36, ):
    TorsionTerm.__init__(self, torsion, reference_torsion, bins=bins)

  def _debin_angle(self, bin_id):
    angle = bin_id / self.bins * 360 - 180
    return angle

  def _interpolate(self, pos, phi, psi):
    phi_bin = self._bin_angle(phi)
    phi_next = (phi_bin + 1) % self.bins
    psi_bin = self._bin_angle(psi)
    psi_next = (psi_bin + 1) % self.bins
    phi_offset = self._debin_angle(phi_bin)
    psi_offset = self._debin_angle(psi_bin)
    phi_p = (phi - phi_offset) / 360 * self.bins % 1
    psi_p = (psi - psi_offset) / 360 * self.bins % 1
    if self.has_ref:
      numerators = np.array([
        [self.reference[pos, phi_bin, psi_bin], self.reference[pos, phi_next, psi_bin]],
        [self.reference[pos, phi_bin, psi_next], self.reference[pos, phi_next, psi_next]]
      ])
    else:
      numerators = np.zeros((2, 2))
    denominators = np.array([
      [self.torsion[pos, phi_bin, psi_bin], self.torsion[pos, phi_next, psi_bin]],
      [self.torsion[pos, phi_bin, psi_next], self.torsion[pos, phi_next, psi_next]]
    ])
    score_points = numerators - denominators
    score = score_points[0, 0] * (1 - phi_p) * (1 - psi_p) \
          + score_points[1, 0] * phi_p * (1 - psi_p) \
          + score_points[0, 1] * (1 - phi_p) * psi_p \
          + score_points[1, 1] * phi_p * psi_p
    return score

  def residue_energy(self, res, pose, emap):
    pos = res.seqpos()
    phi = pose.phi(pos)
    psi = pose.psi(pos)

    score = self._interpolate(pos, phi, psi)

    emap.get().set(self.scoreType, score)

def get_betafold_scoring_function(distograms, torsions,
                                  distance_args=None,
                                  torsion_args=None,
                                  base_score="score2_smooth",
                                  distance_weight=-1.0,
                                  torsion_weight=-1.0):
  if distance_args is None:
    distance_args = {}
  if torsion_args is None:
    torsion_args = {}

  distance = DistanceTerm(*distograms, **distance_args)
  torsion = TorsionTerm(*torsions, **torsion_args)

  score = create_score_function(base_score)
  score.add_extra_method(torsion.scoreType, torsion_weight, torsion)
  score.add_extra_method(torsion.scoreType, distance_weight, distance)

  return score
