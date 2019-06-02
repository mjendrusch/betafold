import numpy as np
import torch
from torch.autograd import grad

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
class InterpolatedDistanceTerm(DistanceTerm):
  def __init__(self, distogram, reference_distogram, min_distance=2, max_distance=22, bins=64):
    DistanceTerm.__init__(self, distogram, reference_distogram,
                          min_distance=min_distance,
                          max_distance=max_distance,
                          bins=bins)
  
  def _debin_distance(self, bin_id):
    distance = bin_id / self.bins * self.max_distance
    return distance

  def _interpolate(self, x, y, distance):
    if not isinstance(distance, torch.Tensor):
      distance = torch.tensor(distance)

    distance_bin = self._bin_distance(distance)
    distance_next = (distance_bin + 1) % self.bins
    distance_offset = self._debin_distance(distance_bin)
    distance_p = (distance - distance_offset) / self.max_distance * self.bins % 1
    if self.has_ref:
      numerators = torch.Tensor([
        self.reference[distance_bin, x, y],
        self.reference[distance_next, x, y]
      ])
    else:
      numerators = torch.zeros(2)
    denominators = torch.Tensor([
      self.distogram[distance_bin, x, y],
      self.distogram[distance_next, x, y]
    ])
    score_points = numerators - denominators
    score = score_points[0] * (1 - distance_p) + score_points[1] * distance_p
    return score

  def residue_pair_energy(self, res1, res2, pose, sf, emap):
    cb1 = res1.atom(res1.atom_index("CB")).xyz
    cb2 = res2.atom(res2.atom_index("CB")).xyz
    pos1 = res1.seqpos() - 1
    pos2 = res2.seqpos() - 1
    distance = (cb2 - cb1).norm()
    score = self._interpolate(pos1, pos2, distance)
    emap.get().set(self.scoreType, score)

  def _internal_distance_derivative(self, x, y):
    f2 = x - y
    distance = f2.length()
    if distance != 0.0:
      invd = 1 / distance
      f1 = x.cross(y)
      f1 *= invd
      f2 *= invd
    else:
      f1 = 0.0
    return f1, f2, distance

  def eval_residue_atom_derivative(self, atom_id, pose, dm, scorefxn, weights, f1, f2):
    pos1 = atom_id.rsd()
    x = pose.conformation().xyz(atom_id)
    for pos2 in range(pose.total_residue()):
      y = pose.conformation().xyz(rosetta.core.id.AtomID(atom_id.atomno(), pos2))
      r1, r2, distance = self._internal_distance_derivative(x, y)
      distance = torch.tensor(distance, requires_grad=True)
      score = self._interpolate(pos1, pos2, distance)
      gradient = grad([score], [distance])[0]
      r1 *= float(gradient)
      r2 *= float(gradient)
      f1 += r1
      f2 += r2

@rosetta.EnergyMethod
class TorsionTerm(ContextIndependentOneBodyEnergy):
  def __init__(self, torsion, reference_torsion=None, bins=36):
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
  def __init__(self, torsion, reference_torsion=None, bins=36,
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

  def _residue_logprob(self, pos, phi, psi):
    if not isinstance(phi, torch.Tensor):
      phi = torch.tensor(phi)
    if not isinstance(psi, torch.Tensor):
      psi = torch.tensor(psi)
    distribution = TorsionDistribution(
      self.mu, self.nu, self.weights[pos],
      k1=self.k, k2=self.k
    )
    return distribution.log_density(phi, psi)

  def residue_energy(self, res, pose, emap):
    pos = res.seqpos()
    phi = res.mainchain_torsion(1)
    psi = res.mainchain_torsion(2)
    return self._residue_logprob(pos, phi, psi)

  def defines_dof_derivatives(self, pose):
    return True

  def eval_residue_dof_derivative(self, res, min_data, dof_id, torsion_id, pose, sfxn, weights):
    if not torsion_id.valid() or torsion_id.type() != rosetta.core.id.TorsionType.BB:
      return 0.0

    phi = torch.tensor(res.mainchain_torsion(1), requires_grad=True)
    psi = torch.tensor(res.mainchain_torsion(2), requires_grad=True)
    prob = self._residue_logprob(phi, psi)
    gradients = grad([prob], [phi, psi])
    return float(gradients[torsion_id.torsion() - 1])

  def eval_dof_derivative(self, dof_id, torsion_id, pose, sfxn, weights):
    res = pose.residue(torsion_id.rsd())
    if not torsion_id.valid() or torsion_id.type() != rosetta.core.id.TorsionType.BB:
      return 0.0

    phi = torch.tensor(res.mainchain_torsion(1), requires_grad=True)
    psi = torch.tensor(res.mainchain_torsion(2), requires_grad=True)
    prob = self._residue_logprob(phi, psi)
    gradients = grad([prob], [phi, psi])
    return float(gradients[torsion_id.torsion() - 1])

@rosetta.EnergyMethod
class SmoothArgmaxTorsionTerm(SmoothTorsionTerm):
  def __init__(self, torsion, reference_torsion=None, bins=36,
               concentration=1 / ((np.pi / 18) ** 2)):
    SmoothTorsionTerm.__init__(self, torsion, reference_torsion, bins=bins)
    self.weights = None
    self.k = concentration
    torsion_max = self.torsion.reshape(
      self.torsion.size(0), -1
    ).argmax(dim=1)
    self.mu = torsion_max % bins
    self.nu = torsion_max // bins

  def _residue_logprob(self, pos, phi, psi):
    if not isinstance(phi, torch.Tensor):
      phi = torch.tensor(phi)
    if not isinstance(psi, torch.Tensor):
      psi = torch.tensor(psi)
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
    if not isinstance(phi, torch.Tensor):
      phi = torch.tensor(phi)
    if not isinstance(psi, torch.Tensor):
      psi = torch.tensor(psi)

    phi_bin = self._bin_angle(phi)
    phi_next = (phi_bin + 1) % self.bins
    psi_bin = self._bin_angle(psi)
    psi_next = (psi_bin + 1) % self.bins
    phi_offset = self._debin_angle(phi_bin)
    psi_offset = self._debin_angle(psi_bin)
    phi_p = (phi - phi_offset) / 360 * self.bins % 1
    psi_p = (psi - psi_offset) / 360 * self.bins % 1
    if self.has_ref:
      numerators = torch.Tensor([
        [self.reference[pos, phi_bin, psi_bin], self.reference[pos, phi_next, psi_bin]],
        [self.reference[pos, phi_bin, psi_next], self.reference[pos, phi_next, psi_next]]
      ])
    else:
      numerators = torch.zeros(2, 2)
    denominators = torch.Tensor([
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
    phi = res.mainchain_torsion(1)
    psi = res.mainchain_torsion(2)
    score = self._interpolate(pos, phi, psi)
    emap.get().set(self.scoreType, score)
  
  def defines_dof_derivatives(self, pose):
    return True

  def eval_residue_dof_derivative(self, res, min_data, dof_id, torsion_id, pose, sfxn, weights):
    if not torsion_id.valid() or torsion_id.type() != rosetta.core.id.TorsionType.BB:
      return 0.0

    phi = torch.tensor(res.mainchain_torsion(1), requires_grad=True)
    psi = torch.tensor(res.mainchain_torsion(2), requires_grad=True)
    prob = self._interpolate(phi, psi)
    gradients = grad([prob], [phi, psi])
    return float(gradients[torsion_id.torsion() - 1])

  def eval_dof_derivative(self, dof_id, torsion_id, pose, sfxn, weights):
    res = pose.residue(torsion_id.rsd())
    if not torsion_id.valid() or torsion_id.type() != rosetta.core.id.TorsionType.BB:
      return 0.0

    phi = torch.tensor(res.mainchain_torsion(1), requires_grad=True)
    psi = torch.tensor(res.mainchain_torsion(2), requires_grad=True)
    prob = self._interpolate(phi, psi)
    gradients = grad([prob], [phi, psi])
    return float(gradients[torsion_id.torsion() - 1])

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
