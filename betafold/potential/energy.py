from pyrosetta import *
from rosetta.core.scoring.methods import \
  ContextIndependentOneBodyEnergy, LongRangeTwoBodyEnergy
from rosetta.core.scoring import ScoreType

@rosetta.EnergyMethod
class DistanceTerm(LongRangeTwoBodyEnergy):
  def __init__(self, distogram, reference_distogram):
    LongRangeTwoBodyEnergy.__init__(self, self.creator())
    self.distogram = distogram
    self.reference = reference_distogram

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
  def __init__(self, torsion, reference_torsion):
    ContextIndependentOneBodyEnergy.__init__(self, self.creator())
    self.torsion = torsion
    self.reference = reference_torsion

  def residue_energy(self, res, pose, emap):
    pos = res.seqpos()
    phi = pose.phi(pos)
    psi = pose.psi(pos)
    phi_bin = self._bin_angle(phi)
    psi_bin = self._bin_angle(psi)
    numerator = self.reference[pos, phi_bin, psi_bin]
    denominator = self.torsion[pos, phi_bin, psi_bin]
    score = numerator - denominator
    emap.get().set(self.scoreType, score)

def get_betafold_scoring_function(distograms, torsions):
  distance = DistanceTerm(*distograms)
  torsion = TorsionTerm(*torsions)

  score = ScoreFunction()
  score.add_extra_method(torsion.scoreType, -1.0, torsion)
  score.add_extra_method(torsion.scoreType, -1.0, distance)
  score.set_weight(ScoreType.vdw, 1.0)

  return score
