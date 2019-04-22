import numpy as np

def norm_von_mises(k1, k2, k3):
  return 1

def density_von_mises(phi, psi, mu=0, nu=0, k1=10, k2=10, k3=0):
  return norm_von_mises(k1, k2, k3) * np.exp(
    k1 * np.cos(phi - mu)
    + k2 * np.cos(psi - nu)
    - k3 * np.cos((phi - mu) - (psi - nu))
  )

def log_von_mises(phi, psi, mu=0, nu=0, k1=10, k2=10, k3=0):
  return k1 * np.cos(phi - mu) \
       + k2 * np.cos(psi - nu) \
       - k3 * np.cos((phi - mu) - (psi - nu))

def density_mixture_von_mises(phi, psi, w=1, mu=0, nu=0, k1=10, k2=10):
  result = w * density_von_mises(phi, psi, mu, nu, k1, k2)
  return result.sum()

def log_mixture_von_mises(phi, psi, w=1, mu=0, nu=0, k1=10, k2=10):
  return np.log(density_mixture_von_mises(phi, psi, w, mu, nu, k1, k2))

class TorsionDistribution():
  def __init__(self, mu, nu, w=1, k1=10, k2=10, k3=0):
    self.mu = mu
    self.nu = nu
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.w = w

  def log_density(self, phi, psi):
    return log_mixture_von_mises(
      phi, psi,
      mu=self.mu, nu=self.nu,
      w=self.w, k1=self.k1, k2=self.k2
    )
