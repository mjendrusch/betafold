"""Attempts to reconstruct the DRAW-like model outlined
on slide 4 of Senior's CASP13 presentation."""

import numpy as np

import torch
from torch import nn
from torch.nn import functional as func

from torchsupport.modules.basic import MLP

class TorsionConditionalPrior(nn.Module):
  def __init__(self, input_net,
               sequence_size=32, hidden_size=64,
               latent_size=64, depth=3,
               activation=func.elu_):
    super(TorsionConditionalPrior, self).__init__()
    mlp_hidden = 2 * latent_size
    self.condition = input_net
    self.activation = activation
    self.postprocess = MLP(
      hidden_size, mlp_hidden,
      hidden_size=mlp_hidden, depth=depth - 1,
      activation=activation
    )
    self.mu = nn.Linear(mlp_hidden, latent_size)
    self.sigma = nn.Linear(mlp_hidden, latent_size)

  def forward(self, inputs):
    condition = self.condition(inputs)
    combine = self.activation(self.postprocess(
      condition.reshape(condition.size(0), -1)
    ))
    mu = self.mu(combine)
    sigma = self.sigma(combine)
    return None, mu, sigma

class TorsionConditionalEncoder(nn.Module):
  def __init__(self, angle_net, input_net,
               sequence_size=32, hidden_size=64,
               latent_size=64, depth=3,
               activation=func.elu_):
    super(TorsionConditionalEncoder, self).__init__()
    mlp_hidden = 2 * latent_size
    self.angle = angle_net
    self.condition = input_net
    self.activation = activation
    self.postprocess = MLP(
      hidden_size, mlp_hidden,
      hidden_size=mlp_hidden, depth=depth - 1,
      activation=activation
    )
    self.mu = nn.Linear(mlp_hidden, latent_size)
    self.sigma = nn.Linear(mlp_hidden, latent_size)

  def forward(self, angles, inputs):
    angle_features = angles * 2 - 1
    base = self.angle(angle_features)
    condition = self.condition(inputs)
    result = base + condition
    combine = self.activation(self.postprocess(
      result.reshape(result.size(0), -1)
    ))
    mu = self.mu(combine)
    sigma = self.sigma(combine)
    return None, mu, sigma

class TorsionConditionalDecoder(nn.Module):
  def __init__(self, angle_net, input_net,
               sequence_size=32,
               latent_size=64, depth=3,
               activation=func.elu_,
               return_torsion=False):
    super(TorsionConditionalDecoder, self).__init__()
    mlp_hidden = 2 * latent_size
    torsion_factor = 4 - 2 * return_torsion
    self.sequence_size = sequence_size
    self.angle = angle_net
    self.condition = input_net
    self.postprocess = MLP(
      mlp_hidden, sequence_size * torsion_factor,
      hidden_size=mlp_hidden, depth=depth - 1,
      activation=activation
    )

  def forward(self, latent, inputs):
    base = self.angle(latent)
    condition = self.condition(inputs)
    print(base.size(), condition.size())
    result = base + condition
    combine = self.postprocess(
      result.reshape(result.size(0), -1)
    )
    return combine.reshape(result.size(0), -1, self.sequence_size)

class TorsionDrawDecoder(TorsionConditionalDecoder):
  def __init__(self, angle_net, input_net, sequence_size=32,
               latent_size=64, depth=3, activation=func.elu_,
               return_torsion=False):
    super(TorsionDrawDecoder, self).__init__(
      angle_net, input_net, sequence_size=sequence_size,
      latent_size=latent_size, depth=depth, activation=activation,
      return_torsion=return_torsion
    )
    mlp_hidden = 2 * latent_size
    torsion_factor = 4 - 2 * return_torsion
    self.predict_angle = MLP(
      mlp_hidden, sequence_size * torsion_factor,
      hidden_size=mlp_hidden, depth=depth - 1,
      activation=activation
    )

  def transform_to_trig_logits(self, data):
    result = torch.cat((
      torch.sin(data), torch.cos(data)
    ), dim=1)
    result = (result + 1) / 2
    logit = torch.log(result / (1 - result + 1e-12) + 1e-12)
    return logit

  def forward(self, latent, inputs):
    base = self.angle(latent)
    condition = self.condition(inputs)
    correction = condition + base
    correction = self.postprocess(
      correction.reshape(correction.size(0), -1)
    )
    predicted = self.predict_angle(
      condition.reshape(condition.size(0), -1)
    )
    result = predicted + correction
    result = result.reshape(result.size(0), -1, self.sequence_size)

    guess = predicted.reshape(result.size(0), -1, self.sequence_size)

    return result, guess


# class TorsionRecurrent(nn.Module):
#   def __init__(self, preprocessor, angle,
#                encoder, decoder, depth=3):
#     super(TorsionRecurrent, self).__init__()
#     self.depth = depth
#     self.preprocessor = preprocessor
#     self.angle = angle
#     self.encoder = encoder
#     self.decoder = decoder

#   def forward(self, angles, inputs):
#     state = self._initial_state()
#     priors = []
#     mus = []
#     logvars = []
#     for _ in range(self.depth):
#       processed, state = self.preprocessor(inputs, state)
#       mu, logvar = self.encoder(angles, processed)
#       sigma = torch.exp(logvar)
#       sample = torch.randn_like(mu) * sigma + mu
#       predicted_angles = self.angle(processed)
#       decoded_angles, prior = self.decoder(sample, processed)
#       angles = predicted_angles + decoded_angles
#       priors.append(prior)
#       mus.append(mu),
#       logvars.append(logvar)
#     return angles, (priors, mus, logvars)
