from torch import nn
from torch.nn import functional as func

from torchsupport.modules.basic import MLP

from betafold.generate.angles import \
  TorsionConditionalPrior, TorsionConditionalEncoder, TorsionConditionalDecoder

class ReshapeMLP(nn.Module):
  def __init__(self, in_size, out_size, **kwargs):
    super(ReshapeMLP, self).__init__()
    self.mlp = MLP(
      in_size, out_size, **kwargs
    )

  def forward(self, inputs):
    out = inputs.view(inputs.size(0), -1)
    out = self.mlp(out)
    return out

class ConvStack(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=4, depth=3):
    super(ConvStack, self).__init__()
    self.blocks = nn.ModuleList([
      nn.Conv1d(in_size, 2 ** hidden_size, 3, padding=1)
    ] + [
      nn.Conv1d(2 ** (hidden_size + idx), 2 ** (hidden_size + idx + 1), 3, padding=1)
      for idx in range(depth - 1)
    ] + [
      nn.Conv1d(2 ** (hidden_size + depth - 1), out_size, 3, padding=1)
    ])
  
  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = func.elu_(block(out))
      out = func.max_pool1d(out, 2)
    out = func.adaptive_max_pool1d(out, 1)
    return out.reshape(out.size(0), -1)

def create_cvae_conv(feature_size, out_size, latent_size,
                     sequence_size, depth, hidden_size):
  condition_conv = ConvStack(
    feature_size, out_size, depth=depth
  )
  prior_conv = ConvStack(
    feature_size, out_size, depth=depth
  )
  encoder_angle_conv = ConvStack(
    4, out_size, depth=depth
  )
  decoder_angle_conv = ReshapeMLP(
    latent_size, out_size,
    hidden_size=hidden_size, depth=depth
  )

  prior = TorsionConditionalPrior(
    prior_conv,
    sequence_size=sequence_size,
    hidden_size=out_size,
    latent_size=latent_size,
    depth=2
  )
  encoder = TorsionConditionalEncoder(
    encoder_angle_conv, condition_conv,
    sequence_size=sequence_size,
    hidden_size=out_size,
    latent_size=latent_size,
    depth=2
  )
  decoder = TorsionConditionalDecoder(
    decoder_angle_conv, condition_conv,
    sequence_size=sequence_size,
    latent_size=latent_size,
    depth=2, return_torsion=False
  )
  
  return encoder, decoder, prior

def create_cvae_mlp(feature_size, out_size, latent_size,
                    sequence_size, depth, hidden_size):
  condition_mlp = ReshapeMLP(
    sequence_size * feature_size, out_size,
    hidden_size=hidden_size, depth=depth
  )
  encoder_angle_mlp = ReshapeMLP(
    sequence_size * 4, out_size,
    hidden_size=hidden_size, depth=depth
  )
  decoder_angle_mlp = ReshapeMLP(
    latent_size, out_size,
    hidden_size=hidden_size, depth=depth
  )

  prior = TorsionConditionalPrior(
    condition_mlp,
    sequence_size=sequence_size,
    hidden_size=out_size,
    latent_size=latent_size,
    depth=2
  )
  encoder = TorsionConditionalEncoder(
    encoder_angle_mlp, condition_mlp,
    sequence_size=sequence_size,
    hidden_size=out_size,
    latent_size=latent_size,
    depth=2
  )
  decoder = TorsionConditionalDecoder(
    decoder_angle_mlp, condition_mlp,
    sequence_size=sequence_size,
    latent_size=latent_size,
    depth=2, return_torsion=False
  )
  
  return encoder, decoder, prior

