from torch import nn
from torch.nn import functional as func

from torchsupport.modules.basic import MLP
from torchsupport.training.vae import ConditionalRecurrentCanvasVAETraining

from betafold.generate.angles import \
  TorsionConditionalPrior, TorsionConditionalEncoder, TorsionDrawDecoder

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

def create_draw_mlp(feature_size, out_size, latent_size,
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
  decoder = TorsionDrawDecoder(
    decoder_angle_mlp, condition_mlp,
    sequence_size=sequence_size,
    latent_size=latent_size,
    depth=2, return_torsion=False
  )
  
  return encoder, decoder, prior

