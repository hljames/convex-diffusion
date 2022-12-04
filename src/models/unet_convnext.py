import torch
import torch.nn as nn
from einops import rearrange

from src.models._base_model import BaseModel
from src.utilities.utils import exists
from src.models.modules.attention import LinearAttention
from src.models.modules.net_norm import LayerNorm, PreNorm
from src.models.modules.misc import Residual, SinusoidalPosEmb


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.time_emb_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)  # same shape as x

        if exists(self.time_emb_mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.time_emb_mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.residual_conv(x)


# model
class UnetConvNext(BaseModel):
    def __init__(
            self,
            dim: int = 64,
            dim_mults=(1, 2, 4, 8),
            input_channels: int = None,
            output_channels: int = None,
            with_time_emb: bool = True,
            double_conv_layer: bool = True,
            output_mean_scale: bool = False,
            residual: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        input_channels = input_channels or self.num_input_channels
        output_channels = output_channels or self.num_output_channels or input_channels

        self.save_hyperparameters()
        self.log_text.info(f"Is time embedding used? {with_time_emb}")
        self.example_input_array = torch.randn(1, input_channels, self.spatial_dims[0], self.spatial_dims[1])

        dims = [input_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_emb_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_emb_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim) if double_conv_layer else nn.Identity(),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, rescale='qk'))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, rescale='qk')))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim) if double_conv_layer else nn.Identity(),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, rescale='qk'))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, output_channels, 1)
        )

    def forward(self, x, time=None, condition=None):
        b, c, h, w = x.shape
        orig_x = x  # for skip connections
        t = self.time_emb_mlp(time) if exists(self.time_emb_mlp) else None

        h = []
        original_mean = torch.mean(x, [1, 2, 3], keepdim=True)

        # input shape = (batch_size, d_in, h, w)
        for i, (convnext, convnext2, attn, downsample) in enumerate(self.downs):
            x = convnext(x, t)       # shape = (batch_size, d_i, h, w), if i=0, d_i = dim, otherwise d_i = d_i-1 * 2
            if self.hparams.double_conv_layer:
                x = convnext2(x, t)  # shape = (batch_size, d_i, h, w)
            x = attn(x)              # shape = (batch_size, d_i, h, w)
            h.append(x)
            x = downsample(x)        # shape = (batch_size, d_i, h/2, w/2)

        x = self.mid_block1(x, t)    # shape = (batch_size, d, h/2, w/2), d is largest channel dim
        x = self.mid_attn(x)         # shape = (batch_size, d, h/2, w/2)
        x = self.mid_block2(x, t)    # shape = (batch_size, d, h/2, w/2)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)  # concatenate along channel dimension
            x = convnext(x, t)
            if self.hparams.double_conv_layer:
                x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        out = self.final_conv(x)
        if self.hparams.residual:
            return out + orig_x

        if self.hparams.output_mean_scale:
            print(f"Original mean: {original_mean.shape}")
            print(out.shape)
            out_mean = torch.mean(out, [1, 2, 3], keepdim=True)
            out = out - original_mean + out_mean

        return out


if __name__ == '__main__':
    model = ConvNextBlock(3, dim_out=64, time_emb_dim=None)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
