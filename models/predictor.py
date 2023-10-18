"""
@Date  : 2022/12/18
@Time  : 15:24
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
from torch import nn as nn
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from .unet import UNet1DModel


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class DiffusionPredictor(nn.Module):
    def __init__(self, dim_model, dim_time, bits):
        super(DiffusionPredictor, self).__init__()
        sinu_pos_emb = LearnedSinusoidalPosEmb(dim_time)
        dim_sinu = dim_time + 1
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim_sinu, 2 * dim_time),
            nn.GELU(),
            nn.Linear(2 * dim_time, 2 * bits),
        )

        self.linear1 = nn.Linear(bits + dim_model, 2 * bits)
        self.linear2 = nn.Linear(2 * bits, bits)
        self.linear3 = nn.Linear(bits, bits)


        self._reset_params()

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_corrupt, time, x_bert):
        """
        predict epsilon or x_0
        Args:
            x_corrupt: [bsz, seq_len, bits]
            time: [bsz]
            x_bert: [bsz, seq_len, dim_model]

        Returns:
            epsilon or x_0: [bsz, seq_len, bits]
        """
        x = F.gelu(self.linear1(torch.cat([x_corrupt, x_bert], dim=-1)))
        x = F.gelu(self.linear2(x))
        x = F.gelu(self.linear3(x))

        time_embeds = self.time_mlp(time)

        # [bsz, seq_len, 2 * dim_model]
        scale_shift = time_embeds.unsqueeze(dim=1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = (scale + 1) * x + shift

        return x

# class DiffusionPredictor(nn.Module):
#     def __init__(self, dim_model, bits):
#         super(DiffusionPredictor, self).__init__()
#         self.down = nn.Sequential(
#             nn.Linear(dim_model, dim_model),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim_model, bits),
#             nn.ReLU(inplace=True),
#             nn.Linear(bits, bits),
#         )
#         self.unet = UNet1DModel(in_channels=bits, out_channels=bits)
#
#
#
#         self._reset_params()
#
#     def _reset_params(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, x_corrupt, time, x_bert):
#         """
#         predict epsilon or x_0
#         Args:
#             x_corrupt: [bsz, seq_len, bits]
#             time: [bsz]
#             x_bert: [bsz, seq_len, dim_model]
#
#         Returns:
#             epsilon or x_0: [bsz, seq_len, bits]
#         """
#         x = self.down(x_bert) + x_corrupt.to(torch.float32)
#         # [bsz, num_channels(4), len]
#         x = self.unet(x.permute(0, 2, 1), time).sample.permute(0, 2, 1).contiguous()
#         return x
