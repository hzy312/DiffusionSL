import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        """

        Args:
            x: [bsz, len, dim]
            mask: [bsz, len]

        Returns:

        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = mask[:, None, None, :]
        mask = (mask != 1) * -10000
        attn += mask.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, mask):
        """

        Args:
            x: [bsz, len, dim]
            mask: [bsz, len]

        Returns:

        """
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, N, dim_head]
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = mask[:, None, None, :]
        mask = (mask != 1) * -10000
        attn += mask.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)


    def forward(self, x, c, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.cross_attn(self.norm2(x), c, mask)
        x = x + self.mlp(self.norm3(x))
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)


    def forward(self, x, c):
        x = self.linear(self.norm_final(x))
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            in_channels=4,
            hidden_size=1152,
            num_steps=100,
            time_dim=128,
            depth=12,
            num_heads=16,
            mlp_ratio=4.0,
    ):
        super().__init__()
        # self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # bits ---> hidden_size
        self.x_embedder = nn.Linear(2 * in_channels, hidden_size)
        self.time_embed = nn.Embedding(num_steps, time_dim)
        self.time_mlp = nn.Linear(time_dim, hidden_size)
        self.bert_mlp = nn.Linear(hidden_size, hidden_size)
        # self.fuse_mlp = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, in_channels)
        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


        # # Initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.time_embed.weight, std=0.02)
        nn.init.normal_(self.time_mlp.weight, std=0.02)
        nn.init.normal_(self.bert_mlp.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, bert_features, attention_mask, x_self_cond=None):
        """
        Forward pass of DiT.
        x: [bsz, len, bits]
        # x: [bsz, len, hidden_size]
        t: (N,) tensor of diffusion timesteps
        bert_featuers: [bsz, len, hidden_size]
        Returns:
            pred_noise or pred_x0: [bsz, len, bits]
        """
        bsz, seq_len = x.shape[0], x.shape[1]
        # [bsz, seq_len, hid]
        pos_embeds = sinusoidal_position_embedding(bsz, seq_len, self.hidden_size).to(x.device)
        if x_self_cond is None:
            x_cond = torch.zeros_like(x)
        else:
            x_cond = x_self_cond
        x = self.x_embedder(torch.cat([x, x_cond], dim=-1).to(torch.float)) + pos_embeds
        t = self.time_mlp(self.time_embed(t)).unsqueeze(dim=1).expand(-1, bert_features.shape[1], -1)
        # b = self.bert_mlp(bert_features)
        # c = t + bert_features  # [bsz, len, hidden_size]
        c = bert_features + t

        for block in self.blocks:
            x = block(x, c, attention_mask)  # (N, T, D)
        x = self.final_layer(x, c)
        return x


def sinusoidal_position_embedding(batch_size, seq_len, output_dim):
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    # [seq_len, out_dim // 2]
    embeddings = position_ids * indices
    # [seq_len, out_dim // 2, 2]
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
    # [sin, cos, sin, cos, ...]
    embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    return embeddings
