"""
Reference from github.com/LTH14/JiT
---
MIT License

Copyright (c) 2025 Tianhong Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat


def broadcat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 16,
        ft_seq_len: int | None = None,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta: float = 10000,
        max_freq: float = 10,
        num_freqs: int = 1,
        num_cls_token: int = 0,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * np.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            _, D = cos_img.shape
            cos_pad = torch.ones(
                num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device
            )
            sin_pad = torch.zeros(
                num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device
            )

            freqs_cos = torch.cat([cos_pad, cos_img], dim=0)
            freqs_sin = torch.cat([sin_pad, sin_img], dim=0)
        else:
            freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp[0].weight.label = "embedder.0"
        self.mlp[2].weight.label = "embedder.2"

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout_p: float = 0.0
) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1))
    with torch.autocast("cuda", enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv.weight.label = "attn.qkv"
        self.proj.weight.label = "attn.proj"

    def forward(
        self, x: torch.Tensor, rope: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

        self.w12.weight.label = "mlp.w12"
        self.w3.weight.label = "mlp.w3"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear.weight.label = "final_layer.linear"
        self.adaLN_modulation[1].weight.label = "adaLN_modulation2"

    @torch.compile
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation[1].weight.label = "adaLN_modulation6"

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        feat_rope: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class AugmentedDiT(nn.Module):
    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 32,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        in_context_len: tuple[int, int] | None = (8, 4),
        in_context_start: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.s_embedder = TimestepEmbedder(hidden_size)
        self.omega_embedder = TimestepEmbedder(hidden_size)
        self.t_min_embedder = TimestepEmbedder(hidden_size)
        self.t_max_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # in-context cls token
        self.in_context_len = None
        if in_context_len:
            self.class_len, self.aux_len = in_context_len
            self.in_context_len = (
                self.class_len
                + self.aux_len * 2  # t, s
                + self.aux_len  # omega
                + self.aux_len * 2  # t_min, t_max
            )

            self.pos_cls = nn.Parameter(
                torch.zeros(1, self.class_len, hidden_size), requires_grad=True
            )
            self.pos_aux = nn.Parameter(
                torch.zeros(1, self.aux_len, hidden_size), requires_grad=True
            )

            torch.nn.init.normal_(self.pos_cls, std=0.02)
            torch.nn.init.normal_(self.pos_aux, std=0.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len or 0,
        )

        # transformer
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                )
                for i in range(depth)
            ]
        )

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.s_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.s_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.omega_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.omega_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_min_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_min_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_max_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_max_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, p: int) -> torch.Tensor:
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        omega: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        y: torch.Tensor,
        mid: list[int] | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        t = t.flatten()
        s = s.flatten()
        if t.shape == (1,):
            assert s.shape == (1,)
            assert omega.shape == (1,)
            assert t_min.shape == (1,)
            assert t_max.shape == (1,)
            t, s = t.repeat(B), s.repeat(B)
            omega, t_min, t_max = omega.repeat(B), t_min.repeat(B), t_max.repeat(B)
        if y.shape == (1,):
            y = y.repeat(B)
        # class and time embeddings
        t_emb = self.t_embedder(t)
        s_emb = self.s_embedder(s)
        omega_emb = self.omega_embedder(omega)
        t_min_emb = self.t_min_embedder(t_min)
        t_max_emb = self.t_max_embedder(t_max)
        y_emb = self.y_embedder(y)
        c = t_emb + s_emb + y_emb + omega_emb + t_min_emb + t_max_emb

        x = self.x_embedder(x)
        x += self.pos_embed

        middle_layers = []
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len and i == self.in_context_start:
                in_context_tokens = torch.cat(
                    [
                        y_emb[:, None].repeat(1, self.class_len, 1) + self.pos_cls,
                        t_emb[:, None].repeat(1, self.aux_len, 1) + self.pos_aux,
                        s_emb[:, None].repeat(1, self.aux_len, 1) + self.pos_aux,
                        omega_emb[:, None].repeat(1, self.aux_len, 1) + self.pos_aux,
                        t_min_emb[:, None].repeat(1, self.aux_len, 1) + self.pos_aux,
                        t_max_emb[:, None].repeat(1, self.aux_len, 1) + self.pos_aux,
                    ],
                    dim=1,
                )
                x = torch.cat([in_context_tokens, x], dim=1)

            rope = (
                self.feat_rope
                if i < self.in_context_start
                else self.feat_rope_incontext
            )
            x = block(x, c, rope)
            if mid and i in mid:
                middle_layers.append(x)

        x = x[:, self.in_context_len :]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if middle_layers:
            return output, middle_layers

        return output


def FSF_DMD_DiT_B_1(**kwargs):
    return AugmentedDiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        in_context_len=(8, 4),
        in_context_start=4,
        patch_size=1,
        **kwargs,
    )


FSFDMD_DIT_MODELS = {
    "FSF-DMD-DiT-B/1": FSF_DMD_DiT_B_1,
}
