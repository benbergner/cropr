import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_
from timm.layers.weight_init import trunc_normal_tf_
from timm.layers.mlp import Mlp


class CrossAttention(nn.Module):
    """Cross-attention layer with learnable queries."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
        num_queries: int = 1,
        training: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))

        self.q = (
            nn.Linear(embed_dim, embed_dim, bias=qkv_bias) if q_proj else nn.Identity()
        )
        self.k = (
            nn.Linear(embed_dim, embed_dim, bias=qkv_bias) if k_proj else nn.Identity()
        )
        self.attn_norm = norm_layer(embed_dim) if pre_attn_norm else nn.Identity()

        if training:
            self.v = (
                nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                if v_proj
                else nn.Identity()
            )
            self.proj = (
                nn.Linear(embed_dim, embed_dim) if num_heads > 1 else nn.Identity()
            )

            if mlp:
                self.mlp_norm = norm_layer(embed_dim)
                self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))

        self.init_weights()

    def init_weights(self):
        D = self.queries.shape[-1]
        trunc_normal_tf_(self.queries, std=D**-0.5)

    def forward_scorer(self, x):
        B, N = x.shape[:2]

        x = self.attn_norm(x)

        q_in = self.queries.expand(B, -1, -1)
        q = self.q(q_in).reshape(B, self.num_queries, -1, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, -1, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1)

        scores = attn.sum((1, 2))
        return scores

    def forward(self, x):
        B, N, D = x.shape

        x = self.attn_norm(x)

        q_in = self.queries.expand(B, -1, -1)
        M = q_in.shape[1]
        q = self.q(q_in).reshape(B, M, -1, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, -1, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, -1, self.head_dim).transpose(1, 2)

        attn = q @ k.transpose(-2, -1)  # B, H, M, N
        attn_sm = (attn * self.scale).softmax(dim=-1)
        x = attn_sm @ v

        x = x.transpose(1, 2).reshape(B, M, D).squeeze(1)
        x = self.proj(x)

        if hasattr(self, "mlp"):
            x = x + self.mlp(self.mlp_norm(x))

        scores = attn.sum((1, 2))  # B, N

        return x, scores


class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.head.weight, std=0.02)
        self.head.bias.data.zero_()

    def forward(self, x):
        x = self.norm(x)
        x = self.head(x)
        return x


class Cropr(nn.Module):
    def __init__(
        self,
        pruning_rate: int = 40,
        num_queries: int = 1024,
        num_classes: int = 80,
        embed_dim: int = 1024,
        num_heads: int = 16,
        pre_attn_norm: bool = False,
        q_proj: bool = False,
        k_proj: bool = False,
        v_proj: bool = False,
        mlp: bool = True,
        mlp_ratio: float = 4.0,
        training: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.pruning_rate = pruning_rate
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mlp = mlp

        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pre_attn_norm=pre_attn_norm,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            mlp=mlp,
            num_queries=num_queries,
            training=training,
        )

        if training:
            self.head = SegmentationHead(embed_dim, num_classes)

    def prune(self, x, scores, pos, idx):

        B, M, D = x.shape
        num_keep = M - self.pruning_rate

        # Sort scores in descending order, get idx of sorted scores
        idx_sorted = torch.argsort(scores, dim=1, descending=True, stable=False)

        # Reorder
        x_reorder = torch.gather(x, 1, idx_sorted.unsqueeze(-1).expand(-1, -1, D))
        pos_reorder = torch.gather(
            pos, 1, (idx_sorted[:, 1:] - 1).unsqueeze(-1).expand(-1, -1, pos.shape[-1])
        )  # skip first id, which is CLS token
        idx_reorder = torch.gather(idx, 1, idx_sorted).view(B, M)

        # Select keep and pruned tokens
        x = x_reorder[:, :num_keep]
        x_r = x_reorder[:, num_keep:]
        pos = pos_reorder[:, : num_keep - 1]  # account for CLS token
        pos_r = pos_reorder[:, num_keep - 1 :]
        idx = idx_reorder[:, :num_keep]
        idx_r = idx_reorder[:, num_keep:]

        return x, x_r, pos, pos_r, idx, idx_r

    def forward(self, x, pos, idx, inference=False):

        if inference:
            scores = self.cross_attn.forward_scorer(x)
        else:
            # use detach to stop gradient flow
            x_aggr, scores = self.cross_attn(x.detach())

        scores[:, 0] = math.inf  # retain CLS token

        x, x_r, pos, pos_r, idx, idx_r = self.prune(x, scores, pos, idx)

        if inference:
            return x, x_r, pos, pos_r, idx, idx_r, None

        pred = self.head(x_aggr)

        return x, x_r, pos, pos_r, idx, idx_r, pred
