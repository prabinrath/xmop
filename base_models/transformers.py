import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, dropout=0., learned=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned = learned

        if not learned:
            # Compute the positional encoding once
            pos_enc = torch.zeros(max_seq_len, dim)
            pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pos_enc[:, 0::2] = torch.sin(pos * div_term)
            pos_enc[:, 1::2] = torch.cos(pos * div_term)
            pos_enc = pos_enc.unsqueeze(0)

            # Register the positional encoding as a buffer to avoid it being
            # considered a parameter when saving the model
            self.register_buffer('pos_enc', pos_enc)
        else:
            self.pos_enc = nn.Parameter(torch.empty(max_seq_len, dim))

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:x.shape[1]]
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    """
    Derived from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Does not support is_causal argument as we want to calculate the mask manually.
    Assumes Q, K, V have same embedding dimension.
    """    
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            fused_attn: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        # serial projections are not the best way to do but keeps code simple
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == self.dim
        assert k.shape[-2] == v.shape[-2]
        B, Nq, _ = q.shape
        _, Nkv, _ = k.shape
        assert mask.shape[-2] == Nq and mask.shape[-1] == Nkv
        q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn += mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForwardMLP(nn.Module):
    """ 
    Derived from https://github.com/huggingface/pytorch-image-models
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(hidden_size, 
                                            num_heads=num_heads, 
                                            qkv_bias=True, 
                                            **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForwardMLP(in_features=hidden_size, 
                                  hidden_features=hidden_size * mlp_ratio, 
                                  drop=0)

    def forward(self, x, x_mask=None):
        x = self.norm1(x)
        x = x + self.self_attn(q=x, k=x, v=x, mask=x_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(hidden_size, 
                                            num_heads=num_heads, 
                                            qkv_bias=True, 
                                            **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(hidden_size, 
                                            num_heads=num_heads, 
                                            qkv_bias=True, 
                                            **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)       
        self.mlp = FeedForwardMLP(in_features=hidden_size, 
                                  hidden_features=hidden_size * mlp_ratio, 
                                  drop=0)

    def forward(self, x, mem, x_mask=None, mem_mask=None):
        x = self.norm1(x)
        x = x + self.self_attn(q=x, k=x, v=x, mask=x_mask)
        x = self.norm2(x)
        x = x + self.cross_attn(q=x, k=mem, v=mem, mask=mem_mask)
        x = x + self.mlp(self.norm3(x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionTransformerEncoderBlock(TransformerEncoderBlock):
    """
    Derived from https://github.com/facebookresearch/DiT
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_mask=None):
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(q=x, k=x, v=x, mask=x_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiffusionTransformerDecoderBlock(TransformerDecoderBlock):
    """
    Derived from https://github.com/facebookresearch/DiT
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, mem, c, x_mask=None, mem_mask=None):
        shift_msa, scale_msa, gate_msa, \
        shift_mca, scale_mca, gate_mca, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(q=x, k=x, v=x, mask=x_mask)
        x = modulate(self.norm2(x), shift_mca, scale_mca)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(q=x, k=mem, v=mem, mask=mem_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Derived from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
