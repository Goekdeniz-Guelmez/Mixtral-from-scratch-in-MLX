# Copyright © 2023 Apple Inc. & Gökdeniz Gülmez

from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from typing import List, Optional, Tuple, Generator, Dict, Union

import numpy as np

import mlx.nn as nn
import mlx.core as mx



@dataclass
class ModelArgs(Serializable):
    creators: Dict
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    rope_traditional: bool
    rope_theta: float
    rope_scaling: Optional[Dict[str, Union[float, str]]]
    num_experts_per_tok: int
    num_local_experts: int
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads



class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output



class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta

        self.repeats = self.num_heads // self.num_key_value_heads

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = nn.RoPE(self.head_dim, traditional=args.rope_traditional, base=args.rope_theta) # For using the build in RoPE embedding from mlx

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        if cache is not None:
            key_cache, value_cache = cache

            queries = self.rope(queries, offset=key_cache.shape[2])

            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)

            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)

        if mask is not None:
            scores += mask

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values)



class MoeFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn_dim = args.intermediate_size
        self.hidden_dim = args.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))



class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size

        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = [MoeFeedForward(args=args) for _ in range(self.num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])  # TODO remove it once we figure out how to fine tune TopK in MOE

        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        if self.training:
            mx.eval(inds)
            inds = np.array(inds)
            y = mx.zeros((x.shape[0], ne, x.shape[-1]))
            for e, expert in enumerate(self.experts):
                idx1, idx2 = map(mx.array, np.where(inds == e))
                if idx1.size == 0:
                    continue
                y[idx1, idx2] = expert(x[idx1])
            y = (y * scores[:, :, None]).sum(axis=1)
        else:
            y = []
            for xt, st, it in zip(x, scores, inds.tolist()):
                yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1)

                y.append(yt[None, :])
            y = mx.concatenate(y)

        return y.reshape(orig_shape)



class MixtralDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.attention = Attention(args)

        self.block_sparse_moe = SparseMoeBlock(args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        r, cache = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.block_sparse_moe(self.attention_norm(h))

        out = h + r

        return out, cache



class MixtralModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [MixtralDecoderLayer(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        mask = None
        T = h.shape[1]

        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.creators = args.creators
        self.model = MixtralModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache

    @property
    def layers(self):
        return self.model.layers