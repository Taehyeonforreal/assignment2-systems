from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.transformer import run_transformer_lm


class TransformerLM(nn.Module):
    """
    run_transformer_lm (함수형 A1 구현)을 nn.Module로 감싼 wrapper.

    nn.ParameterList로 파라미터를 등록하면 .to(device), .parameters(),
    optimizer = AdamW(model.parameters()) 등이 전부 동작한다.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # 파라미터 이름과 초기 텐서를 쌍으로 모음
        named: list[tuple[str, Tensor]] = []

        def reg(name: str, tensor: Tensor):
            named.append((name, tensor))

        std = 0.02
        reg("token_embeddings.weight", torch.randn(vocab_size, d_model) * std)
        reg("lm_head.weight",          torch.randn(vocab_size, d_model) * std)
        reg("ln_final.weight",         torch.ones(d_model))

        for i in range(num_layers):
            reg(f"layers.{i}.ln1.weight",              torch.ones(d_model))
            reg(f"layers.{i}.ln2.weight",              torch.ones(d_model))
            reg(f"layers.{i}.attn.q_proj.weight",      torch.randn(d_model, d_model) * std)
            reg(f"layers.{i}.attn.k_proj.weight",      torch.randn(d_model, d_model) * std)
            reg(f"layers.{i}.attn.v_proj.weight",      torch.randn(d_model, d_model) * std)
            reg(f"layers.{i}.attn.output_proj.weight", torch.randn(d_model, d_model) * std)
            reg(f"layers.{i}.ffn.w1.weight",           torch.randn(d_ff, d_model) * std)
            reg(f"layers.{i}.ffn.w2.weight",           torch.randn(d_model, d_ff) * std)
            reg(f"layers.{i}.ffn.w3.weight",           torch.randn(d_ff, d_model) * std)

        self._weight_names = [name for name, _ in named]
        self.weights = nn.ParameterList([nn.Parameter(t) for _, t in named])

    def _weights_dict(self) -> dict[str, Tensor]:
        return {name: param for name, param in zip(self._weight_names, self.weights)}

    def forward(self, in_indices: Tensor) -> Tensor:
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self._weights_dict(),
            in_indices=in_indices,
        )
