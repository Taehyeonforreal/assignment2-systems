from __future__ import annotations
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import cs336_basics.transformer as _transformer
import cs336_basics.training as _training
import cs336_basics.bpe_tokenizer as _bpe

def run_linear(d_in, d_out, weights, in_features):
    return _transformer.run_linear(d_in, d_out, weights, in_features)

def run_embedding(vocab_size, d_model, weights, token_ids):
    return _transformer.run_embedding(vocab_size, d_model, weights, token_ids)

def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    return _transformer.run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features)

def run_scaled_dot_product_attention(Q, K, V, mask=None):
    return _transformer.run_scaled_dot_product_attention(Q, K, V, mask)

def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
    return _transformer.run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features)

def run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions=None):
    return _transformer.run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions)

def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    return _transformer.run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features)

def run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices):
    return _transformer.run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices)

def run_rmsnorm(d_model, eps, weights, in_features):
    return _transformer.run_rmsnorm(d_model, eps, weights, in_features)

def run_silu(in_features):
    return _transformer.run_silu(in_features)

def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    return _transformer.run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions)

def run_softmax(in_features, dim):
    return _transformer.run_softmax(in_features, dim)

def run_cross_entropy(inputs, targets):
    return _training.run_cross_entropy(inputs, targets)

def run_gradient_clipping(parameters, max_l2_norm):
    return _training.run_gradient_clipping(parameters, max_l2_norm)

def get_adamw_cls():
    return _training.get_adamw_cls()

def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    return _training.run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)

def run_get_batch(dataset, batch_size, context_length, device):
    return _training.run_get_batch(dataset, batch_size, context_length, device)

def run_save_checkpoint(model, optimizer, iteration, out):
    return _training.run_save_checkpoint(model, optimizer, iteration, out)

def run_load_checkpoint(src, model, optimizer):
    return _training.run_load_checkpoint(src, model, optimizer)

def get_tokenizer(vocab, merges, special_tokens=None):
    return _bpe.get_tokenizer(vocab, merges, special_tokens)

def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    return _bpe.run_train_bpe(input_path, vocab_size, special_tokens, **kwargs)
