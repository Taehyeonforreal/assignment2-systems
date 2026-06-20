
import torch
from torch import Tensor
import torch.nn as nn

# Softmax.
def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    x = in_features - in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Linear Model. weight * input
def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    return in_features @ weights.T

# embdedding. integer list -> vector
def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids]

# RMSNorm. (substitution of LayerNorm)
def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    rms = torch.sqrt(in_features.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (in_features / rms) * weights

# SiLU = x * sigma(x).
def run_silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)

# SwiGLU(x) = (SiLU(x @ W1.T) ⊙ (x @ W3.T)) @ W2.T
def run_swiglu(d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    gate = run_silu(in_features @ w1_weight.T)
    up   = in_features @ w3_weight.T
    return (gate * up) @ w2_weight.T


# RoPE Class
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # max_seq_len : 이 모델이 처리할 수 있는 최대 토큰 수

        # 차원마다 frequency 다르게 만들기
        i = torch.arange(0, d_k // 2, device=device)
        freqs = 1.0 / (theta ** (2 * i / d_k))  # (d_k/2,)
        
        # 가능한 각도 모음. 이때 unsqueeze(i) = i-dim에 1차원 추가, *는 broadcasting 이후 element-wise 곱 의미
        # angles : 모든 position 모든 쌍의 각이 저장
        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_seq_len, d_k/2)
        
        # register_buffer: 모델 저장과 GPU 로드에 포함됨. 근데 nn.Parameter와 달리 학습은 X -> sin/cos는 고정 상수니까 이걸로
        # self.cos, self.sin으로 텐서들에 접근 가능
        self.register_buffer('cos', torch.cos(angles))  # (max_seq_len, d_k/2)
        self.register_buffer('sin', torch.sin(angles))  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions: (..., seq_len)
        # 원하는 position의 cos, sin만 가져오기 (보통 seq_len 만큼)
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)
        
        # 짝수/홀수 인덱스 분리. ... : 앞의 차원 유지, 마지막 차원에서 0or1부터 2칸씩
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2)
        
        # 회전 적용
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        
        # 다시 합치기. stack으로 even odd 쌓은 후 flatten으로 원복
        out = torch.stack([out_even, out_odd], dim=-1)
        return out.flatten(-2)  # (..., seq_len, d_k)

def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions) ##nn.Module의 __call__ method 때문에, rope.forward 대신 rope써도 됨. 


def run_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
    d_k = Q.shape[-1]
    # d_k : 각 토큰을 표현하는 벡터의 차원
    # Q K_T 행렬곱으로 유사도 측정, sqrt(d_k)로 scaling (각 토큰 N~(0,1)이라 할때, 분산 : d_k)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # 2. mask 적용 (False인 위치를 -무한대로)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    # 3. softmax로 확률 변환
    attn_weights = run_softmax(scores, dim=-1)
    
    # 4. V와 가중합
    return attn_weights @ V
    
# 이건 멀티헤드 어텐션 + RoPE 구현하기 전 단계. 그냥 멀티헤드만 일단 구현한것. 실제로 안씀
def run_multihead_self_attention(
    d_model: int, num_heads: int,
    q_proj_weight: Tensor, k_proj_weight: Tensor,
    v_proj_weight: Tensor, o_proj_weight: Tensor,
    in_features: Tensor
) -> Tensor:
    
    batch, seq_len, _ = in_features.shape
    d_k = d_model // num_heads  # 각 head의 차원

    # 1. Q, K, V 한번에 projection
    Q = in_features @ q_proj_weight.T  # (batch, seq_len, d_model)
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # 2. num_heads개로 분리
    # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
    Q = Q.view(batch, seq_len, num_heads, d_k)
    K = K.view(batch, seq_len, num_heads, d_k)
    V = V.view(batch, seq_len, num_heads, d_k)

    # 3. head 차원을 앞으로 (Attention을 head별로 병렬 계산하기 위해)
    # (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # 3-1. MASK 추가
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

    # 4. 각 head에서 Attention 계산
    out = run_scaled_dot_product_attention(Q, K, V, mask = mask)
    # (batch, num_heads, seq_len, d_k)

    # 5. head 합치기
    # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
    out = out.transpose(1, 2)
    # (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
    out = out.contiguous().view(batch, seq_len, d_model)

    # 6. output projection
    return out @ o_proj_weight.T


def run_multihead_self_attention_with_rope(
    d_model: int, num_heads: int, max_seq_len: int, theta: float,
    q_proj_weight: Tensor, k_proj_weight: Tensor,
    v_proj_weight: Tensor, o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions = None
) -> Tensor:
    
    batch, seq_len, _ = in_features.shape
    d_k = d_model // num_heads  # 각 head의 차원

    # 1. Q, K, V 한번에 projection
    Q = in_features @ q_proj_weight.T  # (batch, seq_len, d_model)
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # 2. num_heads개로 분리
    # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
    Q = Q.view(batch, seq_len, num_heads, d_k)
    K = K.view(batch, seq_len, num_heads, d_k)
    V = V.view(batch, seq_len, num_heads, d_k)

    # 3. head 차원을 앞으로 (Attention을 head별로 병렬 계산하기 위해)
    # (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # 3-1. MASK 추가
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

    # 4. RoPE 
    # 4.1 (batch, seq_len) → (batch, 1, seq_len) for broadcasting
    # RoPE 과정에서, Broadcasting 예정
    positions = token_positions.unsqueeze(1)

    # 4.2 Q와 K만 RoPE
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_features.device)
    Q = rope(Q, positions)
    K = rope(K, positions)

    # ! GPT-2와 달리, Position Embedding이 각 Transformer Block 내에서 진행
    # 안그러면, Block 내에서 Weight 내적때 상대위치 다 망가져버리니까

    # 5. 각 head에서 Attention 계산
    out = run_scaled_dot_product_attention(Q, K, V, mask = mask)
    # (batch, num_heads, seq_len, d_k)

    # 6. head 합치기
    # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
    out = out.transpose(1, 2)
    # (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
    out = out.contiguous().view(batch, seq_len, d_model)

    # 7. output projection
    return out @ o_proj_weight.T


# Transformer 구조 (Dropout 없이)
# 입력 -> RMSNorm -> Multi-head Attention with RoPE -> Residual Add
# -> RMSNorm -> SwiGLU Feed Forward Network -> Residual Add -> 출력
def run_transformer_block(
    d_model: int, num_heads: int, d_ff: int,
    max_seq_len: int, theta: float,
    weights: dict, in_features: Tensor
) -> Tensor:
    batch, seq_len, _ = in_features.shape

    # token positions 생성
    token_positions = torch.arange(seq_len, device=in_features.device)
    token_positions = token_positions.unsqueeze(0).expand(batch, -1)

    # 1. Pre-Norm
    normed = run_rmsnorm(d_model, 1e-6, weights['ln1.weight'], in_features)
    # 2. Attention
    attn_out = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.output_proj.weight'],
        normed, token_positions
    )
    # 3. Residual
    x = in_features + attn_out  # Residual Connection

    # 4. Pre-Norm
    normed = run_rmsnorm(d_model, 1e-6, weights['ln2.weight'], x)
    # 5. FFN
    ffn_out = run_swiglu(
        d_model, d_ff,
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        normed
    )
    # 6. Residual
    x = x + ffn_out  # Residual Connection

    return x


# Transformer 완성품
def run_transformer_lm(
    vocab_size: int, context_length: int, d_model: int,
    num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
    weights: dict, in_indices: Tensor
) -> Tensor:
    batch, seq_len = in_indices.shape

    # 1. Embedding: 토큰 ID → 벡터
    x = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)

    # 2. Transformer Block N번 반복
    for i in range(num_layers):
        layer_weights = {
            'ln1.weight':              weights[f'layers.{i}.ln1.weight'],
            'ln2.weight':              weights[f'layers.{i}.ln2.weight'],
            'attn.q_proj.weight':      weights[f'layers.{i}.attn.q_proj.weight'],
            'attn.k_proj.weight':      weights[f'layers.{i}.attn.k_proj.weight'],
            'attn.v_proj.weight':      weights[f'layers.{i}.attn.v_proj.weight'],
            'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
            'ffn.w1.weight':           weights[f'layers.{i}.ffn.w1.weight'],
            'ffn.w2.weight':           weights[f'layers.{i}.ffn.w2.weight'],
            'ffn.w3.weight':           weights[f'layers.{i}.ffn.w3.weight'],
        }
        x = run_transformer_block(
            d_model, num_heads, d_ff, context_length, rope_theta, layer_weights, x
        )

    # 3. 마지막 RMSNorm -> 수치 안정화
    x = run_rmsnorm(d_model, 1e-6, weights['ln_final.weight'], x)

    # 4. lm_head: 벡터 → vocab_size
    # lm_head_weight는 (vocab_size, d_model)
    # x = (batch, seq_len, d_model)니까, 
    # seq_len 안에 있는 단어들이, vocab_size 안에 있는 모든 단어와 내적하며 유사도 계산
    return x @ weights['lm_head.weight'].T
