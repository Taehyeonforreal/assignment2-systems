from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Triton 커널: 프로그램 하나 = 배치 1개 + 쿼리 타일 1개

@triton.jit     # 이 함수는 GPU 커널로 컴파일하라
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,  # PyTorch처럼 텐서 받는게 아니라, 포인터만 받음
    stride_qb, stride_qq, stride_qd,    # 각 축(batch/query/d) 을 한 칸 이동할 때의 stride
    stride_kb, stride_kk, stride_kd,    # 포인터만 받으니까, 한 칸 이동할 때 몇 바이트 이동하는지 알려줘야해
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    Nq, Nk, scale,
    B_q: tl.constexpr, B_k: tl.constexpr, d: tl.constexpr,  # 타일크기/model 차원은 컴파일 타임 상수
    is_causal: tl.constexpr,
    out_dtype: tl.constexpr,
):
    tile_q = tl.program_id(0)   # 몇 번째 쿼리 타일
    batch  = tl.program_id(1)   # 몇 번째 배치
    q_start = tile_q * B_q      # 이 프로그램이 담당하는 Q의 시작 행

    # Q 타일을 HBM → SRAM으로 로드 (이 프로그램 전체에서 고정)
    Q_blk = tl.make_block_ptr(
        Q_ptr + batch * stride_qb,      # 어디서부터 시작하는 메모리인가
        shape=(Nq, d),                  # 전체 영역의 크기
        strides=(stride_qq, stride_qd), # 이 영역 안에서 이동하는 법
        offsets=(q_start, 0),           # 이 영역 안에서 내가 볼 작은 창문의 시작점
        block_shape=(B_q, d),           # 그 창문의 크기
        order=(1, 0),
    )
    Q_i = tl.load(Q_blk, boundary_check=(0,)).to(tl.float32)  # SRAM으로 이동

    # 온칩 버퍼 — SRAM에 상주하면서 키 타일 루프 내내 업데이트
    O_i = tl.zeros((B_q, d), dtype=tl.float32)
    l_i = tl.zeros((B_q,),   dtype=tl.float32)
    m_i = tl.full( (B_q,),   float('-inf'), dtype=tl.float32)

    # K/V 블록 포인터 — 루프마다 tl.advance로 B_k씩 내려감
    K_blk = tl.make_block_ptr(
        K_ptr + batch * stride_kb,
        shape=(Nk, d), 
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), 
        block_shape=(B_k, d),
        order=(1, 0),
    )
    V_blk = tl.make_block_ptr(
        V_ptr + batch * stride_vb,
        shape=(Nk, d), 
        strides=(stride_vk, stride_vd),
        offsets=(0, 0), 
        block_shape=(B_k, d), 
        order=(1, 0),
    )

    # 키 타일 루프 — PyTorch 버전의 j 루프와 동일한 역할
    for j in range(0, Nk, B_k):
        K_j = tl.load(K_blk, boundary_check=(0,))  # SRAM으로 이동
        V_j = tl.load(V_blk, boundary_check=(0,))  # SRAM으로 이동

        S_ij = tl.dot(Q_i, tl.trans(K_j).to(tl.float32), out_dtype=tl.float32) * scale  # (B_q, B_k)

        if is_causal:
            q_idx = (q_start + tl.arange(0, B_q))[:, None] # (B_q, 1)
            k_idx = (j      + tl.arange(0, B_k))[None, :] # (1, B_k)
            S_ij  = tl.where(k_idx > q_idx, -1e6, S_ij)

        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1)) # (B_q,)
        P_tilde = tl.exp(S_ij - m_new[:, None]) # (B_q, B_k)
        correction = tl.exp(m_i - m_new) # (B_q,)
        l_i = correction * l_i + tl.sum(P_tilde, axis=1)
        O_i = correction[:, None] * O_i + tl.dot(P_tilde.to(K_j.dtype), V_j, out_dtype=tl.float32)
        m_i = m_new

        K_blk = tl.advance(K_blk, (B_k, 0)) # 다음 타일로 포인터 이동
        V_blk = tl.advance(V_blk, (B_k, 0))

    # 정규화 + logsumexp
    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    # 결과를 HBM에 저장
    O_blk = tl.make_block_ptr(
        O_ptr + batch * stride_ob,
        shape=(Nq, d), 
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0), 
        block_shape=(B_q, d), 
        order=(1, 0),
    )
    tl.store(O_blk, O_i.to(out_dtype), boundary_check=(0,))

    L_blk = tl.make_block_ptr(
        L_ptr + batch * stride_lb,
        shape=(Nq,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(B_q,),
        order=(0,),
    )
    tl.store(L_blk, L_i, boundary_check=(0,))



# Triton autograd.Function

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        assert HAS_TRITON, "Triton이 설치되지 않았습니다"
        B, Nq, d = q.shape
        Nk = k.shape[1]
        B_q, B_k = 32, 32
        scale = d ** -0.5

        O = torch.zeros_like(q)
        L = torch.zeros(B, Nq, device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(Nq, B_q), B)   # (쿼리 타일 수, 배치 크기)
        # 이 grid를 기반으로, flash_fwd_kernel이 (쿼리 타일 수 * 배치 크기) 만큼 병렬로 실행됨
        # cdiv = ceiling division. 나눗셈을 올림을 해서, 타일 갯수 맞추기

    
        triton_dtype = {
            torch.float32: tl.float32,
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
        }[q.dtype]

        flash_fwd_kernel[grid](                     # 인자로 넘기는 것의 종류가 4개
            q, k, v, O, L,                          # 1. 텐서 자체
            q.stride(0), q.stride(1), q.stride(2),  # 2. stride
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, scale,                          # 3. 일반 런타임 값
            B_q=B_q, B_k=B_k, d=d,                  # 4. 컴파일 타임 상수. constexpr
            is_causal=is_causal,
            out_dtype=triton_dtype,
        )

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        # PyTorch 버전 backward 재사용 (recomputation 기반)
        # 왜? forward는 online softmax 때문에 Triton의 이득이 크지만, 
        # backward는 softmax gradient 계산이 병렬화 가능해서, non-tricky. Triton의 이득이 크지 않음
        q, k, v, O, L = ctx.saved_tensors
        return FlashAttentionPyTorch.backward(ctx, dO)


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        B, Nq, d = q.shape
        Nk = k.shape[1]
        scale = d ** -0.5
        B_q, B_k = 32, 32
        # let tile size 32, 32

        O = torch.zeros(B, Nq, d, device=q.device, dtype=q.dtype) # for ouput
        L = torch.zeros(B, Nq, device=q.device, dtype=torch.float32) # logsumexp, for backward

        for i in range(0, Nq, B_q): # Query (row) loop
            Q_i = q[:, i:i+B_q, :] # (B, B_q, d)
            bq = Q_i.shape[1] 

            O_i = torch.zeros(B, bq, d, device=q.device, dtype=torch.float32)
            l_i = torch.zeros(B, bq, device=q.device, dtype=torch.float32)
            m_i = torch.full((B, bq), float('-inf'), device=q.device, dtype=torch.float32)

            for j in range(0, Nk, B_k): # Key (column) loop
                K_j = k[:, j:j+B_k, :] # (B, B_k, d)
                V_j = v[:, j:j+B_k, :] # (B, B_k, d)

                S_ij = Q_i.float() @ K_j.float().transpose(-1, -2) * scale  # (B, bq, bk)

                if is_causal: # 인과 관계 위해 마스크 필터링
                    q_idx = torch.arange(i, i+bq, device=q.device).unsqueeze(1) # (bq, 1)
                    k_idx = torch.arange(j, j+K_j.shape[1], device=q.device).unsqueeze(0) # (1, bk)
                    S_ij = S_ij.masked_fill(k_idx.gt(q_idx).unsqueeze(0), float('-inf')) # 이렇게 하면 exp -> 0

                m_new = torch.maximum(m_i, S_ij.amax(dim=-1)) # (B, bq)
                P_tilde = torch.exp(S_ij - m_new.unsqueeze(-1)) # (B, bq, bk), 중간계산 결과(tilde)
                correction = torch.exp(m_i - m_new) # (B, bq), 바뀐 max에 맞춰 보정
                l_i = correction * l_i + P_tilde.sum(dim=-1) # (B, bq), 가중치 누적합. softmax 분모
                O_i = correction.unsqueeze(-1) * O_i + P_tilde @ V_j.float() # (B, bq, d) # softmax 분자
                m_i = m_new

            O_i = O_i / l_i.unsqueeze(-1) # broadcasting으로 나눗셈 가능. attention output
            L_i = m_i + torch.log(l_i) # logsumexp = m + log(l). recomputation에 쓰이는 정규화 수(softmax 분모 합쳐놓은것)

            O[:, i:i+B_q, :] = O_i.to(q.dtype)
            L[:, i:i+B_q]    = L_i

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        B, Nq, d = q.shape
        Nk = k.shape[1]
        scale = d ** -0.5
        B_q, B_k = 32, 32

        dQ = torch.zeros_like(q, dtype=torch.float32)
        dK = torch.zeros_like(k, dtype=torch.float32)
        dV = torch.zeros_like(v, dtype=torch.float32)

        for i in range(0, Nq, B_q):
            Q_i  = q[:, i:i+B_q, :]
            dO_i = dO[:, i:i+B_q, :]
            O_i  = O[:, i:i+B_q, :]
            L_i  = L[:, i:i+B_q] # forward에서 저장한 logsumexp

            # D_i = rowsum(O@dO): softmax gradient 계산에 필요한 보정값.
            D_i = (O_i.float() * dO_i.float()).sum(dim=-1) # 여기 유도 다시보기.

            dQ_i = torch.zeros_like(Q_i, dtype=torch.float32)

            for j in range(0, Nk, B_k):
                # 여기는 seqeuntial 하지 않음! forward와 달리 병렬화 가능. Q_i만 합산해주면 돼
                K_j = k[:, j:j+B_k, :]
                V_j = v[:, j:j+B_k, :]

                # P 재계산 (recomputation): forward에서 저장한 L로 P를 복원
                S_ij = Q_i.float() @ K_j.float().transpose(-1, -2) * scale # (B, bq, bk)
                if is_causal:
                    q_idx = torch.arange(i, i+Q_i.shape[1], device=q.device).unsqueeze(1)
                    k_idx = torch.arange(j, j+K_j.shape[1], device=q.device).unsqueeze(0)
                    S_ij = S_ij.masked_fill(k_idx.gt(q_idx).unsqueeze(0), float('-inf'))
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1)) # (B, bq, bk)

                # dV: V의 gradient
                dV[:, j:j+B_k, :] += P_ij.transpose(-1, -2) @ dO_i.float() # (B, bk, d)

                # dP → dS → dQ, dK
                dP_ij = dO_i.float() @ V_j.float().transpose(-1, -2) # (B, bq, bk)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1)) # (B, bq, bk)

                dQ_i += dS_ij @ K_j.float() * scale # (B, bq, d)
                dK[:, j:j+B_k, :] += dS_ij.transpose(-1, -2) @ Q_i.float() * scale # (B, bk, d)

            dQ[:, i:i+B_q, :] = dQ_i

        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype), None
