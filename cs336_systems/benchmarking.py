from __future__ import annotations

import time
import torch
from cs336_basics.module import TransformerLM  # A1 wrapper (내 구현)

# A2_OVERVIEW Table 1 — 과제 공통 모델 크기 정의
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE     = 10_000
BATCH_SIZE     = 4
CONTEXT_LENGTH = 512


def timed_run(fn, *, warmup: int = 3, iters: int = 10, device: str = "cpu") -> tuple[float, float]:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def sync():
        if use_cuda:
            torch.cuda.synchronize()
            # since GPU execution is asynchronous
            # Pytorch는 GPU에게 명령을 보내고 바로 다음 줄로 넘어감
            # 즉 GPU 시간을 측정하기 전에 synchronize를 호출하여 모든 GPU 작업이 완료될 때까지 기다려야 한다.

    # warmup. 컴파일, 메모리 할당, 캐시 등의 일회성 비용 빼
    for _ in range(warmup):
        fn()
    sync()

    # Checking times. 10번 iter. mean/std 구함. ms 단위로
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def make_model(size: str, device: str) -> TransformerLM:
    cfg = MODEL_CONFIGS[size]
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        **cfg,
    )
    return model.to(device)


def make_bench_fn(model: TransformerLM, mode: str, device: str):
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)

    # forward mode -> only forward pass
    if mode == "forward":
        def fn():
            with torch.no_grad():
                model(input_ids)

    # backward mode -> forward + loss.backward()
    elif mode == "backward":
        def fn():
            model.zero_grad()
            logits = model(input_ids)
            loss = logits.mean()
            loss.backward()
            # backward는 숫자 하나만 호출 가능.
            # 이때 mean을 쓴 이유는 의미있는 학습 보다는 backward pass의 속도를 측정하기 위함

    # optimizer mode -> forward + backward + optimizer.step()
    elif mode == "optimizer":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def fn():
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = logits.mean()
            loss.backward()
            optimizer.step()

    else:
        raise ValueError(f"mode must be forward/backward/optimizer, got {mode!r}")

    return fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size",   choices=list(MODEL_CONFIGS), default="small")
    parser.add_argument("--mode",   choices=["forward", "backward", "optimizer"], default="forward")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters",  type=int, default=10)
    args = parser.parse_args()

    model = make_model(args.size, args.device)
    fn    = make_bench_fn(model, args.mode, args.device)
    mean, std = timed_run(fn, warmup=args.warmup, iters=args.iters, device=args.device)

    print(f"size={args.size}  mode={args.mode}  device={args.device}")
    print(f"  {mean:.2f} ± {std:.2f} ms")
