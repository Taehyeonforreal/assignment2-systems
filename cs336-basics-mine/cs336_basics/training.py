import os
import torch
from torch import Tensor
from typing import Iterable
from cs336_basics.transformer import run_softmax
import math

# Cross Entropy
def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # input : run_transformer_lm 출력, target : 실제 다음 단어

    batch_size = inputs.shape[0]

    # log-softmax 직접 계산
    # 수치 안전성 위해, 각 마지막 dim 마다 x_max로 빼버려. 지수함수 계산 특성
    x_max = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(inputs - x_max).sum(dim=-1)) + x_max.squeeze(-1)


    losses = torch.zeros(batch_size)
    for i in range(batch_size):
        correct_word_id = targets[i]           # i번째 예시의 정답 단어 ID
        # 그 단어의 확률인데, log 취하고 - 붙이기.
        log_prob = inputs[i][correct_word_id] - log_sum_exp[i]  
        losses[i] = -log_prob
    
    # 평균
    return losses.mean()


# Gradient Clipping, 그래디언트 폭주 방지
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # parameters는 generator 이기에, list로 만들어 재사용 할 수 있게
    params = list(parameters)
    
    # 1. 모든 파라미터의 gradient를 모아서 전체 L2 norm 계산
    total_norm_sq = 0.0
    for param in params:
        if param.grad is not None:
            total_norm_sq += param.grad.pow(2).sum().item()
    total_norm = total_norm_sq ** 0.5

    # 2. norm이 max_l2_norm을 넘으면 scaling
    # scaling을 하면, max_l2_norm을 넘던 것을 딱 max_l2_norm으로 맞춤
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)  # 1e-6은 0으로 나누기 방지
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale)



# AdamW, 클래스만 반환하기
def get_adamw_cls():
    return torch.optim.AdamW

# Learning Rate Sceduling, cosine annealing schedule 사용.
# 식은 Assignment instruction에 나와있는거 보기
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    
    # 구간 1: Warmup
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    
    # 구간 3: Cosine 이후
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # 구간 2: Cosine decay
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_value


# batch random sampling from dataset
def run_get_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str
):
    # 1. 랜덤 시작 위치 batch_size개 샘플링
    # context_length 만큼 읽어야 하니까 max_start를 제한
    max_start = len(dataset) - context_length
    start_indices = torch.randint(0, max_start, (batch_size,))
    # 이렇게 하면, batch_size 만큼 숫자를 고르고 이거를 batch_size length의 1D tensor로.

    # 2. 각 시작 위치에서 input, target 추출
    inputs  = torch.zeros(batch_size, context_length, dtype=torch.long)
    targets = torch.zeros(batch_size, context_length, dtype=torch.long)

    for i, start in enumerate(start_indices):
        start = start.item()  # tensor → int 변환
        inputs[i]  = torch.tensor(dataset[start : start + context_length])
        targets[i] = torch.tensor(dataset[start+1 : start + context_length + 1])

    # 3. 지정된 device로 이동
    return inputs.to(device), targets.to(device)


# CheckPoint 만들기
def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out
) -> None:
    checkpoint = {
        # model.state_dict() : 모델의 모든 가중치를 dictinoary로 꺼냄
        # optimizer.state.dict() : AdamW 의 변수들도 꺼내야 함
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out) # out은 파일 경로, 파일에 dictionary 저장


def run_load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src) # save한 파일 불러오기
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']

