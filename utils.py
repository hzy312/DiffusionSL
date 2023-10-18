"""
@Date  : 2022/12/20
@Time  : 12:46
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import random
import numpy as np
import torch
from torch.backends import cudnn
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule
from torch.optim.optimizer import Optimizer
from typing import Optional


def ensure_reproducibility(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"set all seeds to {seed}")
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_lr_scheduler(
        name: str,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: Optional[int] = None
):
    if name == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif name == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    elif name == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif name == 'cosine_hard_restart':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles
        )
    else:
        scheduler = get_constant_schedule(optimizer=optimizer)

    return scheduler