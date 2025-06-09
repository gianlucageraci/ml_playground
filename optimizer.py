import math
from typing import List
from layers import LinearLayer

import math

class LearningRateDecay:
    def __init__(self, decay_type: str, base_lr: float, total_epochs: int):
        self.decay_type = decay_type
        self.base_lr = base_lr
        self.total_epochs = total_epochs
    
    def get_lr(self, t: int):
        if self.decay_type == 'cosine':
            return self.base_lr * (1 + math.cos(t * math.pi / self.total_epochs))
        elif self.decay_type == 'exponential':
            return self.base_lr * (0.9 ** t)  # Example exponential decay
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        

class Optimizer:
    def __init__(self, lr: float, adam_args: dict = None, lr_decay_type: str = "None", total_epochs: int = None):
        self.base_lr = lr
        self.adam_args = adam_args
        self.lr_decay_type = lr_decay_type
        self.total_epochs = total_epochs
    
    def step(self, t:int, layers: List[LinearLayer]):
        self.current_lr = self.get_lr(t)
        for layer in layers:
            layer.update_params(t, self.current_lr, **self.adam_args)

    def zero_grad(self, layers: List[LinearLayer]):
        for layer in layers:
            layer.zero_grad()
    
    def get_lr(self, t: int):
        if self.lr_decay_type == "None":
            return self.base_lr
        if self.lr_decay_type == 'cosine':
            return 0.5 * self.base_lr * (1 + math.cos(t * math.pi / self.total_epochs))
        elif self.lr_decay_type == 'exponential':
            return self.base_lr * (0.9 ** t)  # Example exponential decay
        else:
            raise ValueError(f"Unknown decay type: {self.lr_decay_type}")
