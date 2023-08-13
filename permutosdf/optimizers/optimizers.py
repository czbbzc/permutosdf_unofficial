"""
Optimizers class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from nerfstudio.configs import base_config
from nerfstudio.utils import writer

from nerfstudio.engine.optimizers import OptimizerConfig

@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with AdamW"""

    _target: Type = torch.optim.AdamW
    amsgrad=False
    betas=(0.9, 0.99)
    eps=1e-15
    weight_decay: float = 0.0
    """The weight decay to use."""
    lr=1e-3