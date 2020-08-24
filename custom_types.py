import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Union, Callable, Type, Iterator
from enum import Enum, unique
import torch.optim.optimizer


N = type(None)
V = np.array
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Union[T, N]
TNS = Union[TS, N]

V_Mesh = Tuple[V, V]
T_Mesh = Tuple[T, T]
D = torch.device
CPU = torch.device('cpu')
CUDA = lambda x: torch.device(f'cuda:{x}')

Optimizer = torch.optim.Adam
NoiseT = Union[int, Tuple[Union[bool, int], ...]]

@unique
class NoiseType(Enum):
    ALL_ZEROS = 0
    RANDOM = 1


