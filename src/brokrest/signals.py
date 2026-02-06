# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F


class Signal(Protocol):
    def __call__(self, data: Tensor, /) -> Tensor: ...


@dcls.dataclass(frozen=True)
class Rsi(Signal):
    window: int = 14
    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"{self.window=} should be a positive number.")

    @typing.override
    def __call__(self, data: Tensor, /) -> Tensor:
        delta = data[1:] - data[:-1]
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.unfold(0, self.window, 1).mean(-1)
        avg_loss = loss.unfold(0, self.window, 1).mean(-1)

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


@dcls.dataclass
class Ema(Signal):
    decay: float = 14

    def __post_init__(self):
        if self.decay <= 0:
            raise ValueError(f"{self.decay=} should be a positive number.")

    @typing.override
    def __call__(self, data: Tensor, /) -> Tensor:
        # Determine kernel size dynamically or fixed
        size = data.size(-1)

        kernel = self.decay * ((1 - self.decay) ** torch.arange(size))
        kernel = kernel.flip(0).view(1, 1, -1)  # shape (1,1,L) for conv1d

        # Apply convolution with padding to make it causal
        padding = size - 1
        y = F.conv1d(data, kernel, padding=padding)

        # Trim the extra padded part to match input length
        return y[..., -size:]
