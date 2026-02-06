# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["Signal", "Rsi", "Ema", "Macd", "BollingerBand"]


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
    decay: float = 0.9

    def __post_init__(self):
        if not 0 <= self.decay <= 1:
            raise ValueError(f"{self.decay=} should be a positive number.")

    @typing.override
    def __call__(self, data: Tensor, /) -> Tensor:
        kernel = self.decay * ((1 - self.decay) ** torch.arange(len(data)))
        kernel = kernel.flip(0)

        return convolve(data, kernel)


@dcls.dataclass
class Macd(Signal):
    """
    MACD signal is just EMA_fast - EMA_slow.
    """

    fast: float = 12
    slow: float = 26

    @typing.override
    def __call__(self, data: Tensor, /) -> Tensor:
        return Ema(self.fast)(data) - Ema(self.slow)(data)


@dcls.dataclass(frozen=True)
class BollingerBand(Signal):
    """
    Bollinger band is a lower, middle, upper band.
    """

    window: int = 20
    num_std: float = 1.5

    @typing.override
    def __call__(self, prices: Tensor):
        """
        Compute Bollinger Bands in PyTorch.
        """

        kernel = torch.ones(self.window) / self.window

        # Rolling mean.
        middle_band = convolve(prices, kernel)

        # Variance: E[x^2] - (E[x])^2
        prices_sq = prices**2
        mean_sq = convolve(prices_sq, kernel)
        variance = mean_sq - middle_band**2
        std = torch.sqrt(variance.clamp(min=1e-8))

        upper_band = middle_band + self.num_std * std
        lower_band = middle_band - self.num_std * std

        return torch.stack([middle_band, upper_band, lower_band], dim=-1)


def convolve(a: Tensor, b: Tensor) -> Tensor:
    assert a.ndim == 1 and b.ndim == 1

    padded = F.pad(a, (len(b) - 1, 0))  # pad only on the left (causal)

    windows = padded.unfold(0, len(b), 1)  # shape: (len(longer), k_len)

    return (windows * b).sum(dim=1)
