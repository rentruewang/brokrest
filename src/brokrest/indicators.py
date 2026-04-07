# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
import typing

import torch

__all__ = ["Indicator", "Rsi", "Ema", "Macd", "BollingerBand"]


class Indicator(typing.Protocol):
    """
    `Signal` is a callable that converts the raw datapoint into some signals.
    It must have the same number of datapoints, matching the original input.
    """

    def __call__(self, data: torch.Tensor, /) -> torch.Tensor: ...


@dcls.dataclass(frozen=True)
class Rsi(Indicator):
    window: int = 14

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"{self.window=} should be a positive number.")

    @typing.override
    def __call__(self, data: torch.Tensor, /) -> torch.Tensor:
        delta = data[1:] - data[:-1]
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.unfold(0, self.window, 1).mean(-1)
        avg_loss = loss.unfold(0, self.window, 1).mean(-1)

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


@dcls.dataclass
class Ema(Indicator):
    decay: float = 0.9

    def __post_init__(self):
        if not 0 <= self.decay <= 1:
            raise ValueError(f"{self.decay=} should be a number between 0 and 1.")

    @typing.override
    def __call__(self, data: torch.Tensor, /) -> torch.Tensor:
        kernel = self.decay * ((1 - self.decay) ** torch.arange(len(data)))
        kernel = kernel.flip(0)

        return convolve(data, kernel)


@dcls.dataclass
class Macd(Indicator):
    """
    MACD signal is just EMA_fast - EMA_slow.
    """

    fast: float = 12
    slow: float = 26

    @typing.override
    def __call__(self, data: torch.Tensor, /) -> torch.Tensor:
        return Ema(1 / self.fast)(data) - Ema(1 / self.slow)(data)


@dcls.dataclass(frozen=True)
class BollingerBand(Indicator):
    """
    Bollinger band is a lower, middle, upper band.
    """

    window: int = 20
    num_std: float = 1.5

    @typing.override
    def __call__(self, prices: torch.Tensor):
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


def convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a) < len(b):
        a, b = b, a

    return _convolve(a, b)


def _convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(a) >= len(b)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Both arrays should have ndim=1. {a.ndim=}, {b.ndim=}.")

    # pad only on the left (causal)
    padded = torch.cat([torch.zeros([len(b) - 1, *a.shape[1:]]), a])
    # shape: (len(longer), *b.shape)
    rolling_a = padded.unfold(0, len(b), 1)

    return (rolling_a * b.unsqueeze(0)).sum(dim=1)
