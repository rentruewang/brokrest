# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest
import torch

from brokrest.data.yquery import load_yahooquery
from brokrest.topos import Candle
from brokrest.topos._turnaround import cumsum_with_reset


@pytest.fixture
def tensor():
    return torch.arange(10)


@pytest.fixture
def reset():
    return torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0]).bool()


@pytest.fixture
def candles():
    return load_yahooquery(interval="1h")


def test_cumsum_with_reset(tensor: np.ndarray, reset: np.ndarray):
    answer = [0, 1, 3, 6, 4, 9, 15, 22, 8, 17]

    assert cumsum_with_reset(tensor, reset).round().int().tolist() == answer


def test_turnaround_segments(candles: Candle):
    segments = candles.to_turnaround_segments()
    signs = segments.dy.sign()

    assert (
        signs[:-1] != signs[1:]
    ).all(), "There shouldn't be consecutive equivalent signs."
