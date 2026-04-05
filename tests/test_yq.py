# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest

from brokrest.data.yquery import load
from brokrest.topos import Candle


@pytest.fixture
def default_candles():
    return load()


def test_yquery(default_candles):
    assert isinstance(default_candles, Candle)
