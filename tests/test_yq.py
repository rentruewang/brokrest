# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest

from brokrest.data.yquery import load_yahooquery
from brokrest.topos import Candle


@pytest.fixture
def default_candles():
    return load_yahooquery()


def test_yquery(default_candles):
    assert isinstance(default_candles, Candle)
