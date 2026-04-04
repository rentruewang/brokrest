# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest

from brokrest import topos
from brokrest.data import yquery


@pytest.fixture
def default_candles():
    return yquery.load()


def test_yquery(default_candles):
    assert isinstance(default_candles, topos.Candle)
