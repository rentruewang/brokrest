# Copyright (c) The BrokRest Authors - All Rights Reserved

from yahooquery import Ticker

from brokrest.topos import BothCandle

__all__ = ["load"]


def load(symbol: str = "btc") -> BothCandle:
    btc = Ticker(symbols=symbol, asynchronous=True)

    yq_data = btc.history(period="ytd", interval="15m")

    raise NotImplementedError
