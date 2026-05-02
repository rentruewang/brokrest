# Copyright (c) The BrokRest Authors - All Rights Reserved

import typing

import numpy as np
import pandas as pd
import yahooquery as yq

from brokrest.topos import LeftCandle

__all__ = ["load_yahooquery"]


Interval = typing.Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]

Period = typing.Literal[
    "1d", "5d", "7d", "60d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]


def load_yahooquery(
    symbol: str = "btc", *, interval: Interval = "1d", period: Period = "ytd"
) -> LeftCandle:
    if interval not in (ok := typing.get_args(Interval)):
        raise ValueError(f"{interval=} not in accepted values: {ok}")

    if period not in (ok := typing.get_args(Period)):
        raise ValueError(f"{period=} not in accepted values: {ok}")

    return _yq_load(symbol=symbol, interval=interval, period=period)


def _yq_load(symbol: str, interval: Interval, period: Period) -> LeftCandle:
    btc = yq.Ticker(symbols=symbol, asynchronous=True)

    df = btc.history(period=period, interval=interval)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    lc = LeftCandle(
        enter=np.array(df["open"].values.astype("float32")),
        exit=np.array(df["close"].values.astype("float32")),
        low=np.array(df["low"].values.astype("float32")),
        high=np.array(df["high"].values.astype("float32")),
        start=np.array(df.index.values.astype("datetime64[s]").astype("int64")),
    )
    lc.start -= lc.start.min()
    return lc
