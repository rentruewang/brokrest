# Copyright (c) The BrokRest Authors - All Rights Reserved

import typing

import pandas as pd
import torch
import yahooquery as yq

from brokrest.topos import LeftCandle

__all__ = ["load"]

_MINUTE = "m"
_HOUR = "h"
_MONTH = "mo"
_WEEK = "wk"
_YEAR = "y"

Interval = typing.Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]

Period = typing.Literal[
    "1d", "5d", "7d", "60d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]


def load(
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

    return LeftCandle(
        enter=torch.from_numpy(df["open"].values.astype("float32")),
        exit=torch.from_numpy(df["close"].values.astype("float32")),
        low=torch.from_numpy(df["low"].values.astype("float32")),
        high=torch.from_numpy(df["high"].values.astype("float32")),
        start=torch.from_numpy(df.index.values.astype("datetime64[s]").astype("int64")),
    )
