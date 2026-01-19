# Copyright (c) The BrokRest Authors - All Rights Reserved

import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.sampledata import stocks
from pandas import DataFrame

from brokrest.plotting import Canvas, Window
from brokrest.topos import LeftCandle

df = DataFrame(stocks.MSFT)[60:120]
df["date"] = pd.to_datetime(df["date"])

inc = df.close > df.open
dec = df.open > df.close
w = 16 * 60 * 60 * 1000  # milliseconds

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
_dates: list[float] = [v.value for v in df.date]
width = _dates[1] - _dates[0]
p = plotting.figure(
    x_axis_type="datetime",
    tools=TOOLS,
    width=1000,
    height=400,
    title="MSFT Candlestick",
    background_fill_color="#efefef",
)
p.xaxis.major_label_orientation = 0.8  # radians
candle = LeftCandle(
    enter=np.array(df.open),
    exit=np.array(df.close),
    low=np.array(df.low),
    high=np.array(df.high),
    start=np.array(_dates),
)
cv = Canvas(Window(), figure=p)
candle.draw(cv)
plotting.show(p)
