# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Plotting utilities for price data and trend analysis.

Provides both matplotlib (via Canvas) and Bokeh backends.
"""

from __future__ import annotations

import dataclasses as dcls
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from brokrest.candles import Candle, CandleChart
from brokrest.painters import Box, Canvas
from brokrest.vectors import Vec2d

from .regression import PiecewiseRegression, TrendSegment, detect_trends

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "TrendPlotter",
    "plot_trends_mpl",
    "plot_trends_bokeh",
]


@dcls.dataclass
class TrendPlotter:
    """
    Plotter for price data with trend lines.
    
    Supports both matplotlib and Bokeh backends.
    """

    title: str = "Price Analysis"
    width: int = 12
    height: int = 6

    def plot_matplotlib(
        self,
        prices: np.ndarray,
        regression: PiecewiseRegression,
        show_price: bool = True,
        figsize: Optional[tuple[int, int]] = None,
    ) -> "Figure":
        """
        Plot using matplotlib.
        
        Args:
            prices: Price array
            regression: Fitted regression model
            show_price: Whether to show price line
            figsize: Optional figure size
            
        Returns:
            Matplotlib figure
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (self.width, self.height))
        ax.set_title(self.title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        # Plot prices
        if show_price:
            ax.plot(prices, color="gray", alpha=0.5, linewidth=1, label="Price")

        # Plot trend segments with connected lines
        junction_ys = []
        for i, seg in enumerate(regression.segments):
            y_start, y_end = seg.predict_range()
            if i == 0:
                junction_ys.append(y_start)
            junction_ys.append(y_end)

        for i, seg in enumerate(regression.segments):
            seg_start_y = junction_ys[i]
            seg_end_y = junction_ys[i + 1]
            is_rising = seg_end_y > seg_start_y
            color = "green" if is_rising else "red"

            ax.plot(
                [seg.start_idx, seg.end_idx],
                [seg_start_y, seg_end_y],
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

            # Mark junction points
            if i < len(regression.segments) - 1:
                next_seg = regression.segments[i + 1]
                next_rising = junction_ys[i + 2] > junction_ys[i + 1]

                if is_rising and not next_rising:
                    # Peak
                    ax.scatter([seg.end_idx], [seg_end_y], color="purple", s=100, zorder=5)
                elif not is_rising and next_rising:
                    # Valley
                    ax.scatter([seg.end_idx], [seg_end_y], color="blue", s=100, zorder=5)

        ax.legend()
        plt.tight_layout()
        return fig

    def plot_with_canvas(
        self,
        prices: np.ndarray,
        regression: PiecewiseRegression,
    ) -> Canvas:
        """
        Plot using the Canvas system (rentruewang's painter architecture).
        
        Args:
            prices: Price array
            regression: Fitted regression model
            
        Returns:
            Canvas object
        """
        canvas = Canvas(left=0, right=len(prices), interval=1)
        
        # Let the regression draw itself
        regression.plot(canvas)
        
        return canvas


def plot_trends_mpl(
    df: pd.DataFrame,
    n_segments: Optional[int] = None,
    auto_segments: bool = True,
    title: str = "Price Trend Analysis",
    figsize: tuple[int, int] = (12, 6),
) -> "Figure":
    """
    Create matplotlib plot with trend lines.

    Args:
        df: DataFrame with 'close' column
        n_segments: Number of segments (auto if None)
        auto_segments: Auto-determine segments
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    prices = df["close"].values
    regression = detect_trends(
        prices,
        n_segments=n_segments,
        auto=auto_segments,
    )

    plotter = TrendPlotter(title=title)
    return plotter.plot_matplotlib(prices, regression, figsize=figsize)


def plot_trends_bokeh(
    df: pd.DataFrame,
    n_segments: Optional[int] = None,
    auto_segments: bool = True,
    title: str = "Price Trend Analysis",
    width: int = 1200,
    height: int = 600,
    show_ohlc: bool = True,
):
    """
    Create Bokeh interactive plot with trend lines.

    Args:
        df: DataFrame with OHLC columns
        n_segments: Number of segments (auto if None)
        auto_segments: Auto-determine segments
        title: Plot title
        width: Plot width
        height: Plot height
        show_ohlc: Show OHLC candlesticks

    Returns:
        Bokeh figure
    """
    from bokeh.models import Label
    from bokeh.plotting import figure

    # Prepare dates
    if "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"])
    elif "timestamp" in df.columns:
        dates = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    else:
        dates = df.index

    # Create figure
    fig = figure(
        title=title,
        x_axis_label="Time",
        y_axis_label="Price",
        x_axis_type="datetime",
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
    )

    # Draw OHLC if requested
    if show_ohlc and all(col in df.columns for col in ["open", "high", "low", "close"]):
        if len(dates) > 1:
            interval_ms = (dates.iloc[1] - dates.iloc[0]).total_seconds() * 1000
            bar_width = interval_ms * 0.8
        else:
            bar_width = 3600 * 1000 * 0.8

        up = df["close"] >= df["open"]
        down = ~up

        # Wicks
        fig.segment(x0=dates, y0=df["high"], x1=dates, y1=df["low"], color="gray")

        # Up candles
        fig.vbar(
            x=dates[up],
            top=df["close"][up],
            bottom=df["open"][up],
            width=bar_width,
            fill_color="#26a69a",
            line_color="#26a69a",
        )

        # Down candles
        fig.vbar(
            x=dates[down],
            top=df["open"][down],
            bottom=df["close"][down],
            width=bar_width,
            fill_color="#ef5350",
            line_color="#ef5350",
        )

    # Detect trends
    prices = df["close"].values
    regression = detect_trends(prices, n_segments=n_segments, auto=auto_segments)

    # Compute junction y-values
    junction_ys = []
    for i, seg in enumerate(regression.segments):
        y_start, y_end = seg.predict_range()
        if i == 0:
            junction_ys.append(y_start)
        junction_ys.append(y_end)

    # Draw trend lines and markers
    peak_x, peak_y = [], []
    valley_x, valley_y = [], []

    for i, seg in enumerate(regression.segments):
        start_date = dates.iloc[seg.start_idx]
        end_date = dates.iloc[seg.end_idx]
        seg_start_y = junction_ys[i]
        seg_end_y = junction_ys[i + 1]
        is_rising = seg_end_y > seg_start_y

        color = "#4caf50" if is_rising else "#f44336"

        fig.line(
            x=[start_date, end_date],
            y=[seg_start_y, seg_end_y],
            color=color,
            line_width=3,
            line_dash="dashed",
            alpha=0.8,
        )

        # Mark peaks and valleys
        if i < len(regression.segments) - 1:
            next_rising = junction_ys[i + 2] > junction_ys[i + 1]

            if is_rising and not next_rising:
                peak_x.append(end_date)
                peak_y.append(seg_end_y)
            elif not is_rising and next_rising:
                valley_x.append(end_date)
                valley_y.append(seg_end_y)

        # Add R² label
        mid_idx = (seg.start_idx + seg.end_idx) // 2
        mid_date = dates.iloc[mid_idx]
        mid_y = (seg_start_y + seg_end_y) / 2

        label = Label(
            x=mid_date,
            y=mid_y,
            text=f"R²={seg.r_squared:.2f}",
            text_font_size="10px",
            text_color=color,
        )
        fig.add_layout(label)

    # Draw markers
    if valley_x:
        fig.scatter(x=valley_x, y=valley_y, size=15, color="#1565c0", legend_label="Valley")

    if peak_x:
        fig.scatter(x=peak_x, y=peak_y, size=15, color="#7b1fa2", legend_label="Peak")

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    return fig


def save_bokeh(fig, path: str | Path, title: str = "Chart"):
    """Save Bokeh figure to HTML."""
    from bokeh.io import save
    from bokeh.resources import CDN

    save(fig, filename=str(path), resources=CDN, title=title)


def show_bokeh(fig):
    """Display Bokeh figure in browser."""
    from bokeh.io import show

    show(fig)
