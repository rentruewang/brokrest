# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Plotting utilities for price data and trend analysis.

Uses bokeh for interactive visualization.
"""

from __future__ import annotations

import dataclasses as dcls
from datetime import datetime as DateTime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .regression import PiecewiseRegression, TrendSegment

if TYPE_CHECKING:
    from bokeh.plotting import figure as Figure


__all__ = [
    "TrendPlotter",
    "plot_price_with_trends",
    "quick_plot",
]


@dcls.dataclass
class TrendPlotter:
    """
    Interactive plotter for price data with trend lines.

    Uses bokeh for visualization.
    """

    title: str = "Price Analysis"
    width: int = 1200
    height: int = 600

    def create_figure(self) -> "Figure":
        """Create bokeh figure with proper configuration."""
        from bokeh.plotting import figure

        fig = figure(
            title=self.title,
            x_axis_label="Time",
            y_axis_label="Price",
            x_axis_type="datetime",
            width=self.width,
            height=self.height,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            tooltips=[
                ("Time", "@x{%F %T}"),
                ("Price", "@y{0,0.00}"),
            ],
        )

        fig.xaxis.formatter.days = "%Y-%m-%d"
        fig.hover.formatters = {"@x": "datetime"}

        return fig

    def plot_ohlc(
        self,
        df: pd.DataFrame,
        fig: "Figure | None" = None,
    ) -> "Figure":
        """
        Plot OHLC candlestick chart.

        Args:
            df: DataFrame with columns: timestamp/datetime, open, high, low, close
            fig: Existing figure or None to create new

        Returns:
            Bokeh figure
        """
        from bokeh.plotting import figure

        if fig is None:
            fig = self.create_figure()

        # Prepare data
        if "datetime" in df.columns:
            dates = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            dates = df.index

        # Calculate bar width (80% of interval)
        if len(dates) > 1:
            interval_ms = (dates.iloc[1] - dates.iloc[0]).total_seconds() * 1000
            bar_width = interval_ms * 0.8
        else:
            bar_width = 3600 * 1000 * 0.8  # default 1h

        # Determine up/down
        up = df["close"] >= df["open"]
        down = ~up

        # Draw candlesticks
        # Wicks (high-low lines)
        fig.segment(
            x0=dates,
            y0=df["high"],
            x1=dates,
            y1=df["low"],
            color="gray",
            line_width=1,
        )

        # Up candles (green)
        fig.vbar(
            x=dates[up],
            top=df["close"][up],
            bottom=df["open"][up],
            width=bar_width,
            fill_color="#26a69a",
            line_color="#26a69a",
        )

        # Down candles (red)
        fig.vbar(
            x=dates[down],
            top=df["open"][down],
            bottom=df["close"][down],
            width=bar_width,
            fill_color="#ef5350",
            line_color="#ef5350",
        )

        return fig

    def plot_line(
        self,
        df: pd.DataFrame,
        column: str = "close",
        fig: "Figure | None" = None,
        color: str = "#2196f3",
        line_width: int = 2,
        legend_label: str | None = None,
    ) -> "Figure":
        """
        Plot price as line chart.

        Args:
            df: DataFrame with price data
            column: Column to plot
            fig: Existing figure or None
            color: Line color
            line_width: Line width
            legend_label: Optional legend label
        """
        if fig is None:
            fig = self.create_figure()

        if "datetime" in df.columns:
            x = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            x = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            x = df.index

        fig.line(
            x=x,
            y=df[column],
            color=color,
            line_width=line_width,
            legend_label=legend_label,
        )

        return fig

    def plot_trends(
        self,
        df: pd.DataFrame,
        regression: PiecewiseRegression,
        fig: "Figure | None" = None,
        up_color: str = "#4caf50",
        down_color: str = "#f44336",
        line_width: int = 3,
        valley_color: str = "#1565c0",       # Deep blue for true valleys (↓→↑)
        peak_color: str = "#7b1fa2",          # Deep purple for true peaks (↑→↓)
        valley_light_color: str = "#64b5f6",  # Light blue for same-direction valley-like
        peak_light_color: str = "#ce93d8",    # Light purple for same-direction peak-like
        point_size: int = 15,
        merge: bool = False,                  # Merge same-direction segments
    ) -> "Figure":
        """
        Plot connected trend lines with peak/valley markers.

        Marker colors:
        - Deep purple: True peak (↑→↓)
        - Deep blue: True valley (↓→↑)
        - Light purple: Same-direction peak-like (acceleration point)
        - Light blue: Same-direction valley-like (deceleration point)
        
        If merge=True, same-direction segments are merged into single lines,
        leaving only true peaks and valleys.
        """
        if fig is None:
            fig = self.create_figure()

        if "datetime" in df.columns:
            dates = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            dates = df.index

        from bokeh.models import Label

        segments = regression.segments
        if not segments:
            return fig

        # First pass: compute all junction y-values for connected lines
        junction_ys = []
        for i, seg in enumerate(segments):
            y_start, y_end = seg.predict_range()
            if i == 0:
                junction_ys.append(y_start)
            junction_ys.append(y_end)

        # Build segment info with actual directions
        segment_info = []
        for i, seg in enumerate(segments):
            seg_start_y = junction_ys[i]
            seg_end_y = junction_ys[i + 1]
            is_rising = seg_end_y > seg_start_y
            segment_info.append({
                'start_idx': seg.start_idx,
                'end_idx': seg.end_idx,
                'start_y': seg_start_y,
                'end_y': seg_end_y,
                'is_rising': is_rising,
                'r_squared': seg.r_squared,
            })

        # Merge same-direction segments if requested
        if merge and len(segment_info) > 1:
            merged = [segment_info[0].copy()]
            for i in range(1, len(segment_info)):
                curr = segment_info[i]
                prev = merged[-1]
                
                # Check if same direction (both rising or both falling)
                if prev['is_rising'] == curr['is_rising']:
                    # Merge: extend previous segment
                    prev['end_idx'] = curr['end_idx']
                    prev['end_y'] = curr['end_y']
                    # R² is approximate average
                    prev['r_squared'] = (prev['r_squared'] + curr['r_squared']) / 2
                else:
                    # Different direction: start new segment
                    merged.append(curr.copy())
            
            segment_info = merged
            
            # Rebuild junction_ys for merged segments
            junction_ys = [segment_info[0]['start_y']]
            for seg in segment_info:
                junction_ys.append(seg['end_y'])

        # Collect junction points by type
        true_peak_x, true_peak_y = [], []
        true_valley_x, true_valley_y = [], []
        light_peak_x, light_peak_y = [], []
        light_valley_x, light_valley_y = [], []

        # Draw segments and mark junctions
        for i, seg in enumerate(segment_info):
            start_date = dates.iloc[seg['start_idx']]
            end_date = dates.iloc[seg['end_idx']]

            seg_start_y = seg['start_y']
            seg_end_y = seg['end_y']
            is_rising = seg['is_rising']

            color = up_color if is_rising else down_color

            # Draw segment line
            fig.line(
                x=[start_date, end_date],
                y=[seg_start_y, seg_end_y],
                color=color,
                line_width=line_width,
                line_dash="dashed",
                alpha=0.8,
            )

            # Mark junction AFTER this segment
            if i < len(segment_info) - 1:
                junction_date = end_date
                junction_y = seg_end_y

                curr_rising = seg['is_rising']
                next_rising = segment_info[i + 1]['is_rising']

                # Determine peak or valley based on actual direction change
                if curr_rising and not next_rising:
                    # TRUE PEAK: ↑ → ↓
                    true_peak_x.append(junction_date)
                    true_peak_y.append(junction_y)
                elif not curr_rising and next_rising:
                    # TRUE VALLEY: ↓ → ↑
                    true_valley_x.append(junction_date)
                    true_valley_y.append(junction_y)
                elif not merge:
                    # Only show light markers if not merged
                    if curr_rising and next_rising:
                        # Both rising: check if this point is a local high or low
                        prev_mid = (junction_ys[i] + junction_y) / 2
                        next_mid = (junction_y + junction_ys[i + 2]) / 2
                        if junction_y > prev_mid and junction_y > next_mid:
                            light_peak_x.append(junction_date)
                            light_peak_y.append(junction_y)
                        else:
                            light_valley_x.append(junction_date)
                            light_valley_y.append(junction_y)
                    else:
                        # Both falling
                        prev_mid = (junction_ys[i] + junction_y) / 2
                        next_mid = (junction_y + junction_ys[i + 2]) / 2
                        if junction_y < prev_mid and junction_y < next_mid:
                            light_valley_x.append(junction_date)
                            light_valley_y.append(junction_y)
                        else:
                            light_peak_x.append(junction_date)
                            light_peak_y.append(junction_y)

            # Add R² annotation
            mid_idx = (seg['start_idx'] + seg['end_idx']) // 2
            mid_date = dates.iloc[mid_idx]
            mid_y = (seg_start_y + seg_end_y) / 2

            label = Label(
                x=mid_date,
                y=mid_y,
                text=f"R²={seg['r_squared']:.2f}",
                text_font_size="10px",
                text_color=color,
            )
            fig.add_layout(label)

        # Draw markers in order: light first (background), then true (foreground)

        # Light valley points (light blue)
        if light_valley_x:
            fig.scatter(
                x=light_valley_x,
                y=light_valley_y,
                size=point_size,
                color=valley_light_color,
                marker="circle",
                legend_label="Valley-like (減速)",
                alpha=0.85,
            )

        # Light peak points (light purple)
        if light_peak_x:
            fig.scatter(
                x=light_peak_x,
                y=light_peak_y,
                size=point_size,
                color=peak_light_color,
                marker="circle",
                legend_label="Peak-like (加速)",
                alpha=0.85,
            )

        # True valley points (deep blue)
        if true_valley_x:
            fig.scatter(
                x=true_valley_x,
                y=true_valley_y,
                size=point_size,
                color=valley_color,
                marker="circle",
                legend_label="Valley (谷底)",
                alpha=0.95,
            )

        # True peak points (deep purple)
        if true_peak_x:
            fig.scatter(
                x=true_peak_x,
                y=true_peak_y,
                size=point_size,
                color=peak_color,
                marker="circle",
                legend_label="Peak (山頂)",
                alpha=0.95,
            )

        return fig

    def plot_contours(
        self,
        df: pd.DataFrame,
        fig: "Figure | None" = None,
        upper_color: str = "#ff9800",
        lower_color: str = "#03a9f4",
        line_width: int = 2,
    ) -> "Figure":
        """
        Plot upper and lower contour bounds (convex hull edges).

        Uses shapely for geometric computation.
        """
        from shapely import MultiPoint

        if fig is None:
            fig = self.create_figure()

        if "datetime" in df.columns:
            dates = pd.to_datetime(df["datetime"])
        elif "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            dates = df.index

        # Create points
        timestamps = dates.astype("int64") // 10**6  # to milliseconds
        prices = df["close"].values

        points = MultiPoint(list(zip(timestamps, prices)))
        hull = points.convex_hull

        if hull.is_empty or hull.geom_type == "Point":
            return fig

        hull_coords = np.array(hull.exterior.coords)

        # Separate upper and lower hull
        # Upper hull: points where we go right (x increases)
        # Lower hull: points where we go left
        x_coords = hull_coords[:, 0]
        y_coords = hull_coords[:, 1]

        # Find leftmost and rightmost points
        left_idx = np.argmin(x_coords)
        right_idx = np.argmax(x_coords)

        # Upper hull (from left to right, going through top)
        if left_idx < right_idx:
            upper_indices = list(range(left_idx, right_idx + 1))
        else:
            upper_indices = list(range(left_idx, len(x_coords))) + list(range(0, right_idx + 1))

        # Lower hull (from right to left, going through bottom)
        if right_idx < left_idx:
            lower_indices = list(range(right_idx, left_idx + 1))
        else:
            lower_indices = list(range(right_idx, len(x_coords))) + list(range(0, left_idx + 1))

        # Convert back to dates
        upper_x = pd.to_datetime(x_coords[upper_indices], unit="ms")
        upper_y = y_coords[upper_indices]

        lower_x = pd.to_datetime(x_coords[lower_indices], unit="ms")
        lower_y = y_coords[lower_indices]

        fig.line(
            x=upper_x,
            y=upper_y,
            color=upper_color,
            line_width=line_width,
            legend_label="Upper Bound",
        )

        fig.line(
            x=lower_x,
            y=lower_y,
            color=lower_color,
            line_width=line_width,
            legend_label="Lower Bound",
        )

        return fig

    def show(self, fig: "Figure"):
        """Display figure in browser."""
        from bokeh.io import show

        show(fig)

    def save(self, fig: "Figure", path: str | Path):
        """Save figure to HTML file."""
        from bokeh.io import save
        from bokeh.resources import CDN

        save(fig, filename=str(path), resources=CDN, title=self.title)


def plot_price_with_trends(
    df: pd.DataFrame,
    n_segments: int | None = None,
    auto_segments: bool = True,
    show_contours: bool = True,
    title: str = "BTC/USD Price Analysis",
    merge: bool = False,
) -> "Figure":
    """
    Create complete price analysis plot with trends and contours.

    Args:
        df: OHLC DataFrame
        n_segments: Number of trend segments (ignored if auto_segments=True)
        auto_segments: Auto-determine optimal segments
        show_contours: Show convex hull contours
        title: Plot title
        merge: Merge same-direction segments (only show true peaks/valleys)

    Returns:
        Bokeh figure
    """
    from .regression import detect_trends

    plotter = TrendPlotter(title=title)

    # Create figure with OHLC
    fig = plotter.plot_ohlc(df)

    # Detect and plot trends
    prices = df["close"].values
    regression = detect_trends(
        prices,
        n_segments=n_segments,
        auto=auto_segments,
    )

    fig = plotter.plot_trends(df, regression, fig=fig, merge=merge)

    # Add contours if requested
    if show_contours:
        fig = plotter.plot_contours(df, fig=fig)

    # Configure legend
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    return fig


def quick_plot(
    zip_path: str | Path,
    pair: str = "XBTUSD",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | Path | None = None,
) -> "Figure":
    """
    Quick analysis and plot from Kraken ZIP file.

    Args:
        zip_path: Path to Kraken_Trading_History.zip
        pair: Trading pair
        interval: OHLC interval
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        output_path: Optional output HTML path

    Returns:
        Bokeh figure

    Example:
        >>> fig = quick_plot(
        ...     "data/Kraken_Trading_History.zip",
        ...     start_date="2020-01-01",
        ...     end_date="2024-01-01",
        ... )
    """
    from ..loaders import load_xbtusd

    # Load data
    df = load_xbtusd(
        zip_path,
        interval=interval,  # type: ignore
        start_date=start_date,
        end_date=end_date,
    )

    print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    # Create plot
    fig = plot_price_with_trends(
        df.reset_index(),
        title=f"{pair} Price Analysis ({interval})",
    )

    # Save if output path provided
    if output_path:
        TrendPlotter().save(fig, output_path)
        print(f"Saved to {output_path}")

    return fig

