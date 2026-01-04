# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Command-line interface for brokrest trend analysis.

Usage:
    python -m brokrest plot data/xbtusd_ohlc_sample.csv
    python -m brokrest plot data/Kraken_Trading_History.zip --start 2020-01-01 --end 2021-01-01
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import fire


class BrokrestCLI:
    """Brokrest trend line analysis CLI."""

    def plot(
        self,
        data_path: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        segments: Optional[int] = None,
        output: Optional[str] = None,
        no_open: bool = False,
        backend: str = "bokeh",
    ):
        """
        Plot price trends.

        Args:
            data_path: CSV or ZIP file path
            start: Start date YYYY-MM-DD (ZIP only)
            end: End date YYYY-MM-DD (ZIP only)
            interval: Bar interval 1min/5min/15min/1h/4h/1d (ZIP only)
            segments: Number of segments (auto if not set)
            output: Output HTML path
            no_open: Don't open browser
            backend: Plotting backend (bokeh or mpl)

        Examples:
            python -m brokrest plot data/xbtusd_ohlc_sample.csv
            python -m brokrest plot data/Kraken_Trading_History.zip --start 2020-01-01
        """
        import pandas as pd

        path = Path(data_path)

        # Load data
        if path.suffix.lower() == ".csv":
            print(f"📂 Loading CSV: {path}")
            df = pd.read_csv(path)

            if "datetime" not in df.columns and "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

            if start or end:
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    if start:
                        df = df[df["datetime"] >= start]
                    if end:
                        df = df[df["datetime"] <= end]

        elif path.suffix.lower() == ".zip":
            print(f"📦 Loading ZIP: {path}")
            from .loaders import load_xbtusd

            df = load_xbtusd(path, interval=interval, start_date=start, end_date=end)  # type: ignore
            df = df.reset_index()
        else:
            print(f"❌ Unsupported format: {path.suffix}")
            return

        print(f"📊 Loaded {len(df)} bars")

        if len(df) == 0:
            print("❌ No data!")
            return

        # Detect trends
        from .shapes.regression import detect_trends

        prices = df["close"].values
        regression = detect_trends(
            prices,
            n_segments=segments,
            auto=segments is None,
            min_segment_size=max(10, len(prices) // 50),
        )

        print(regression.trend_summary())

        # Plot
        output_path = output or "btc_analysis.html"
        title = f"Price Analysis ({interval})"

        if backend == "mpl":
            from .shapes.plotting import plot_trends_mpl

            fig = plot_trends_mpl(df, n_segments=segments, title=title)
            fig.savefig(output_path.replace(".html", ".png"))
            print(f"\n✅ Saved: {output_path.replace('.html', '.png')}")
        else:
            from .shapes.plotting import plot_trends_bokeh, save_bokeh

            fig = plot_trends_bokeh(df, n_segments=segments, title=title)
            save_bokeh(fig, output_path, title=title)
            print(f"\n✅ Saved: {output_path}")

            if not no_open:
                self._open_file(output_path)

    def extract(
        self,
        zip_path: str,
        pair: str = "XBTUSD",
        limit: int = 200000,
        output: Optional[str] = None,
    ):
        """
        Extract sample data from ZIP to CSV.

        Args:
            zip_path: ZIP file path
            pair: Trading pair
            limit: Max trades to extract
            output: Output path
        """
        import pandas as pd

        from .loaders import KrakenZipLoader

        print(f"📦 Extracting {pair} from {zip_path}...")

        with KrakenZipLoader(Path(zip_path)) as loader:
            df = loader.load_trades(pair, limit=limit)

        print(f"📊 Loaded {len(df)} trades")

        # Save raw trades
        trades_path = output or f"data/{pair.lower()}_sample.csv"
        df.to_csv(trades_path, index=False)
        print(f"💾 Trades: {trades_path}")

        # Convert to OHLC
        df["bar_ts"] = (df["timestamp"] // 86400) * 86400
        ohlc = df.groupby("bar_ts").agg({"price": ["first", "max", "min", "last"], "volume": "sum"})
        ohlc.columns = ["open", "high", "low", "close", "volume"]
        ohlc = ohlc.reset_index().rename(columns={"bar_ts": "timestamp"})
        ohlc["datetime"] = pd.to_datetime(ohlc["timestamp"], unit="s", utc=True)

        ohlc_path = trades_path.replace("_sample.csv", "_ohlc_sample.csv")
        ohlc.to_csv(ohlc_path, index=False)
        print(f"📈 OHLC: {ohlc_path}")

    def list_pairs(self, zip_path: str):
        """List all trading pairs in ZIP."""
        from .loaders import KrakenZipLoader

        with KrakenZipLoader(Path(zip_path)) as loader:
            pairs = loader.list_pairs()

        print(f"📦 Found {len(pairs)} pairs")
        for p in sorted(pairs)[:30]:
            print(f"   {p}")
        if len(pairs) > 30:
            print(f"   ... and {len(pairs) - 30} more")

    def ruler(
        self,
        data_path: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        output: Optional[str] = None,
        no_open: bool = False,
        backend: str = "bokeh",
        no_rotate: bool = False,
        tolerance: bool = False,
        tolerance_factor: float = 0.1,
        no_clamp: bool = False,
        invalid_penalty: float = 0.0,
    ):
        """
        Plot support and resistance lines (rulers).

        Args:
            data_path: CSV or ZIP file path
            start: Start date YYYY-MM-DD (ZIP only)
            end: End date YYYY-MM-DD (ZIP only)
            interval: Bar interval (ZIP only)
            output: Output path
            no_open: Don't open browser
            backend: Plotting backend (bokeh or mpl)
            no_rotate: Only shift, keep lines parallel to regression
            tolerance: Move lines inward to capture more points in band
            tolerance_factor: Band size as fraction of std (default 0.1 = 10%)
            no_clamp: Use pure linear regression slope (ignore constraint bounds)
            invalid_penalty: Penalty for points that violate line constraint in tolerance mode.
                             0.0 = ignore violations (default), 0.5 = moderate, 1.0 = full.

        Examples:
            python -m brokrest ruler data/xbtusd_ohlc_sample.csv
            python -m brokrest ruler data/xbtusd_ohlc_sample.csv --tolerance
            python -m brokrest ruler data/xbtusd_ohlc_sample.csv --tolerance --invalid-penalty 0.5
        """
        import pandas as pd

        path = Path(data_path)

        # Load data
        if path.suffix.lower() == ".csv":
            print(f"📂 Loading CSV: {path}")
            df = pd.read_csv(path)

            if "datetime" not in df.columns and "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

            if start or end:
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    if start:
                        df = df[df["datetime"] >= start]
                    if end:
                        df = df[df["datetime"] <= end]

        elif path.suffix.lower() == ".zip":
            print(f"📦 Loading ZIP: {path}")
            from .loaders import load_xbtusd

            df = load_xbtusd(path, interval=interval, start_date=start, end_date=end)  # type: ignore
            df = df.reset_index()
        else:
            print(f"❌ Unsupported format: {path.suffix}")
            return

        print(f"📊 Loaded {len(df)} bars")

        if len(df) == 0:
            print("❌ No data!")
            return

        # Find rulers
        from .shapes.rulers import find_rulers, plot_rulers_bokeh, plot_rulers_mpl

        prices = df["close"].values
        topline, bottomline = find_rulers(
            prices, 
            rotate=not no_rotate, 
            tolerance=tolerance,
            tolerance_factor=tolerance_factor,
            clamp=not no_clamp,
            invalid_penalty=invalid_penalty,
        )

        mode_parts = []
        if no_rotate:
            mode_parts.append("parallel")
        else:
            mode_parts.append("optimized")
        if tolerance:
            mode_parts.append("tolerance")
        if no_clamp:
            mode_parts.append("no-clamp")
        mode = "+".join(mode_parts)
        print(f"\n📐 Mode: {mode}")
        print(f"📈 Resistance: y = {topline.slope:.4f}x + {topline.intercept:.2f}")
        print(f"📉 Support:    y = {bottomline.slope:.4f}x + {bottomline.intercept:.2f}")

        # Plot
        output_path = output or "rulers.html"
        title = f"Support & Resistance ({interval})"
        
        # Pass tolerance_factor for point coloring in tolerance mode
        tol_arg = tolerance_factor if tolerance else None

        if backend == "mpl":
            fig = plot_rulers_mpl(prices, topline, bottomline, title=title, 
                                  tolerance_factor=tol_arg)
            output_png = output_path.replace(".html", ".png")
            fig.savefig(output_png)
            print(f"\n✅ Saved: {output_png}")
        else:
            from .shapes.plotting import save_bokeh

            fig = plot_rulers_bokeh(prices, topline, bottomline, title=title,
                                    tolerance_factor=tol_arg)
            save_bokeh(fig, output_path, title=title)
            print(f"\n✅ Saved: {output_path}")

            if not no_open:
                self._open_file(output_path)

    def _open_file(self, path: str):
        """Open file in default application."""
        if sys.platform == "darwin":
            subprocess.run(["open", path])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", path])
        elif sys.platform == "win32":
            os.startfile(path)  # type: ignore


def main():
    """Entry point."""
    fire.Fire(BrokrestCLI)


if __name__ == "__main__":
    main()
