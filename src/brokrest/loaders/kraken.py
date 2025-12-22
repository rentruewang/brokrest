# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Kraken historical trade data loader.

Loads data directly from ZIP files without extraction.
"""

from __future__ import annotations

import dataclasses as dcls
import zipfile
from collections.abc import Iterator
from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta
from datetime import timezone as TimeZone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "KrakenZipLoader",
    "TradeRecord",
    "OHLCBar",
    "load_xbtusd",
]


@dcls.dataclass(frozen=True)
class TradeRecord:
    """Single trade record from Kraken."""

    timestamp: int  # Unix timestamp
    price: float
    volume: float

    @property
    def datetime(self) -> DateTime:
        return DateTime.fromtimestamp(self.timestamp, tz=TimeZone.utc)


@dcls.dataclass(frozen=True)
class OHLCBar:
    """OHLC bar aggregated from trades."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime(self) -> DateTime:
        return DateTime.fromtimestamp(self.timestamp, tz=TimeZone.utc)


@dcls.dataclass
class KrakenZipLoader:
    """
    Load Kraken historical trade data directly from ZIP without extraction.

    Data format: timestamp, price, volume (no header)
    """

    zip_path: Path
    _zipfile: zipfile.ZipFile | None = dcls.field(default=None, repr=False)

    def __post_init__(self):
        if isinstance(self.zip_path, str):
            self.zip_path = Path(self.zip_path)

    def __enter__(self):
        self._zipfile = zipfile.ZipFile(self.zip_path, "r")
        return self

    def __exit__(self, *args):
        if self._zipfile:
            self._zipfile.close()
            self._zipfile = None

    def list_pairs(self) -> list[str]:
        """List all available trading pairs in the ZIP."""
        if self._zipfile is None:
            raise RuntimeError("Use context manager: with KrakenZipLoader(...) as loader:")
        return [n.replace(".csv", "") for n in self._zipfile.namelist() if n.endswith(".csv")]

    def iter_trades(
        self,
        pair: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over trades in chunks (memory efficient for large files).

        Args:
            pair: Trading pair name (e.g., 'XBTUSD')
            start_ts: Optional start timestamp filter
            end_ts: Optional end timestamp filter
            chunk_size: Number of rows per chunk
        """
        if self._zipfile is None:
            raise RuntimeError("Use context manager: with KrakenZipLoader(...) as loader:")

        filename = f"{pair}.csv"
        with self._zipfile.open(filename) as f:
            # Read in chunks
            for chunk in pd.read_csv(
                f,
                names=["timestamp", "price", "volume"],
                chunksize=chunk_size,
                dtype={"timestamp": "int64", "price": "float64", "volume": "float64"},
            ):
                # Apply timestamp filters
                if start_ts is not None:
                    chunk = chunk[chunk["timestamp"] >= start_ts]
                if end_ts is not None:
                    chunk = chunk[chunk["timestamp"] <= end_ts]

                if len(chunk) > 0:
                    yield chunk

    def load_trades(
        self,
        pair: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Load all trades for a pair into a DataFrame.

        Args:
            pair: Trading pair name (e.g., 'XBTUSD')
            start_ts: Optional start timestamp filter
            end_ts: Optional end timestamp filter
            limit: Optional limit on number of rows
        """
        chunks = []
        total_rows = 0

        for chunk in self.iter_trades(pair, start_ts=start_ts, end_ts=end_ts):
            if limit is not None:
                remaining = limit - total_rows
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            chunks.append(chunk)
            total_rows += len(chunk)

            if limit is not None and total_rows >= limit:
                break

        if not chunks:
            return pd.DataFrame(columns=["timestamp", "price", "volume"])

        return pd.concat(chunks, ignore_index=True)

    def load_ohlc(
        self,
        pair: str,
        *,
        interval: Literal["1min", "5min", "15min", "1h", "4h", "1d"] = "1h",
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        """
        Load trades and aggregate to OHLC bars.

        Args:
            pair: Trading pair name
            interval: Bar interval
            start_ts: Optional start timestamp
            end_ts: Optional end timestamp

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        interval_seconds = {
            "1min": 60,
            "5min": 300,
            "15min": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }[interval]

        ohlc_data: dict[int, dict] = {}

        for chunk in self.iter_trades(pair, start_ts=start_ts, end_ts=end_ts):
            # Calculate bar timestamps
            chunk["bar_ts"] = (chunk["timestamp"] // interval_seconds) * interval_seconds

            for bar_ts, group in chunk.groupby("bar_ts"):
                bar_ts = int(bar_ts)
                if bar_ts not in ohlc_data:
                    ohlc_data[bar_ts] = {
                        "timestamp": bar_ts,
                        "open": group["price"].iloc[0],
                        "high": group["price"].max(),
                        "low": group["price"].min(),
                        "close": group["price"].iloc[-1],
                        "volume": group["volume"].sum(),
                    }
                else:
                    existing = ohlc_data[bar_ts]
                    existing["high"] = max(existing["high"], group["price"].max())
                    existing["low"] = min(existing["low"], group["price"].min())
                    existing["close"] = group["price"].iloc[-1]
                    existing["volume"] += group["volume"].sum()

        if not ohlc_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(list(ohlc_data.values()))
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


def load_xbtusd(
    zip_path: str | Path,
    interval: Literal["1min", "5min", "15min", "1h", "4h", "1d"] = "1h",
    start_date: str | DateTime | None = None,
    end_date: str | DateTime | None = None,
) -> pd.DataFrame:
    """
    Convenience function to load BTC/USD OHLC data.

    Args:
        zip_path: Path to Kraken_Trading_History.zip
        interval: OHLC bar interval
        start_date: Optional start date (string 'YYYY-MM-DD' or DateTime)
        end_date: Optional end date

    Returns:
        OHLC DataFrame with datetime index
    """
    # Convert dates to timestamps
    start_ts = None
    end_ts = None

    if start_date:
        if isinstance(start_date, str):
            start_date = DateTime.fromisoformat(start_date)
        start_ts = int(start_date.timestamp())

    if end_date:
        if isinstance(end_date, str):
            end_date = DateTime.fromisoformat(end_date)
        end_ts = int(end_date.timestamp())

    with KrakenZipLoader(Path(zip_path)) as loader:
        df = loader.load_ohlc("XBTUSD", interval=interval, start_ts=start_ts, end_ts=end_ts)

    # Add datetime column
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("datetime")

    return df

