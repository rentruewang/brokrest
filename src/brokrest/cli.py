# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Command-line interface for brokrest trend analysis.

Usage:
    # 從樣本 CSV 畫圖（快速）
    python -m brokrest plot data/xbtusd_ohlc_sample.csv
    
    # 從 ZIP 載入指定時間段
    python -m brokrest plot data/Kraken_Trading_History.zip --start 2020-01-01 --end 2021-01-01
    
    # 指定分段數
    python -m brokrest plot data/xbtusd_ohlc_sample.csv --segments 8
    
    # 列出 ZIP 中的交易對
    python -m brokrest list-pairs data/Kraken_Trading_History.zip
    
    # 提取樣本數據
    python -m brokrest extract data/Kraken_Trading_History.zip --limit 200000
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import fire


class BrokrestCLI:
    """Brokrest 趨勢線分析 CLI"""

    def plot(
        self,
        data_path: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        segments: Optional[int] = None,
        output: Optional[str] = None,
        no_open: bool = False,
        contours: bool = False,
        pair: str = "XBTUSD",
        merge: bool = False,
        spike: bool = False,
        spike_prominence: float = 0.05,
    ):
        """
        繪製趨勢線分析圖表

        Args:
            data_path: CSV 或 ZIP 檔案路徑
            start: 開始日期 YYYY-MM-DD（僅 ZIP）
            end: 結束日期 YYYY-MM-DD（僅 ZIP）
            interval: K線週期 1min/5min/15min/1h/4h/1d（僅 ZIP）
            segments: 趨勢分段數（預設自動）
            output: 輸出 HTML 路徑
            no_open: 不自動開啟瀏覽器
            contours: 顯示凸包邊界
            pair: 交易對（僅 ZIP）
            merge: 合併同方向線段（只留真正峰谷）
            spike: 使用 spike 感知模式（優先在局部極值點切分）
            spike_prominence: Spike 顯著性門檻 0-1（預設 0.05 = 5% 價格範圍）

        Examples:
            # 從 CSV
            python -m brokrest plot data/xbtusd_ohlc_sample.csv
            
            # 從 ZIP 指定時間
            python -m brokrest plot data/Kraken_Trading_History.zip --start 2020-01-01 --end 2021-01-01
            
            # 指定分段數
            python -m brokrest plot data/xbtusd_ohlc_sample.csv --segments 10
            
            # 合併同方向線段
            python -m brokrest plot data/xbtusd_ohlc_sample.csv --merge
            
            # Spike 感知模式（優先捕捉局部極值）
            python -m brokrest plot data/xbtusd_ohlc_sample.csv --spike
            
            # 調整 spike 敏感度（越小越敏感）
            python -m brokrest plot data/xbtusd_ohlc_sample.csv --spike --spike-prominence 0.02
        """
        import pandas as pd
        from .shapes.plotting import TrendPlotter, plot_price_with_trends
        from .shapes.regression import detect_trends

        path = Path(data_path)
        
        # 判斷是 CSV 還是 ZIP
        if path.suffix.lower() == '.csv':
            print(f"📂 載入 CSV: {path}")
            df = pd.read_csv(path)
            
            # 確保有 datetime 欄位
            if 'datetime' not in df.columns and 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            
            # 時間篩選
            if start or end:
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    if start:
                        df = df[df['datetime'] >= start]
                    if end:
                        df = df[df['datetime'] <= end]
                        
        elif path.suffix.lower() == '.zip':
            print(f"📦 載入 ZIP: {path}")
            print(f"   交易對: {pair}, 週期: {interval}")
            if start:
                print(f"   開始: {start}")
            if end:
                print(f"   結束: {end}")
            
            from .loaders import load_xbtusd
            df = load_xbtusd(
                path,
                interval=interval,  # type: ignore
                start_date=start,
                end_date=end,
            )
            df = df.reset_index()
        else:
            print(f"❌ 不支援的檔案格式: {path.suffix}")
            return

        print(f"📊 載入 {len(df)} 筆資料")
        
        if len(df) == 0:
            print("❌ 沒有資料！請檢查時間範圍")
            return
            
        # 顯示時間範圍
        if 'datetime' in df.columns:
            print(f"📅 時間範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
        
        # 顯示價格範圍
        if 'close' in df.columns:
            print(f"💰 價格範圍: ${df['close'].min():,.2f} ~ ${df['close'].max():,.2f}")

        # 趨勢偵測
        print(f"\n🔍 分析趨勢...")
        if spike:
            print("   🎯 Spike 感知模式（優先捕捉局部極值）")
        prices = df['close'].values
        regression = detect_trends(
            prices,
            n_segments=segments,
            auto=segments is None,
            min_segment_size=max(10, len(prices) // 50),
            spike_mode=spike,
            spike_prominence=spike_prominence,
        )
        
        print(regression.trend_summary())

        # 繪圖
        print(f"\n🎨 生成圖表...")
        if merge:
            print("   📎 合併同方向線段")
        title = f"{pair} 趨勢分析"
        if start and end:
            title += f" ({start} ~ {end})"
        elif start:
            title += f" (從 {start})"
        elif end:
            title += f" (到 {end})"
        if spike:
            title += " [Spike]"
        if merge:
            title += " [Merged]"
            
        fig = plot_price_with_trends(
            df,
            n_segments=segments,
            auto_segments=segments is None,
            show_contours=contours,
            title=title,
            merge=merge,
        )

        # 儲存
        output_path = output or "btc_analysis.html"
        TrendPlotter().save(fig, output_path)
        print(f"\n✅ 已儲存: {output_path}")

        # 開啟瀏覽器
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
        從 ZIP 提取樣本數據為 CSV

        Args:
            zip_path: ZIP 檔案路徑
            pair: 交易對
            limit: 最大交易筆數
            output: 輸出路徑

        Example:
            python -m brokrest extract data/Kraken_Trading_History.zip --limit 500000
        """
        import pandas as pd
        from .loaders import KrakenZipLoader

        print(f"📦 從 {zip_path} 提取 {pair} 資料...")
        
        with KrakenZipLoader(Path(zip_path)) as loader:
            df = loader.load_trades(pair, limit=limit)
            
        print(f"📊 載入 {len(df)} 筆交易")

        # 儲存原始交易
        trades_path = output or f"data/{pair.lower()}_sample.csv"
        df.to_csv(trades_path, index=False)
        print(f"💾 交易資料: {trades_path}")

        # 轉換為 OHLC
        df['bar_ts'] = (df['timestamp'] // 86400) * 86400
        ohlc = df.groupby('bar_ts').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlc = ohlc.reset_index().rename(columns={'bar_ts': 'timestamp'})
        ohlc['datetime'] = pd.to_datetime(ohlc['timestamp'], unit='s', utc=True)

        ohlc_path = trades_path.replace('_sample.csv', '_ohlc_sample.csv')
        ohlc.to_csv(ohlc_path, index=False)
        print(f"📈 OHLC 資料: {ohlc_path}")
        print(f"📅 時間範圍: {ohlc['datetime'].min()} ~ {ohlc['datetime'].max()}")

    def list_pairs(self, zip_path: str):
        """
        列出 ZIP 中所有交易對

        Example:
            python -m brokrest list-pairs data/Kraken_Trading_History.zip
        """
        from .loaders import KrakenZipLoader

        with KrakenZipLoader(Path(zip_path)) as loader:
            pairs = loader.list_pairs()

        # 分類顯示
        btc_pairs = sorted([p for p in pairs if 'XBT' in p or 'BTC' in p])
        eth_pairs = sorted([p for p in pairs if 'ETH' in p])
        
        print(f"📦 找到 {len(pairs)} 個交易對\n")
        
        print(f"🟠 BTC 相關 ({len(btc_pairs)}):")
        for p in btc_pairs[:20]:
            print(f"   {p}")
        if len(btc_pairs) > 20:
            print(f"   ... 還有 {len(btc_pairs) - 20} 個")
            
        print(f"\n🔷 ETH 相關 ({len(eth_pairs)}):")
        for p in eth_pairs[:20]:
            print(f"   {p}")
        if len(eth_pairs) > 20:
            print(f"   ... 還有 {len(eth_pairs) - 20} 個")

    def info(self, zip_path: str, pair: str = "XBTUSD"):
        """
        顯示交易對資訊

        Example:
            python -m brokrest info data/Kraken_Trading_History.zip XBTUSD
        """
        from datetime import datetime, timezone
        from .loaders import KrakenZipLoader

        print(f"📦 讀取 {pair} 資訊...")
        
        with KrakenZipLoader(Path(zip_path)) as loader:
            first_chunk = None
            last_chunk = None
            total_trades = 0

            for chunk in loader.iter_trades(pair, chunk_size=100000):
                if first_chunk is None:
                    first_chunk = chunk
                last_chunk = chunk
                total_trades += len(chunk)

            if first_chunk is None:
                print(f"❌ 找不到 {pair} 的資料")
                return

            first_ts = first_chunk["timestamp"].iloc[0]
            last_ts = last_chunk["timestamp"].iloc[-1]
            first_date = datetime.fromtimestamp(first_ts, tz=timezone.utc)
            last_date = datetime.fromtimestamp(last_ts, tz=timezone.utc)

            print(f"\n📊 {pair}")
            print(f"   總交易筆數: {total_trades:,}")
            print(f"   時間範圍: {first_date.date()} ~ {last_date.date()}")
            print(f"   首筆價格: ${first_chunk['price'].iloc[0]:,.2f}")
            print(f"   末筆價格: ${last_chunk['price'].iloc[-1]:,.2f}")

    def _open_file(self, path: str):
        """開啟檔案"""
        if sys.platform == 'darwin':
            subprocess.run(['open', path])
        elif sys.platform == 'linux':
            subprocess.run(['xdg-open', path])
        elif sys.platform == 'win32':
            os.startfile(path)


def main():
    """Entry point"""
    fire.Fire(BrokrestCLI)


if __name__ == "__main__":
    main()
