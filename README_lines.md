# BTC Trend Line Analysis Tool

Analyze BTC price trends from Kraken historical trading data using Piecewise Linear Regression to automatically detect trend turning points.

## Features

- 📦 **Direct ZIP Reading**: No need to extract 10GB of historical data
- 📈 **Piecewise Linear Regression**: Automatic trend segmentation using Binary Segmentation algorithm (0.08s for 915 data points)
- 🎨 **Interactive Charts**: Zoomable, pannable HTML charts using Bokeh
- 🔴🟢 **Trend Line Colors**: Green for uptrend, red for downtrend
- 🟣🔵 **Peak/Valley Markers**:
  - 🟣 Deep Purple: True peak (↑→↓)
  - 🔵 Deep Blue: True valley (↓→↑)
  - 💜 Light Purple: Same-direction high point (acceleration)
  - 💙 Light Blue: Same-direction low point (deceleration)

## Installation

```bash
pip install numpy pandas scipy scikit-learn shapely bokeh fire
```

## CLI Quick Start

```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/brokrest/src

# Plot from CSV (fastest)
python -m brokrest plot data/xbtusd_ohlc_sample.csv

# Specify date range
python -m brokrest plot data/xbtusd_ohlc_sample.csv --start 2014-01-01 --end 2015-01-01

# Specify number of segments
python -m brokrest plot data/xbtusd_ohlc_sample.csv --segments 8

# Load from ZIP (slower but complete)
python -m brokrest plot data/Kraken_Trading_History.zip --start 2020-01-01 --end 2021-01-01

# Use hourly candlesticks
python -m brokrest plot data/Kraken_Trading_History.zip --start 2023-01-01 --end 2023-02-01 --interval 1h

# Merge same-direction segments (keep only true peaks/valleys)
python -m brokrest plot data/xbtusd_ohlc_sample.csv --merge

# Don't auto-open browser
python -m brokrest plot data/xbtusd_ohlc_sample.csv --no-open

# Show convex hull contours
python -m brokrest plot data/xbtusd_ohlc_sample.csv --contours

# Output to specific path
python -m brokrest plot data/xbtusd_ohlc_sample.csv -o my_chart.html
```

### Other CLI Commands

```bash
# Extract sample data (from large ZIP to small CSV)
python -m brokrest extract data/Kraken_Trading_History.zip --limit 500000

# List trading pairs in ZIP
python -m brokrest list-pairs data/Kraken_Trading_History.zip

# View trading pair info
python -m brokrest info data/Kraken_Trading_History.zip XBTUSD

# View command help
python -m brokrest plot --help
```

## Quick Start (Python API)

### 1. Prepare Data

Place Kraken historical trading data at `data/Kraken_Trading_History.zip`

### 2. Extract Sample Data (Optional, speeds up development)

```python
import sys
sys.path.insert(0, 'src')

from brokrest.loaders import KrakenZipLoader
from pathlib import Path
import pandas as pd

with KrakenZipLoader(Path('data/Kraken_Trading_History.zip')) as loader:
    # Extract 200k trades
    df = loader.load_trades('XBTUSD', limit=200000)
    df.to_csv('data/xbtusd_sample.csv', index=False)
    
    # Convert to daily OHLC
    df['bar_ts'] = (df['timestamp'] // 86400) * 86400
    ohlc = df.groupby('bar_ts').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlc = ohlc.reset_index().rename(columns={'bar_ts': 'timestamp'})
    ohlc['datetime'] = pd.to_datetime(ohlc['timestamp'], unit='s', utc=True)
    ohlc.to_csv('data/xbtusd_ohlc_sample.csv', index=False)
```

### 3. Generate Trend Analysis Chart

```python
import sys
sys.path.insert(0, 'src')

import pandas as pd
from brokrest.shapes.plotting import TrendPlotter, plot_price_with_trends

# Load data
ohlc = pd.read_csv('data/xbtusd_ohlc_sample.csv')

# Generate chart
fig = plot_price_with_trends(
    ohlc,
    auto_segments=True,      # Auto-determine segment count
    show_contours=False,     # Show convex hull contours
    title='BTC/USD Trend Analysis'
)

# Save as HTML
TrendPlotter().save(fig, 'btc_analysis.html')
```

### 4. Open Chart

```bash
open btc_analysis.html  # macOS
# or
xdg-open btc_analysis.html  # Linux
```

## Advanced Usage

### Load OHLC Directly from ZIP

```python
from brokrest.loaders import load_xbtusd

df = load_xbtusd(
    'data/Kraken_Trading_History.zip',
    interval='1d',           # 1min, 5min, 15min, 1h, 4h, 1d
    start_date='2020-01-01',
    end_date='2024-01-01',
)
```

### Custom Trend Detection

```python
from brokrest.shapes.regression import detect_trends

prices = ohlc['close'].values

# Auto-determine segment count
regression = detect_trends(prices, auto=True, min_segment_size=20)

# Or specify segment count
regression = detect_trends(prices, n_segments=5, auto=False)

# View results
print(regression.trend_summary())
```

Example output:
```
Segment 1: ↑ [0:44] slope=10.4143 R²=0.788
Segment 2: ↓ [45:121] slope=-1.3625 R²=0.075
Segment 3: ↓ [122:224] slope=-2.5780 R²=0.737
Segment 4: ↓ [225:564] slope=-1.3774 R²=0.875
Segment 5: ↑ [565:914] slope=0.7002 R²=0.757
Total R²: 0.941
```

### Plot Individual Elements

```python
from brokrest.shapes.plotting import TrendPlotter
from brokrest.shapes.regression import detect_trends

plotter = TrendPlotter(title='My Chart', width=1400, height=700)

# Create figure
fig = plotter.create_figure()

# Plot candlesticks
fig = plotter.plot_ohlc(ohlc, fig=fig)

# Plot trend lines with peak/valley markers
regression = detect_trends(ohlc['close'].values, auto=True)
fig = plotter.plot_trends(ohlc, regression, fig=fig)

# Plot convex hull contours (optional)
fig = plotter.plot_contours(ohlc, fig=fig)

# Save
plotter.save(fig, 'custom_chart.html')
```

## Module Structure

```
src/brokrest/
├── loaders/
│   ├── __init__.py
│   └── kraken.py          # Kraken ZIP data loader
├── shapes/
│   ├── __init__.py
│   ├── regression.py      # Piecewise linear regression
│   ├── plotting.py        # Bokeh plotting
│   ├── contours.py        # Convex hull contours
│   └── histories.py       # Price history data structures
└── cli.py                 # Command-line interface
```

## Chart Legend

| Element | Color | Meaning |
|---------|-------|---------|
| Candlestick (up) | Green | Close > Open |
| Candlestick (down) | Red | Close < Open |
| Trend line (up) | Green dashed | End point > Start point |
| Trend line (down) | Red dashed | End point < Start point |
| True Peak | 🟣 Deep Purple | Up → Down (sell signal) |
| True Valley | 🔵 Deep Blue | Down → Up (buy signal) |
| Same-dir High | 💜 Light Purple | Relative high in same direction |
| Same-dir Low | 💙 Light Blue | Relative low in same direction |

### --merge Mode

Using `--merge` merges same-direction segments, keeping only true peaks and valleys:

```
Original (5 segments, 4 junctions):
  ↑ [0:44]    34 → 492   💙 Light Blue
  ↑ [45:121]  492 → 775  🟣 Deep Purple (True Peak)
  ↓ [122:224] 775 → 409  💜 Light Purple
  ↓ [225:564] 409 → 159  🔵 Deep Blue (True Valley)
  ↑ [565:914] 159 → 447

After merge (3 segments, 2 junctions):
  ↑ [0:121]   34 → 775   🟣 Deep Purple (True Peak)
  ↓ [122:564] 775 → 159  🔵 Deep Blue (True Valley)
  ↑ [565:914] 159 → 447
```

## Performance

| Operation | Data Size | Time |
|-----------|-----------|------|
| Load CSV | 915 bars | < 0.01s |
| Trend Detection | 915 bars | 0.08s |
| Generate Chart | 915 bars | 0.4s |
| Save HTML | - | 0.1s |
