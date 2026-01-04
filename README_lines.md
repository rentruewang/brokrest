# BrokRest Trend Analysis

Piecewise linear regression for cryptocurrency price trend detection.

## Quick Start

### Plot from CSV

```bash
python -m brokrest plot data/xbtusd_ohlc_sample.csv
```

### Plot from ZIP (Kraken data)

```bash
# Specify date range
python -m brokrest plot data/Kraken_Trading_History.zip --start 2020-01-01 --end 2021-01-01

# Different intervals: 1min, 5min, 15min, 1h, 4h, 1d
python -m brokrest plot data/Kraken_Trading_History.zip --interval 4h --start 2023-01-01
```

### CLI Options

```bash
python -m brokrest plot <file> [options]

Options:
  --start         Start date YYYY-MM-DD (ZIP only)
  --end           End date YYYY-MM-DD (ZIP only)
  --interval      Bar interval: 1min/5min/15min/1h/4h/1d (default: 1d)
  --segments      Number of trend segments (default: auto)
  --output        Output HTML path (default: btc_analysis.html)
  --no-open       Don't open browser after saving
  --backend       Plotting backend: bokeh or mpl (default: bokeh)
```

### Plot Support & Resistance Lines

```bash
python -m brokrest ruler data/xbtusd_ohlc_sample.csv
python -m brokrest ruler data/Kraken_Trading_History.zip --start 2020-01-01 --end 2021-01-01

# Parallel mode (no slope optimization, lines parallel to regression)
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --no-rotate
```

### Other Commands

```bash
# List trading pairs in ZIP
python -m brokrest list-pairs data/Kraken_Trading_History.zip

# Extract sample data to CSV
python -m brokrest extract data/Kraken_Trading_History.zip --limit 200000
```

## Python API

### Trend Detection

```python
from brokrest.shapes.regression import detect_trends
import numpy as np

prices = np.array([100, 102, 105, 103, 101, 99, 102, 106, 110])
result = detect_trends(prices, auto=True)

print(result.trend_summary())
# Segment 1: ↑ [0:2] slope=2.50 R²=1.000
# Segment 2: ↓ [3:5] slope=-2.00 R²=1.000
# ...
```

### Plotting with Bokeh

```python
import pandas as pd
from brokrest.shapes.plotting import plot_trends_bokeh, save_bokeh, show_bokeh

df = pd.read_csv("data/xbtusd_ohlc_sample.csv")
fig = plot_trends_bokeh(df, title="BTC Trend Analysis")

save_bokeh(fig, "output.html")
show_bokeh(fig)  # Opens in browser
```

### Plotting with Matplotlib

```python
from brokrest.shapes.plotting import plot_trends_mpl

df = pd.read_csv("data/xbtusd_ohlc_sample.csv")
fig = plot_trends_mpl(df, title="BTC Trend Analysis")
fig.savefig("output.png")
```

### Using the Painter Protocol

`TrendSegment` and `PiecewiseRegression` implement the `Painter` protocol:

```python
from brokrest.painters import Canvas
from brokrest.shapes.regression import detect_trends
import numpy as np

prices = np.random.randn(100).cumsum() + 100
result = detect_trends(prices)

# Draw on matplotlib canvas
canvas = Canvas(left=0, right=len(prices), interval=1)
result.plot(canvas)  # Draws all trend lines

# Access equation for each segment
for seg in result.segments:
    print(f"y = {seg.equation.m:.4f}x + {seg.equation.b:.4f}")
```

### Support & Resistance Lines (Rulers)

```python
from brokrest.shapes.rulers import find_rulers, plot_rulers_mpl
import numpy as np

prices = np.array([100, 105, 103, 108, 106, 112, 110, 115])

# With slope optimization (default)
topline, bottomline = find_rulers(prices, rotate=True)

# Parallel lines (no rotation)
topline, bottomline = find_rulers(prices, rotate=False)

print(f"Resistance: y = {topline.slope:.4f}x + {topline.intercept:.2f}")
print(f"Support: y = {bottomline.slope:.4f}x + {bottomline.intercept:.2f}")

# Plot
fig = plot_rulers_mpl(prices, topline, bottomline)
fig.savefig("rulers.png")
```

### Loading Kraken Data

```python
from brokrest.loaders import load_xbtusd, KrakenZipLoader

# Simple loader
df = load_xbtusd(
    "data/Kraken_Trading_History.zip",
    interval="1d",
    start_date="2020-01-01",
    end_date="2021-01-01",
)

# Advanced: iterate in chunks (memory efficient)
with KrakenZipLoader("data/Kraken_Trading_History.zip") as loader:
    for chunk in loader.iter_trades("XBTUSD", chunk_size=100000):
        print(f"Processing {len(chunk)} trades...")
```

## Module Structure

```
src/brokrest/
├── candles.py       # Candle charts (Painter)
├── equations.py     # Linear equations (PyTorch)
├── painters.py      # Canvas/Painter protocol
├── vectors.py       # 2D vector operations
├── cli.py           # Command-line interface
├── loaders/
│   └── kraken.py    # Kraken ZIP data loader
└── shapes/
    ├── contours.py  # Convex hull boundaries
    ├── plotting.py  # Trend visualization
    ├── regression.py # Piecewise linear regression
    └── rulers.py    # Support/resistance lines
```

## Algorithm

The trend detection uses **binary segmentation** with O(1) RSS computation:

1. Start with entire price series as one segment
2. Find the split point that maximizes RSS reduction
3. Repeat until reaching target segment count or R² improvement threshold
4. Fit linear regression to each segment

The algorithm uses prefix sums for efficient O(1) computation of segment statistics.

