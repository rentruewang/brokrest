# Rulers: Support & Resistance Line Detection

This document explains the algorithm for finding optimal **topline (resistance)** and **bottomline (support)** lines for price data.

## Quick Start

```bash
# Basic usage
python -m brokrest ruler data/xbtusd_ohlc_sample.csv

# With matplotlib backend
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --backend mpl

# Tolerance mode (capture more points in band)
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --tolerance

# Parallel lines (no rotation)
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --no-rotate
```

---

## Algorithm Overview

The algorithm finds two bounding lines for price data:
- **Topline (Resistance)**: A line above all data points, as close as possible
- **Bottomline (Support)**: A line below all data points, as close as possible

### Step-by-Step Process

#### Step 1: Linear Regression Baseline

Start with a standard linear regression line through all data points:

```
y = m_base * x + b_base
```

This gives us the "natural" slope of the data.

#### Step 2: Shift to Contact

**For Topline:**
- Shift the line **up** until all points are below or touching the line
- The shift amount is `δ = max(y_i - pred_i)` for all points

**For Bottomline:**
- Shift the line **down** until all points are above or touching the line  
- The shift amount is `δ = min(y_i - pred_i)` for all points

After shifting, at least one point will be **touching** the line (contact point).

#### Step 3: Pivot and Optimize Slope

Using the contact point as a **pivot**, we rotate the line to minimize MSE while respecting constraints.

**The optimization problem:**
```
minimize:    MSE = mean((y_i - pred_i)²)
subject to:  All points on correct side of line
```

**Key insight:** The line passes through the pivot point, so:
```
y = m * (x - pivot_x) + pivot_y
```

We transform to pivot-centered coordinates:
```
dx_i = x_i - pivot_x
dy_i = y_i - pivot_y
```

**Constraint derivation (for topline):**
- Points to the **right** of pivot (`dx > 0`): slope must be steep enough → `m ≥ dy_i/dx_i`
- Points to the **left** of pivot (`dx < 0`): slope must not be too steep → `m ≤ dy_i/dx_i`

This gives us bounds: `m_lower ≤ m ≤ m_upper`

#### Step 4: Analytic Solution

The unconstrained optimal slope (minimizing MSE) has a **closed-form solution**:

```
m_opt = mean(dx * dy) / mean(dx²)
```

This is derived by setting `d(MSE)/dm = 0`:

```
MSE(m) = mean((dy - m*dx)²)
      = mean(dy² - 2m*dx*dy + m²*dx²)

d(MSE)/dm = -2*mean(dx*dy) + 2m*mean(dx²) = 0

∴ m = mean(dx*dy) / mean(dx²)
```

**Final step:** Clamp to feasible range (unless `--no-clamp`):
```
m_final = clip(m_opt, m_lower, m_upper)
```

---

## Command-Line Options

### `--no-rotate`

**Effect:** Keep lines parallel to the original regression line.

Only performs Step 1-2 (shift), skips Step 3-4 (rotation/optimization).

```bash
python -m brokrest ruler data.csv --no-rotate
```

| Mode | Topline Slope | Bottomline Slope |
|------|---------------|------------------|
| Default | Optimized independently | Optimized independently |
| `--no-rotate` | Same as regression | Same as regression |

---

### `--no-clamp`

**Effect:** Use the pure analytic solution without constraint clamping.

In Step 4, instead of:
```python
m_final = clip(m_opt, m_lower, m_upper)
```

We use:
```python
m_final = m_opt  # May violate constraints
```

This means the line may **pass through** some data points, which can be useful when constraints are too restrictive.

```bash
python -m brokrest ruler data.csv --no-clamp
```

---

### `--tolerance`

**Effect:** Move lines inward to capture more points within a "tolerance band".

After finding the strict bounding line, we try to shift it **inward** (down for topline, up for bottomline) to maximize the number of points within a band around the line.

**Band size:** `band = tolerance_factor × std(prices)`

**Optimization objective:**
```
score = in_band_count - invalid_penalty × invalid_count
```

Where:
- `in_band_count`: Points within distance `band` from the line
- `invalid_count`: Points on the wrong side of the line (violations)

```bash
# Default tolerance (10% of std)
python -m brokrest ruler data.csv --tolerance

# Larger band (20% of std)
python -m brokrest ruler data.csv --tolerance --tolerance-factor 0.2
```

---

### `--tolerance-factor`

**Effect:** Control the band size in tolerance mode.

| Value | Band Size | Effect |
|-------|-----------|--------|
| 0.05 | 5% of std | Tight band, fewer points captured |
| 0.1 (default) | 10% of std | Moderate |
| 0.2 | 20% of std | Wide band, more points captured |

---

### `--invalid-penalty`

**Effect:** Control how much to penalize constraint violations in tolerance mode.

The scoring function is:
```
score = in_band_count - invalid_penalty × invalid_count
```

| Value | Effect |
|-------|--------|
| 0.0 (default) | Ignore violations, purely maximize in-band count |
| 0.5 | Moderate penalty: 1 violation = 0.5 in-band points |
| 1.0 | Full penalty: 1 violation = 1 in-band point |

```bash
# Moderate penalty for violations
python -m brokrest ruler data.csv --tolerance --invalid-penalty 0.5

# Strict: violations are heavily penalized
python -m brokrest ruler data.csv --tolerance --invalid-penalty 1.0
```

---

## Visual Output

In tolerance mode, points are colored by their relationship to the lines:

| Color | Meaning |
|-------|---------|
| 🔴 Red | Near resistance line (within band) |
| 🔵 Blue | Near support line (within band) |
| 🟢 Green | Other points (not in any band) |

---

## Complexity Analysis

| Step | Complexity |
|------|------------|
| Linear regression | O(n) |
| Shift to contact | O(n) |
| Compute slope bounds | O(n) |
| Analytic solution | O(n) |
| **Total (default)** | **O(n)** |
| Tolerance optimization | O(n²) |
| **Total (with tolerance)** | **O(n²)** |

The tolerance mode is O(n²) because we try each data point as a candidate intercept and compute in-band counts for each.

---

## Mathematical Details

### Why Analytic Solution Works

The MSE is a **quadratic function** of slope `m`:

```
MSE(m) = am² + bm + c
```

Where:
- `a = mean(dx²)` > 0 (positive, so parabola opens upward)
- `b = -2 × mean(dx × dy)`
- `c = mean(dy²)`

A quadratic with positive leading coefficient has exactly one minimum at:
```
m* = -b/(2a) = mean(dx × dy) / mean(dx²)
```

No iterative search needed!

### Constraint Geometry

For the topline, we require `y_i ≤ m(x_i - pivot_x) + pivot_y` for all `i`.

Rearranging: `dy_i ≤ m × dx_i`

- If `dx_i > 0`: `m ≥ dy_i/dx_i` (lower bound)
- If `dx_i < 0`: `m ≤ dy_i/dx_i` (upper bound)

The tightest constraints determine `m_lower` and `m_upper`.

If `m_lower > m_upper`, the constraints are **infeasible** (the pivot is a convex point), and we fall back to convex hull methods.

---

## Examples

### Example 1: Basic Support & Resistance

```bash
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --backend mpl
```

Output:
```
📐 Mode: optimized
📈 Resistance: y = -0.8444x + 1209.34
📉 Support:    y = 0.1170x + 122.00
```

### Example 2: Tolerance Mode with Penalty

```bash
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --tolerance --invalid-penalty 0.5 --backend mpl
```

This finds lines that balance:
- Maximizing points in the tolerance band
- Minimizing constraint violations

### Example 3: Parallel Lines

```bash
python -m brokrest ruler data/xbtusd_ohlc_sample.csv --no-rotate --backend mpl
```

Both lines have the same slope as the regression line, forming a "channel".

---

## API Usage

```python
from brokrest.shapes.rulers import find_rulers, plot_rulers_mpl
import numpy as np

# Sample data
prices = np.array([100, 105, 103, 110, 108, 115, 112, 120])

# Find rulers
topline, bottomline = find_rulers(
    prices,
    rotate=True,           # Optimize slope
    tolerance=False,       # Strict bounding
    clamp=True,            # Respect constraints
)

print(f"Resistance: y = {topline.slope:.4f}x + {topline.intercept:.2f}")
print(f"Support:    y = {bottomline.slope:.4f}x + {bottomline.intercept:.2f}")

# Plot
fig = plot_rulers_mpl(prices, topline, bottomline)
fig.savefig("rulers.png")
```

---

## Summary Table

| Option | Effect | Use When |
|--------|--------|----------|
| (default) | Optimized slope, strict bounds | General use |
| `--no-rotate` | Parallel to regression | Want a "channel" |
| `--no-clamp` | Ignore constraint bounds | Constraints too restrictive |
| `--tolerance` | Capture more points in band | Want softer bounds |
| `--invalid-penalty` | Penalize violations | Balance coverage vs. strictness |

