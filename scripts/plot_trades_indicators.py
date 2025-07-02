import pandas as pd
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trades_indicators(signals_path, trades_path, prices_path):
    # 2. Load your OHLCV prices CSV
    #    — must have columns: date, open, high, low, close, volume (or drop volume)
    prices = pd.read_feather(prices_path)
    prices.set_index("date", inplace=True)

    # 3. Load signals.pkl
    signals = joblib.load(signals_path)
    strategy = next(iter(signals))
    pair     = next(iter(signals[strategy]))
    sig_df   = signals[strategy][pair].copy()
    sig_df["date"] = pd.to_datetime(sig_df["date"])
    sig_df.set_index("date", inplace=True)

    # 3a. Determine the signal date window
    start_date = sig_df.index.min()
    end_date   = sig_df.index.max()

    # 4. Merge only the condition flags onto your price index
    condition_cols = [c for c in sig_df.columns if c.startswith("cond_")]
    df = prices.join(sig_df[condition_cols], how="left").fillna(0)
    print(f"Condition columns: {condition_cols}")
    # 4a. Trim to the signal window
    df = df.loc[start_date : end_date]

    # 5. Load trades.json
    with open(trades_path) as f:
        data = json.load(f)
    trades = pd.DataFrame(data["strategy"][strategy]["trades"])
    trades["open_date"]  = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades = trades[trades["pair"] == pair]

    # 5a. Snap to bar frequency (e.g. 5T)
    freq = prices.index.freq or "5T"
    trades["open_date"]  = trades["open_date"].dt.floor(freq)
    trades["close_date"] = trades["close_date"].dt.floor(freq)

    # 5b. Filter trades to the same window
    trades = trades[(trades["open_date"]  >= start_date) &
                    (trades["open_date"]  <= end_date)   |
                (trades["close_date"] >= start_date) &
                    (trades["close_date"] <= end_date)]

    # 6. Build entry/exit tables that line up with df.index
    entries = trades[trades["open_date"].isin(df.index)]
    exits   = trades[trades["close_date"].isin(df.index)]
    wins    = exits[exits["profit_ratio"] > 0]
    losses  = exits[exits["profit_ratio"] <= 0]

    # 7. Build a 2-row subplot: candles on top, flags below
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.7, 0.3],
        subplot_titles=(f"{pair} Price + Trades", "Condition Flags")
    )

    # — Price candles —
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ), row=1, col=1)

    # — Entry / exit markers —
    fig.add_trace(go.Scatter(
        x=entries["open_date"],
        y=df.loc[entries["open_date"], "open"],
        mode="markers", name="Entry",
        marker=dict(symbol="circle", size=10, color="blue", opacity=0.6)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=wins["close_date"],
        y=df.loc[wins["close_date"], "close"],
        mode="markers", name="Exit (Win)",
        marker=dict(symbol="triangle-up", size=15, color="green")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=losses["close_date"],
        y=df.loc[losses["close_date"], "close"],
        mode="markers+text",
        name="Exit (Loss)",
        marker=dict(symbol="triangle-down", size=15, color="red"),
        text=losses["exit_reason"],            # show the exit reason as text
        textposition="top center",             # position label above the marker
        hovertemplate="Exit: %{text}<br>Price: %{y}<extra></extra>"
    ), row=1, col=1)

    # — Condition flags as bar chart —
    colors = ["orange","blue","green","purple","red","gray"]
    for i, col in enumerate(condition_cols):
        fig.add_trace(go.Bar(
            x=df.index, y=df[col],
            name=col,
            marker_color=colors[i % len(colors)],
            opacity=0.6,
        ), row=2, col=1)

    # 8. Final layout tweaks
    fig.update_layout(
        title=f"{pair} Strategy Trades & Conditions ({start_date.date()} to {end_date.date()})",
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend_orientation="h",
        margin=dict(t=40, b=40)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Flag (On=1)", row=2, col=1, range=[-0.1, 1.2])

    fig.write_html(
        trades_path.name.replace('json', 'html'),
        include_plotlyjs="cdn",  # or "embed" to inline the JS
        full_html=True,          # wraps in a complete HTML document
        auto_open=False          # set to True to open in your default browser
    )
    fig.show()
    

if __name__ == "__main__":
    # 1. — UPDATE THESE PATHS —
    signals_path = Path("C:/Users/youss/freqtrade-bot/freqtrade/user_data/backtest_results/extracted_results/ichiV1Conditions_20250630_214442-2025-06-30_21-44-50_exited.pkl")
    trades_path  = Path("C:/Users/youss/freqtrade-bot/freqtrade/user_data/backtest_results/extracted_results/ichiV1Conditions_20250630_214442-2025-06-30_21-44-50.json")
    prices_path  = Path("C://Users//youss//freqtrade-bot//freqtrade//user_data//data//okx//BTC_USDT-15m.feather")
    plot_trades_indicators(signals_path, trades_path, prices_path)

