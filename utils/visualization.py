import pandas as pd
import plotly.graph_objects as go

# Load your full dataset
df = pd.read_csv("data/BTCUSD_15min_Data.csv", parse_dates=["datetime"])

# Sort by datetime
df = df.sort_values("datetime")

# Define split datetime (based on your example)
split_date = "2024-09-20 00:00:00"

# Create figure
fig = go.Figure()

# Candlestick OHLCV
fig.add_trace(go.Candlestick(
    x=df["datetime"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="OHLC"
))

# # Add vertical split line
fig.add_vline(
    x=split_date,
    line_width=2,
    line_dash="dash",
    line_color="red",
)
# ✅ Add separate annotation
fig.add_annotation(
    x=split_date,
    y=max(df["High"]),  # adjust based on chart scale
    text="Train/Test Split",
    showarrow=True,
    arrowhead=2,
    ax=0,
    ay=-40,
    bgcolor="white"
)

# Layout
fig.update_layout(
    title="OHLCV with Train/Test Split",
    xaxis_title="Datetime",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
    template="plotly_white"
)

fig.show()

# ✅ Save as interactive HTML
fig.write_html("figures/ohlcv_train_test_split.html")
print("✅ Saved as interactive HTML: ohlcv_train_test_split.html")


