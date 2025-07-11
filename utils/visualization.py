import pandas as pd
import plotly.graph_objects as go

# Assuming train_data and test_data are already defined DataFrames
train_data = pd.read_csv("data/train_data.csv", parse_dates=["datetime"])
test_data = pd.read_csv("data/test_data.csv", parse_dates=["datetime"])
# Concatenate both sets
df = pd.concat([train_data, test_data])

# Ensure datetime is parsed and sorted
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# Define split datetime as the last date in train_data
split_date = train_data["datetime"].max()

# Create figure
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df["datetime"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="OHLC"
))

# Add vertical split line
fig.add_vline(
    x=split_date,
    line_width=2,
    line_dash="dash",
    line_color="red",
)

# Add annotation for the split
fig.add_annotation(
    x=split_date,
    y=df["High"].max(),
    text="Train/Test Split",
    showarrow=True,
    arrowhead=2,
    ax=0,
    ay=-40,
    bgcolor="white"
)

# Update layout
fig.update_layout(
    title="OHLCV with Train/Test Split",
    xaxis_title="Datetime",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
    template="plotly_white"
)

# Show figure
fig.show()

# Save as interactive HTML
fig.write_html("figures/ohlcv_train_test_split.html")
print("✅ Saved as interactive HTML: ohlcv_train_test_split.html")

