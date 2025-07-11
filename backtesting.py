import pandas as pd 
import backtrader as bt
import numpy as np
from sb3_contrib import RecurrentPPO
import csv 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/test_data.csv", parse_dates=["datetime"])
df = df.rename(columns={
    "datetime": "datetime",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})
df.set_index("datetime", inplace=True)
df = df[["open", "high", "low", "close", "volume"]]

class RLStrategy(bt.Strategy):
    def __init__(self, model):
        self.model = model
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.order = None
        self.trades_log = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # order is active, no action
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED, Price: {order.executed.price}, Size: {order.executed.size}")
            elif order.issell():
                print(f"SELL EXECUTED, Price: {order.executed.price}, Size: {order.executed.size}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Canceled/Margin/Rejected")
            self.order = None

    def next(self):
        if self.order:
            return  # waiting for pending order to complete

        obs = np.array([[self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.data.volume[0]]], dtype=np.float32)
        action, self.lstm_states = self.model.predict(obs, episode_start=self.episode_start, deterministic=False)
        self.episode_start = np.array([False]) 

        print(f"action: {action}, position size: {self.position.size}")

        if action == 1 and not self.position:
            print('BUy condition met')
            self.order = self.buy()
        elif action == 2 and self.position:
            print('Sell condition met')
            self.order = self.sell()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_log.append({
                'entry_datetime': trade.open_datetime(),
                'exit_datetime': self.data.datetime.datetime(0),
                'action': 'SELL SIDE' if trade.isopen else 'BUY SIDE',
                'entry_price': trade.price,
                'exit_price': self.data.close[0],
                'profit': trade.pnl,
                'bar_len': trade.barlen,
                'size': trade.size,
                'cash': self.broker.get_cash(),
                'portfolio_value': self.broker.getvalue()
            })

    def stop(self):
        # Save the trades to CSV at the end
        with open("results/backtest_results/executed_trades.csv", mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                'entry_datetime', 'exit_datetime', 'action', 'entry_price', 'exit_price', 
                'profit', 'bar_len', 'size', 'cash', 'portfolio_value'
            ])
            writer.writeheader()
            writer.writerows(self.trades_log)
        print("✅ Trades saved to 'executed_trades.csv'")

# Load your trained model
model = RecurrentPPO.load("model/recurrent_ppo_trading")

# Prepare backtrader data feed
data_feed = bt.feeds.PandasData(dataname=df)

# Set up backtrader engine
cerebro = bt.Cerebro()
cerebro.addstrategy(RLStrategy, model=model)
cerebro.adddata(data_feed)
cerebro.broker.set_cash(100000)  # starting capital

# Run backtest
cerebro.run()

# Plot result
cerebro.plot()


# Save the final portfolio value
final_value = cerebro.broker.getvalue()
print(f"Final Portfolio Value: {final_value:.2f}")

# Trade Analysis 
# Load trades
df = pd.read_csv("results/backtest_results/executed_trades.csv")

# Ensure correct datetime and sorting
df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
df = df.sort_values(by='entry_datetime').reset_index(drop=True)

# Basic stats
total_trades = len(df)
total_pnl = df['profit'].sum()
average_pnl = df['profit'].mean()
win_rate = (df['profit'] > 0).mean() * 100

# Streaks
df['is_win'] = df['profit'] > 0
streaks = (df['is_win'] != df['is_win'].shift()).cumsum()
win_streaks = df[df['is_win']].groupby(streaks)['is_win'].count()
lose_streaks = df[~df['is_win']].groupby(streaks)['is_win'].count()
longest_win_streak = win_streaks.max() if not win_streaks.empty else 0
longest_lose_streak = lose_streaks.max() if not lose_streaks.empty else 0

# Holding time
df['holding_time'] = (df['exit_datetime'] - df['entry_datetime']).dt.total_seconds() / 60  # in minutes
average_holding_time = df['holding_time'].mean()

# Display results
print("📊 Summary Statistics")
print(f"Total number of trades: {total_trades}")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Average PnL per trade: {average_pnl:.2f}")
print(f"Win rate: {win_rate:.2f}%")
print(f"Longest winning streak: {longest_win_streak}")
print(f"Longest losing streak: {longest_lose_streak}")
print(f"Average holding time: {average_holding_time:.2f} minutes")

# Save summary statistics to CSV
summary_stats = {
    'Total Trades': total_trades,
    'Total PnL': total_pnl,
    'Average PnL': average_pnl,
    'Win Rate (%)': win_rate,
    'Longest Win Streak': longest_win_streak,
    'Longest Lose Streak': longest_lose_streak,
    'Average Holding Time (min)': average_holding_time
}
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv("results/backtest_results/summary_statistics.csv", index=False)

df['cumulative_pnl'] = df['profit'].cumsum()

# # Set plotting style
# sns.set(style="whitegrid")
# plt.figure(figsize=(12, 6))

# 1. 📈 Equity Curve
plt.figure(figsize=(12, 4))
plt.plot(df['exit_datetime'], df['portfolio_value'], label="Equity Curve", color='blue')
plt.xlabel("Time")
plt.ylabel("portfolio_value")
plt.title("Equity Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/equity_curve.png")
plt.show()


# 2. 📊 Profit/Loss Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['profit'], bins=30, kde=True, color='purple')
plt.title("Profit/Loss Distribution")
plt.xlabel("PnL per Trade")
plt.tight_layout()
plt.savefig("figures/profit_loss_distribution.png")
plt.show()


# 3. 💰 Cumulative Profit
plt.figure(figsize=(10, 4))
plt.plot(df['exit_datetime'], df['cumulative_pnl'], color='green')
plt.title("Cumulative Profit Over Time")
plt.xlabel("Time")
plt.ylabel("Cumulative Profit")
plt.tight_layout()
plt.savefig("figures/cumulative_profit.png")
plt.show()


# 4. ⏳ Trade Duration vs Profit
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='holding_time', y='profit', hue='profit', palette='coolwarm')
plt.title("Trade Duration vs Profit")
plt.xlabel("Duration (minutes)")
plt.ylabel("PnL")
plt.tight_layout()
plt.savefig("figures/trade_duration_vs_profit.png")
plt.show()



# 5. 🕒 Heatmap of Trades by Time of Day
df['hour'] = df['entry_datetime'].dt.hour
df['weekday'] = df['entry_datetime'].dt.day_name()

# Pivot table
heatmap_data = df.pivot_table(index='weekday', columns='hour', values='profit', aggfunc='sum').fillna(0)

# Sort weekdays properly
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(weekday_order)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='RdYlGn', linewidths=0.5, annot=True, fmt=".0f")
plt.title("Heatmap of Trades by Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Weekday")
plt.tight_layout()
plt.savefig("figures/heatmap_trades_by_time_of_day.png")
plt.show()

