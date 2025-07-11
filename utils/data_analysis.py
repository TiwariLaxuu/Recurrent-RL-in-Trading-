import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/BTCUSD_15min_Data.csv')

print("Data Overview:")
print(df.head())
print(df.tail())
print("\nData Description:")
print(df.describe())
print("\nData Columns:")
print(df.columns) 

''' 
Data Columns:
Index(['Unnamed: 0', 'datetime', 'Open', 'High', 'Low', 'Close', 'Volume'], dtype='object')
Start Date : 2024-08-01 00:00:00
End Date: 2024-09-29 19:45:00 
'''

df.sort_values("datetime", inplace=True)
df = df.drop(columns=['Unnamed: 0'])
# Define your split date
split_date = "2024-09-20 00:00:00"

# Split into train/test sets
train_df = df[df["datetime"] < split_date].reset_index(drop=True)
test_df = df[df["datetime"] >= split_date].reset_index(drop=True)

print("Train data:", train_df.shape, train_df.columns) # 4800, 6 
print("Test data:", test_df.shape, test_df.columns) # 944, 6

# Save the train and test data to CSV files
train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)


# Plotting the closing prices
plt.figure(figsize=(14, 7))
plt.plot(df['datetime'], df['Close'], label='Close Price', color='blue')
plt.axvline(pd.to_datetime(split_date), color='red', linestyle='--', label='Split Date')
plt.title('BTCUSD Closing Prices')
plt.xlabel('Date')  
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45) 
plt.tight_layout()
plt.savefig("figures/btcusd_closing_prices.png")
plt.show()