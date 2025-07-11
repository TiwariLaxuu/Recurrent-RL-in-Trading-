import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.max_steps = len(df) - 1
        self.current_step = 0
        self.position = 0  
        self.initial_balance = 10000
        self.balance = self.initial_balance

        # Observation: past 10 candles (OHLCV) = (10, 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5), dtype=np.float32)

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.current_step = 10
        self.position = 0
        self.balance = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        print(obs.shape)
        return obs 
        # return self.df.iloc[self.current_step-10:self.current_step].values.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0
        price = self.df.iloc[self.current_step]['Close']

        # reward based on position and price movement
        if action == 1:  # buy
            if self.position == 0:
                self.entry_price = price
                self.position = 1
        elif action == 2:  # sell
            if self.position == 1:
                reward = price - self.entry_price
                self.balance += reward
                self.position = 0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}")


# Load and normalize data
df = pd.read_csv("data/train_data.csv")[['Open', 'High', 'Low', 'Close', 'Volume']]
df = (df - df.mean()) / df.std()

# Wrap your environment
env = DummyVecEnv([lambda: TradingEnv(df)])

# Use RecurrentPPO
model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=env,
    verbose=1,
    tensorboard_log="tensorboard_logs/recurrentppo_trading/",
    n_steps=128,
    batch_size=64, 
    learning_rate=3e-4,
)

# Train the model
model.learn(total_timesteps=1000)
model.save("model/recurrent_ppo_trading")