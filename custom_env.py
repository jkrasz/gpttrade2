import gym
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df, leverage=2):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.leverage = leverage
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),))
        self.reset()

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.buy_price = 0
        return self._next_observation()

    def _next_observation(self):
        return self.df.iloc[self.current_step]

    def step(self, action):
        reward = 0
        done = False
        self.current_step += 1
        current_price = self.df.iloc[self.current_step]['Close']

        if action == 1:  # Buy
            if not self.holding:
                self.holding = True
                self.buy_price = current_price
                reward = 1  # Reward for opening a position
            else:
                reward = -1  # Penalty for unnecessary action
        elif action == 2 and self.holding:  # Sell
            profit = (current_price - self.buy_price) * self.leverage
            reward = profit
            self.holding = False
            self.buy_price = 0
        else:  # Hold
            if self.holding:
                reward = (current_price - self.buy_price) * self.leverage

        done = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}, Holding: {self.holding}, Buy Price: {self.buy_price}')
