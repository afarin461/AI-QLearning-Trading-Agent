import pandas as pd
import numpy as np

class TradingEnvironment:

    ACTION_SHORT = 0 
    ACTION_HOLD = 1  
    ACTION_LONG = 2  

    def __init__(self, data_df, trading_cost=0.0005, episode_max_steps=None):

        if 'Returns' not in data_df.columns or 'State' not in data_df.columns:
            raise ValueError("data_df must contain 'Returns' and 'State' columns.")
            
        self.data_df = data_df
        self.trading_cost = trading_cost
        
        self.actions = {
            self.ACTION_SHORT: -1, 
            self.ACTION_HOLD: 0,   
            self.ACTION_LONG: 1    
        }
        self.action_labels = {
            self.ACTION_SHORT: "Short",
            self.ACTION_HOLD: "Hold",
            self.ACTION_LONG: "Long"
        }

        self.current_step = 0
        self.episode_max_steps = episode_max_steps if episode_max_steps is not None else len(self.data_df) - 1
        
        self.initial_nav = 1.0 
        self.nav = self.initial_nav
        self.prev_action = self.ACTION_HOLD 
        
        self.state_space = tuple(data_df['State'].unique())
        
        print(f"Environment initialized with {len(self.data_df)} days of data.")
        print(f"Unique states in environment: {len(self.state_space)}")

    def reset(self):

        self.current_step = 0
        self.nav = self.initial_nav
        self.prev_action = self.ACTION_HOLD 
        
        if len(self.data_df) == 0:
            raise ValueError("Provided data_df is empty after processing. Cannot reset environment.")
        
        if self.current_step >= len(self.data_df):
            self.current_step = 0 
            
        return self.data_df['State'].iloc[self.current_step]

    def step(self, action):

        if self.current_step >= self.episode_max_steps:
            return None, 0, True, {"NAV": self.nav, "DailyReturn": 0, "Action": self.action_labels[action]}

        current_day_data = self.data_df.iloc[self.current_step]
        
        daily_return = current_day_data['Returns'] 
        position = self.actions[action]
        
        reward = position * daily_return
        
        cost = 0.0
        if action != self.prev_action:
            if action == self.ACTION_SHORT or action == self.ACTION_LONG:
                cost = self.trading_cost
            elif self.prev_action == self.ACTION_SHORT or self.prev_action == self.ACTION_LONG:

                cost = self.trading_cost 
        
        reward -= cost

        if self.prev_action == self.ACTION_LONG:
            self.nav *= (1 + daily_return)
        elif self.prev_action == self.ACTION_SHORT:
            self.nav *= (1 - daily_return) 

        self.prev_action = action
        
        self.current_step += 1
        
        done = self.current_step >= self.episode_max_steps # Check if max steps reached
        if self.nav <= 0:
            done = True
            print(f"NAV hit zero at step {self.current_step}. Episode done.")

        next_state = None
        if not done:
            next_state = self.data_df['State'].iloc[self.current_step]
        
        info = {
            "NAV": self.nav, 
            "DailyReturn": daily_return, 
            "CostIncurred": cost,
            "ActionTaken": self.action_labels[action]
        }
        
        return next_state, reward, done, info