
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def buy_and_hold_benchmark(prices_series, initial_investment=1.0):

    if prices_series.empty:
        print("Price series is empty for Buy-and-Hold benchmark.")
        return pd.Series([initial_investment], index=[pd.Timestamp.now()])

    returns = prices_series.pct_change().fillna(0) # Fill first NaN with 0
    
    nav_history = initial_investment * (1 + returns).cumprod()
    
    if not nav_history.empty and nav_history.iloc[0] == 0:
        nav_history.iloc[0] = initial_investment
    
    return nav_history

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.0):
 
    if returns_series.empty:
        return 0.0
    
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1 
    
    excess_returns = returns_series - daily_risk_free_rate
    
 
    annualized_excess_return = excess_returns.mean() * 252 # Assuming 252 trading days in a year
    annualized_std_dev = excess_returns.std() * np.sqrt(252)
    
    if annualized_std_dev == 0:
        return 0.0 #
    
    sharpe_ratio = annualized_excess_return / annualized_std_dev
    return sharpe_ratio

def calculate_max_drawdown(nav_series):

    if nav_series.empty:
        return 0.0
    
    peak_nav = nav_series.expanding(min_periods=1).max()
    drawdown = (nav_series / peak_nav) - 1
    max_drawdown = drawdown.min()
    return abs(max_drawdown) 

def plot_performance(agent_nav_history, benchmark_nav_history, title="Agent vs. Buy-and-Hold Performance"):

    plt.figure(figsize=(12, 6))
    plt.plot(agent_nav_history.index, agent_nav_history, label='Q-Agent NAV', color='blue')
    plt.plot(benchmark_nav_history.index, benchmark_nav_history, label='Buy-and-Hold NAV', color='green', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Net Asset Value (NAV)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_rewards(training_rewards, title="Q-Agent Training Rewards Over Episodes"):

    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards, label='Episode Reward', color='purple', alpha=0.7)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()