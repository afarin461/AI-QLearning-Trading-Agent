import pandas as pd
import numpy as np
import data_preparation as dp
import environment as env_mod
import agent as agent_mod
import analysis as anl

TICKER_SYMBOL = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

RETURNS_BINS = [-np.inf, -0.005, 0.005, np.inf] 
RSI_BINS = [0, 30, 70, 100]

TRADING_COST = 0.0005
EPISODE_MAX_STEPS = None

ALPHA = 0.5
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

NUM_TRAINING_EPISODES = 2000
NUM_EVAL_EPISODES = 1

if __name__ == "__main__":
    print("--- Starting Q-Learning Trading Project ---")

    raw_data_df = dp.fetch_historical_data(TICKER_SYMBOL, START_DATE, END_DATE)
    
    if raw_data_df is None or raw_data_df.empty:
        print("Exiting: No valid data to process.")
        exit()

    processed_data_for_env = dp.calculate_and_discretize_features(
        raw_data_df, 
        returns_bins=RETURNS_BINS, 
        rsi_bins=RSI_BINS
    )

    if processed_data_for_env.empty:
        print("Exiting: Processed data is empty after feature calculation and NaN removal.")
        exit()

    unique_states = tuple(processed_data_for_env['State'].unique())
    print(f"\nDiscretized State Space size: {len(unique_states)}")

    trading_env = env_mod.TradingEnvironment(
        data_df=processed_data_for_env, 
        trading_cost=TRADING_COST,
        episode_max_steps=EPISODE_MAX_STEPS
    )

    q_agent = agent_mod.QLearningAgent(
        state_space=unique_states, 
        action_space_size=len(trading_env.actions), 
        alpha=ALPHA, 
        gamma=GAMMA,
        epsilon=EPSILON, 
        epsilon_decay=EPSILON_DECAY, 
        epsilon_min=EPSILON_MIN
    )

    print(f"\n--- Starting Training for {NUM_TRAINING_EPISODES} Episodes ---")
    training_rewards = []
    for episode in range(NUM_TRAINING_EPISODES):
        state = trading_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = q_agent.choose_action(state)
            next_state, reward, done, info = trading_env.step(action)
            
            q_agent.update_q_value(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break 
        
        training_rewards.append(episode_reward)
        q_agent.decay_epsilon()
        
        if (episode + 1) % (NUM_TRAINING_EPISODES // 10) == 0 or episode == NUM_TRAINING_EPISODES - 1:
            print(f"Episode {episode + 1}/{NUM_TRAINING_EPISODES}: "
                  f"Total Reward = {episode_reward:.4f}, "
                  f"Epsilon = {q_agent.epsilon:.4f}, "
                  f"Final NAV = {trading_env.nav:.4f}")
    
    print("--- Training Completed ---")
    anl.plot_training_rewards(training_rewards)

    print(f"\n--- Starting Evaluation for {NUM_EVAL_EPISODES} Episodes ---")
    eval_agent_nav_history = pd.Series()
    
    original_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0
    
    for eval_episode in range(NUM_EVAL_EPISODES):
        state = trading_env.reset()
        done = False
        current_episode_nav_history = [trading_env.initial_nav]
        
        while not done:
            action = q_agent.choose_action(state)
            next_state, reward, done, info = trading_env.step(action)
            
            current_episode_nav_history.append(info["NAV"])
            
            state = next_state
            if done:
                break
        
        eval_agent_nav_history = pd.Series(current_episode_nav_history, 
                                           index=processed_data_for_env.index[:len(current_episode_nav_history)])
        
        agent_daily_returns = eval_agent_nav_history.pct_change().dropna()
        
        print(f"Evaluation Episode {eval_episode + 1}: Final NAV = {eval_agent_nav_history.iloc[-1]:.4f}")
        print(f"  Sharpe Ratio: {anl.calculate_sharpe_ratio(agent_daily_returns):.4f}")
        print(f"  Max Drawdown: {anl.calculate_max_drawdown(eval_agent_nav_history):.4f}")

    q_agent.epsilon = original_epsilon
    print("--- Evaluation Completed ---")

    print("\n--- Buy-and-Hold Benchmark ---")
    benchmark_nav_history = anl.buy_and_hold_benchmark(raw_data_df['Close'].loc[processed_data_for_env.index])
    
    print(f"Buy-and-Hold Final NAV: {benchmark_nav_history.iloc[-1]:.4f}")
    
    print("\n--- Plotting Performance ---")
    anl.plot_performance(eval_agent_nav_history, benchmark_nav_history)

    print("\n--- Project Finished ---")