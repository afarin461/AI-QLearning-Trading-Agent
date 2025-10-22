# 🤖 Q-Learning Based Algorithmic Trading Agent

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![RL](https://img.shields.io/badge/Algorithm-Q--Learning-orange)](https://en.wikipedia.org/wiki/Q-learning)
[![Financial Market](https://img.shields.io/badge/Application-Stock_Trading-brightgreen)]()

---

## 🌟 Project Overview & Academic Context

This project implements a **Q-Learning based Reinforcement Learning (RL) agent** to perform **algorithmic trading** on a single financial asset (AAPL). The primary goal is for the agent to learn an **optimal trading policy** to maximize cumulative profit while managing trading costs.

This was developed as an **AI course project** at K. N. Toosi University of Technology (Spring 2024).

---

## 💡 Core RL Components & Implementation

The project utilizes **Tabular Q-Learning** on a carefully engineered **discretized state space** to make daily trading decisions.

### The Agent's Decision Cycle

The agent operates daily, choosing one of three discrete actions:

| Action | Description | Effect on Position |
| :--- | :--- | :--- |
| **0 - Short** | Opens a short position (bets on price drop). | Position = $-1$ |
| **1 - Hold** | Maintains current position (no trade). | Position = $0$ |
| **2 - Long** | Opens a long position (bets on price increase). | Position = $+1$ |

### State Space Engineering (Discretization)

Since Tabular Q-Learning requires a finite state space, three key financial indicators were **discretized** into categorical bins to define the state. The agent's state is a tuple of these three categories, e.g., `('Up', 'Neutral', 'Medium')`.

| Feature | Source File | Categories (Bins) |
| :--- | :--- | :--- |
| **Returns** | `data_preparation.py` | Down ($<$-0.5%), Flat ($\pm$0.5%), Up ($>$$0.5\%$) |
| **RSI (Relative Strength Index)** | `data_preparation.py` | Oversold (0-30), Neutral (30-70), Overbought (70-100) |
| **Volume** | `data_preparation.py` | Low, Medium, High (based on data quantiles) |

---

## 🚀 Getting Started

### Prerequisites

The project requires **Python 3.x** and the following libraries:

```bash
pip install pandas numpy yfinance ta matplotlib
```

### Execution

The `main.py` file handles data fetching, processing, agent training, and final evaluation.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/QLearning-Trading-Agent.git](https://github.com/YourUsername/QLearning-Trading-Agent.git) 
    cd QLearning-Trading-Agent
    ```
2.  Run the full cycle:
    ```bash
    python main.py
    ```
### Project File Structure

| File | Description |
| :--- | :--- |
| **`main.py`** | Main entry point. Defines hyperparameters, orchestrates training, and runs evaluation. |
| **`agent.py`** | Implements the **`QLearningAgent`** class, including Q-table management and $\epsilon$-greedy action selection. |
| **`environment.py`** | Implements the **`TradingEnvironment`** class, handling rewards calculation, NAV updates, and trading costs. |
| **`data_preparation.py`** | Responsible for fetching historical data (AAPL), calculating technical indicators, and **state discretization**. |
| **`analysis.py`** | Contains utility functions for financial metrics: **Sharpe Ratio** calculation, **Max Drawdown**, and plotting. |
| **`report.txt`** | The comprehensive academic report for the course submission. |
| **`console_output.txt`** | The training and evaluation results printed to the console. |
| **`training_rewards_plot.png`** | Output plot showing reward convergence over episodes. |
| **`performance_comparison_plot.png`** | Output plot comparing agent NAV vs. Buy-and-Hold benchmark. |

### 📊 Output and Results

The project generates two types of output that are essential for evaluation:

1.  **Console Output (`console_output.txt`):**
    * During the execution of `python main.py`, all training progress (Total Reward, Epsilon Decay, Final NAV per episode) and the final evaluation metrics (**Sharpe Ratio** and **Max Drawdown**) are saved to the file **`console_output.txt`**.
    * This file provides the *numeric* comparison data.

2.  **Visualizations (`.png` files):**
    * The script generates two graphical files in the root directory after execution:
        * **`training_rewards_plot.png`**: Shows the learning curve (convergence of rewards).
        * **`performance_comparison_plot.png`**: Visually compares the Q-Agent's NAV growth against the Buy-and-Hold benchmark.



    
