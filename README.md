

````markdown
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
````

### Execution

The `main.py` file handles data fetching, processing, agent training (2000 episodes), and final evaluation.

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
| **`main.py`** | Main entry point. Orchestrates training, evaluation, and plotting. |
| **`agent.py`** | Implements the **`QLearningAgent`** class, including Q-table and $\epsilon$-greedy policy. |
| **`environment.py`** | Implements the **`TradingEnvironment`** class, handling rewards, NAV, and trading costs. |
| **`data_preparation.py`** | Fetches data (AAPL), calculates indicators (RSI), and performs **state discretization**. |
| **`analysis.py`** | Contains functions for financial metrics: **Sharpe Ratio** and **Max Drawdown**. |
| **`report.txt`** | The comprehensive academic report for the course submission. |

-----

## 📈 Results and Analysis

The project evaluates the trained Q-Agent against a naive **Buy-and-Hold** strategy over the same time period (2020-01-01 to 2024-01-01).

### Training Convergence

The `training_rewards_plot.png` demonstrates the agent's learning stability over 2000 episodes:

### Evaluation Metrics

The final evaluation provides key performance indicators (Values extracted from `console_output.txt`):

| Metric | Q-Agent (Evaluation) | Buy-and-Hold Benchmark |
| :--- | :--- | :--- |
| **Final NAV** | **3.6190** | **2.2858** |
| **Sharpe Ratio** | **1.2173** | **0.8710** |
| **Max Drawdown** | **0.4285** | N/A |

### NAV Trajectory Comparison

The final plot compares the Net Asset Value (NAV) growth of the Q-Agent against the benchmark:

-----

## ⚙️ Configuration Parameters

The key hyperparameters defined in `main.py` used for training are:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `ALPHA` ($\alpha$) | $0.5$ | Learning Rate for Q-Learning. |
| `GAMMA` ($\gamma$) | $0.99$ | Discount Factor. |
| `EPSILON` ($\epsilon$) | $1.0$ | Initial exploration rate. |
| `EPSILON_DECAY` | $0.999$ | Decay rate per episode. |
| `NUM_TRAINING_EPISODES` | $2000$ | Total number of training runs. |
| `TRADING_COST` | $0.0005$ | Transaction cost per trade (0.05%). |

```
```
