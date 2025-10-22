# 🤖 Q-Learning Based Algorithmic Trading Agent

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/) 
[![RL](https://img.shields.io/badge/Algorithm-Q--Learning-orange)](https://en.wikipedia.org/wiki/Q-learning)

---

## 🎯 Project Overview

This is an **Artificial Intelligence course project** dedicated to solving a sequential decision-making problem (financial trading) using **Reinforcement Learning (RL)**.

The agent is trained to maximize its Net Asset Value (NAV) by selecting an optimal trading action (**Short**, **Hold**, or **Long**) daily, while being penalized for **trading costs**.

## 💡 Core RL Components

| Component | File | Description |
| :--- | :--- | :--- |
| **Agent** | `agent.py` | Implements the **Tabular Q-Learning** algorithm, managing the Q-table and implementing the **$\epsilon$-greedy** action selection policy. |
| **Environment** | `environment.py` | Simulates the trading environment, calculates **rewards** based on daily returns and trading costs, and updates the **NAV**. |
| **State Space** | `data_preparation.py` | Converts continuous market features (Returns, RSI, Volume) into a **finite, discretized state space** (e.g., `('Up', 'Neutral', 'High')`) necessary for tabular Q-learning. |

## 📊 Performance and Evaluation

The agent's performance is compared against a **Buy-and-Hold** benchmark strategy.

### Key Results (from `console_output.txt`)

| Metric | Q-Agent (Evaluation) | Buy-and-Hold Benchmark |
| :--- | :--- | :--- |
| **Final NAV** | (مقدار نهایی NAV از خروجی کنسول) | (مقدار نهایی NAV از خروجی کنسول) |
| **Sharpe Ratio** | (Sharpe Ratio از خروجی کنسول) | (Sharpe Ratio از خروجی کنسول) |
| **Max Drawdown** | (Max Drawdown از خروجی کنسول) | N/A |

### Visualizations

The two main plots demonstrate the agent's learning process and final performance:

1.  **Training Rewards:** Shows the accumulation of rewards over 2000 episodes, indicating policy convergence.
2.  **Performance Comparison:** Compares the Q-Agent's NAV trajectory against the Buy-and-Hold strategy.

---

## 🚀 How to Run

### Prerequisites

You need **Python 3.x** and the required libraries (pandas, numpy, yfinance, ta, matplotlib).

```bash
pip install pandas numpy yfinance ta matplotlib
