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

The project utilizes **Tabular Q-Learning** on a carefully engineered **discretized state space**.

### The Agent's Decision Cycle

The agent operates daily, choosing one of three discrete actions:

| Action | Description | Effect on Position |
| :--- | :--- | :--- |
| **0 - Short** | Opens a short position (bets on price drop). | Position = $-1$ |
| **1 - Hold** | Maintains current position (no trade). | Position = $0$ |
| **2 - Long** | Opens a long position (bets on price increase). | Position = $+1$ |

### State Space Engineering (Discretization)

Since Tabular Q-Learning requires a finite state space, three key financial indicators were **discretized** into categorical bins to define the state:

| Feature | Source File | Categories (Example Bins) |
| :--- | :--- | :--- |
| **Returns** | `data_preparation.py` | Down, Flat, Up |
| **RSI (Relative Strength Index)** | `data_preparation.py` | Oversold (0-30), Neutral (30-70), Overbought (70-100) |
| **Volume** | `data_preparation.py` | Low, Medium, High (based on quantiles) |

**The Agent's State** is a tuple of these three categories, e.g., `('Up', 'Neutral', 'Medium')`.

---

## 🚀 Getting Started

### Prerequisites

The project requires Python 3.x and the following libraries:

```bash
pip install pandas numpy yfinance ta matplotlib
