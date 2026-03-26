# Energy Price Simulator

Stochastic simulation of Belgian day-ahead electricity prices using an Ornstein-Uhlenbeck process with seasonal mean reversion, calibrated on real ENTSO-E data.

## Overview

Electricity prices exhibit two key behaviours that standard models fail to capture: **mean reversion** (prices always return to a long-run average) and **seasonality** (prices are systematically higher in winter than in summer). This simulator addresses both using a seasonal OU process:

```
dS = κ(μ(t) - S) dt + σ dW
```

where `μ(t) = μ₀ + A·cos(2πt/T + φ)` is a time-varying mean fitted to monthly price averages.

## Features

- Fetches Belgian day-ahead prices directly from the **ENTSO-E Transparency Platform** via API (or loads from a local CSV cache)
- Fits a **Geometric Brownian Motion** (GBM) as a baseline — and demonstrates why it fails for electricity
- Fits a **seasonal Ornstein-Uhlenbeck** model by OLS regression with optional winsorisation of residuals
- **Monte Carlo simulation** of N price trajectories over a chosen horizon
- **Forward price** computation via the analytical OU formula
- **European call option pricing** by Monte Carlo with 95% confidence interval
- **Rolling backtest** over the full dataset with MAE reporting

## Project Structure

```
.
├── main.py          # Main script — model classes and entry point
├── .env             # ENTSO-E API key (not committed)
└── README.md
```

## Installation

```bash
pip install pandas numpy scipy matplotlib mplcyberpunk python-dotenv entsoe-py
```

## Configuration

Create a `.env` file at the root with your ENTSO-E API token:

```
ENTSOE_KEY=your_token_here
```

To get a token: log in to [transparency.entsoe.eu](https://transparency.entsoe.eu) and generate one under Account Settings → Web API Security Token.

## Usage

```python
dt = 1/8760          # hourly time step (fraction of a year)
T  = 3 * 24          # forecast horizon: 3 days
N  = 100             # number of simulated trajectories

OU = OrnsteinUhlenbeck(data, dt)
OU.fit(winsorize=True, alpha=0.05, seasonal=True)
OU.simulate(N, T)
OU.plot(n_hist=24*5, mean=False)
```

To run the rolling backtest (52-week training window, 1-week test window):

```python
OU.backtest_rolling(train_weeks=52, test_days=7)
```

## Model Parameters

After calibration on Belgian day-ahead prices (March 2023 – March 2026):

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| κ (kappa) | ~623 | Very fast mean reversion — prices return to μ within hours |
| μ₀ | ~81 EUR/MWh | Long-run average price |
| A | ~13 EUR/MWh | Seasonal amplitude (winter vs summer spread) |
| σ | ~1160 | Annualised volatility (high due to price spikes) |

## Limitations

The OU model assumes a single volatility regime. Extreme price spikes (scarcity events, geopolitical shocks) are not well captured and inflate σ. The natural next step is a **jump-diffusion extension** to model spikes as a separate Poisson process.

## Data Source

[ENTSO-E Transparency Platform](https://transparency.entsoe.eu) — Day-Ahead Prices, bidding zone BE (Belgium).
