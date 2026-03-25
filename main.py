import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

plt.style.use("cyberpunk")

DATES_COL = "MTU"
PRICES_COL = "Prices [EUR / MWh]"

class StochasticProcess:

    def __init__(self, prices, dt):

        self.prices = prices.dropna()
        self.dt = dt
        self.trajectories = None

    def plot(self, n_hist=100, n_traj=10):

        hist = self.prices.iloc[-n_hist:]
        hist.plot()

        last_date = hist.index[-1]
        future_dates = pd.date_range(
            start=last_date,
            periods=self.trajectories.shape[1] + 1,
            freq="h"
        )[1:]

        for i in range(min(n_traj, self.trajectories.shape[0])):
            traj = pd.Series(data=self.trajectories[i], index=future_dates)
            traj.plot(alpha=.4, lw=1)

        plt.show()

class GeometricBrownianMotion(StochasticProcess):

    def __init__(self, prices, dt):
        super().__init__(prices, dt)
        self.mu = 0.
        self.sigma = 0.

    def __compute_logreturns(self, winsorize):

        prices_pos = self.prices[self.prices > 0]
        self.__logreturns = np.log(prices_pos/prices_pos.shift(1))
        self.__logreturns = self.__logreturns.dropna()

        if winsorize:
            lower = self.__logreturns[PRICES_COL].quantile(0.01)
            upper = self.__logreturns[PRICES_COL].quantile(0.99)
            self.__logreturns[PRICES_COL] = self.__logreturns[PRICES_COL].clip(lower, upper)

    def show_logreturns(self):

        self.__logreturns.plot()
        plt.show()

    def fit(self, winsorize=False):

        self.__compute_logreturns(winsorize=winsorize)
        mean_logreturns = self.__logreturns.mean().item()
        std_logreturns = self.__logreturns.std().item()
        self.mu = mean_logreturns / self.dt
        self.sigma = std_logreturns / np.sqrt(self.dt)
        print(
            ">>> INFO: Log-returns statistics\n",
            f"         mean: {mean_logreturns:.3f} std: {std_logreturns:.3f}\n",
            "         GBM parameters\n",
            f"         mu: {self.mu:.3f} sigma: {self.sigma:.3f}"
        )

    def simulate(self, N, steps):

        S0 = self.prices.tail(1).values[0]
        self.trajectories = np.zeros((N, steps))

        cst1 = np.exp((self.mu - self.sigma*self.sigma/2) * self.dt)
        cst2 = self.sigma * np.sqrt(self.dt)
        self.trajectories[:, 0] = S0 * cst1 * np.exp(cst2 * np.random.randn(N))

        for i in range(1, steps):
            self.trajectories[:, i] = self.trajectories[:, i-1] * cst1 * np.exp(cst2 * np.random.randn(N))

class OrnsteinUhlenbeck(StochasticProcess):

    def __init__(self, prices, dt):
        super().__init__(prices, dt)
        self.kappa = 0.
        self.mu = 0.
        self.sigma = 0.

    def fit(self, winsorize=False, alpha=0.01):

        X = self.prices.iloc[:-1].to_numpy().flatten()
        Y = self.prices.iloc[1:].to_numpy().flatten() - X

        # OLS regression
        b, a = np.polyfit(X, Y, 1)
        self.kappa = -b / self.dt
        self.mu = a / (self.kappa * self.dt)
        residuals = Y - (a + b * X)

        if winsorize:
            residuals_clipped = pd.Series(residuals).clip(
                pd.Series(residuals).quantile(alpha),
                pd.Series(residuals).quantile(1-alpha)
            )
            self.sigma = np.std(residuals_clipped) / np.sqrt(dt)
        else:
            self.sigma = np.std(residuals) / np.sqrt(self.dt)
        
        print(
            ">>> INFO: Ornstein-Uhlenbeck parameters\n",
            f"         kappa: {self.kappa:.3f} mu: {self.mu:.3f} sigma: {self.sigma:.3f}"
        )

    def simulate(self, N, steps):
        
        self.S0 = self.prices.tail(1).values[0]
        self.trajectories = np.zeros((N, steps))

        self.trajectories[:, 0] = self.S0 + self.kappa * (self.mu - self.S0) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)
        for i in range(1, steps):
            self.trajectories[:, i] = self.trajectories[:, i-1] + self.kappa * (self.mu - self.trajectories[:, i-1]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)

    def pricing_forward(self, T):
        return self.mu + (self.S0 - self.mu)*np.exp(-self.kappa*T)

    def pricing_call(self, T):
        payoff = np.exp(-0.03*T) * np.maximum(self.trajectories[:, int(T / self.dt) - 1] - self.S0, 0)
        mean = np.mean(payoff)
        std = np.std(payoff)
        sqrt_N = np.sqrt(self.trajectories.shape[0])

        CI = (mean - 1.96 * std/sqrt_N, mean + 1.96 * std/sqrt_N)
        return mean, CI

if __name__ == "__main__":

    start_date = '20230324'
    end_date = '20260324'

    fname = start_date + "_" + end_date + ".csv"
    if not os.path.isfile(fname):

        load_dotenv() # you need to have your api key in a .env file
        api_key = os.getenv("ENTSOE_KEY")
        client = EntsoePandasClient(api_key=api_key)
        
        start = pd.Timestamp(start_date, tz='Europe/Brussels')
        end = pd.Timestamp(end_date, tz='Europe/Brussels')
        country_code = 'BE'

        data = client.query_day_ahead_prices(country_code, start=start, end=end)
        data = data.to_frame(name=PRICES_COL)
        data.to_csv(fname, index_label=DATES_COL) 

    data = pd.read_csv(fname, index_col=0, parse_dates=True, date_format="%Y-%m-%d %H:%M:%S%:z")
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.resample('h').mean()   ### recent dt is now 15 min instead of 1h
    
    ### Show historical data
    # data.plot()
    # plt.show()

    dt = 1/8760
    T = int(1/dt * 0.1)
    N = 1
    OU = OrnsteinUhlenbeck(data, dt)

    OU.fit(winsorize=True, alpha=0.05)
    OU.simulate(N, T)
    OU.plot(n_hist=800)