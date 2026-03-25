import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class StochasticProcess:

    def __init__(self, prices, dt):

        self.prices = prices.dropna()
        self.dt = dt
        self.trajectories = None

    def plot(self):
        for i in range(10):
            plt.plot(self.trajectories[i][:100], alpha=0.3, lw=0.8)
        plt.show()

class GeometricBrownianMotion(StochasticProcess):

    def __init__(self, prices, dt):
        super().__init__(prices, dt)
        self.mu = 0.
        self.sigma = 0.

    def __compute_logreturns(self, winsorize=True):

        prices_pos = self.prices[self.prices > 0]
        self.__logreturns = np.log(prices_pos/prices_pos.shift(1))
        self.__logreturns = self.__logreturns.dropna()

        if winsorize:
            lower = self.__logreturns.quantile(0.01)
            upper = self.__logreturns.quantile(0.99)
            self.__logreturns = self.__logreturns.clip(lower, upper)

    def fit(self):

        self.__compute_logreturns()
        mean_logreturns = self.__logreturns.mean()
        std_logreturns = self.__logreturns.std()
        self.mu = mean_logreturns / self.dt
        self.sigma = std_logreturns / np.sqrt(self.dt)

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

    def fit(self):

        X = self.prices.iloc[:-1].to_numpy()
        Y = self.prices.iloc[1:].to_numpy() - X

        # OLS regression
        b, a = np.polyfit(X, Y, 1)
        self.kappa = -b / self.dt
        self.mu = a / (self.kappa * self.dt)
        residuals = Y - (a + b * X)
        self.sigma = np.std(residuals) / np.sqrt(self.dt)

    def simulate(self, N, steps):
        
        S0 = self.prices.tail(1).values[0]
        self.trajectories = np.zeros((N, steps))

        self.trajectories[:, 0] = S0 + self.kappa * (self.mu - S0) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)
        for i in range(1, steps):
            self.trajectories[:, i] = self.trajectories[:, i-1] + self.kappa * (self.mu - self.trajectories[:, i-1]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)

if __name__ == "__main__":

    data = pd.read_csv("be_day-ahead_prices.csv")
    data["MTU"] = pd.to_datetime(data["MTU"])
    prices = pd.Series(data=data["[EUR / MWh]"].values, index=data["MTU"], name="[EUR / MWh]")
    prices = prices.sort_index()

    dt = 1/8760
    GBM = GeometricBrownianMotion(prices, dt)
    OU = OrnsteinUhlenbeck(prices, dt)

    GBM.fit()
    GBM.simulate(10, 100)
    GBM.plot()

    OU.fit()
    OU.simulate(10, 100)
    OU.plot()
