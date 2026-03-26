import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplcyberpunk
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

plt.style.use("cyberpunk")

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "Libertinus Serif",
    # "legend.loc": "best",
    # "image.cmap": "hsv",
    # "grid.linewidth": 0.5,
    # "grid.color": "gray",
    # "figure.titlesize": "large",
    # "figure.titleweight": "bold",
    # "figure.dpi": 125,
    "figure.figsize": (10, 5)
})

DATES_COL = "MTU"
PRICES_COL = "Prices [EUR / MWh]"

class StochasticProcess:
    """ Base class for stochastic processes. """

    def __init__(self, prices, dt):

        self.prices = prices.dropna()
        self.dt = dt
        self.trajectories = None

    def plot(self, n_hist=100, mean=False):
        """ Plot the historical data and the simulated trajectories. """

        hist = self.prices.iloc[-n_hist:]
        hist.plot()

        last_date = hist.index[-1]
        future_dates = pd.date_range(
            start=last_date,
            periods=self.trajectories.shape[1] + 1,
            freq="h"
        )[1:]

        if not mean:
            for i in range(self.trajectories.shape[0]):
                traj = pd.Series(data=self.trajectories[i], index=future_dates)
                traj.plot(alpha=.4, lw=1)
        else:
            traj = pd.Series(data=self.trajectories.mean(axis=0), index=future_dates)
            traj.plot()

        plt.show()

class GeometricBrownianMotion(StochasticProcess):
    """ Geometric Brownian Motion: dS_t = mu * S_t * dt + sigma * S_t * dW_t """

    def __init__(self, prices, dt):
        super().__init__(prices, dt)
        self.mu = 0.
        self.sigma = 0.

    def __compute_logreturns(self, winsorize):
        """ Compute the log-returns time series. """

        prices_pos = self.prices[self.prices > 0]
        self.__logreturns = np.log(prices_pos/prices_pos.shift(1))
        self.__logreturns = self.__logreturns.dropna()

        if winsorize:
            lower = self.__logreturns[PRICES_COL].quantile(0.01)
            upper = self.__logreturns[PRICES_COL].quantile(0.99)
            self.__logreturns[PRICES_COL] = self.__logreturns[PRICES_COL].clip(lower, upper)

    def show_logreturns(self):
        """ Show the log-returns time series. """

        self.__logreturns.plot()
        plt.show()

    def fit(self, winsorize=False):
        """ Fit the GBM parameters using the log-returns statistics. """

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
        """ Simulate N trajectories of length steps. """

        S0 = self.prices.tail(1).values[0]
        self.trajectories = np.zeros((N, steps))

        cst1 = np.exp((self.mu - self.sigma*self.sigma/2) * self.dt)
        cst2 = self.sigma * np.sqrt(self.dt)
        self.trajectories[:, 0] = S0 * cst1 * np.exp(cst2 * np.random.randn(N))

        for i in range(1, steps):
            self.trajectories[:, i] = self.trajectories[:, i-1] * cst1 * np.exp(cst2 * np.random.randn(N))

class OrnsteinUhlenbeck(StochasticProcess):
    """ Ornstein-Uhlenbeck process: dS_t = kappa * (mu(t) - S_t) * dt + sigma * dW_t """

    def __init__(self, prices, dt):
        super().__init__(prices, dt)
        self.kappa = 0.
        self.mu = 0.
        self.sigma = 0.

    def fit(self, winsorize=False, alpha=0.01, seasonal=True, show=True):
        """ Fit the OU parameters using OLS regression. """

        X = self.prices.iloc[:-1].to_numpy().flatten()
        Y = self.prices.iloc[1:].to_numpy().flatten() - X

        # OLS regression
        b, a = np.polyfit(X, Y, 1)
        self.kappa = -b / self.dt
        if not seasonal:
            self.mu = a / (self.kappa * self.dt)
            self.__seasonal = False
        else:
            self.__opt_seasonal_mu(show=show)
            self.__seasonal = True
        residuals = Y - (a + b * X)

        if winsorize:
            residuals_clipped = pd.Series(residuals).clip(
                pd.Series(residuals).quantile(alpha),
                pd.Series(residuals).quantile(1-alpha)
            )
            self.sigma = np.std(residuals_clipped) / np.sqrt(dt)
        else:
            self.sigma = np.std(residuals) / np.sqrt(self.dt)
        
        if not self.__seasonal:
            print(
                ">>> INFO: Ornstein-Uhlenbeck parameters\n",
                f"         kappa: {self.kappa:.3f} mu: {self.mu:.3f} sigma: {self.sigma:.3f}"
            )
        else:
            print(
                ">>> INFO: Ornstein-Uhlenbeck parameters\n",
                f"         kappa: {self.kappa:.3f}\n",
                f"         mu(t): {self.__mu0:.2f} + {self.__A:.2f} * cos(2*pi*t/T + {self.__phi:.2f})\n",
                f"         sigma: {self.sigma:.3f}"
            )

    def simulate(self, N, steps):
        """ Simulate N trajectories of length steps. """
        
        self.S0 = self.prices.tail(1).values[0]
        self.trajectories = np.zeros((N, steps))

        if not self.__seasonal:
            self.trajectories[:, 0] = self.S0 + self.kappa * (self.mu - self.S0) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)
            for i in range(1, steps):
                self.trajectories[:, i] = self.trajectories[:, i-1] + self.kappa * (self.mu - self.trajectories[:, i-1]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)
        else:
            self.trajectories[:, 0] = self.S0 + self.kappa * (self.mu(0) - self.S0) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)
            for i in range(1, steps):
                self.trajectories[:, i] = self.trajectories[:, i-1] + self.kappa * (self.mu(i) - self.trajectories[:, i-1]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(N)
            
    def pricing_forward(self, T):
        """ Price a forward contract with maturity T. """
        return self.mu + (self.S0 - self.mu)*np.exp(-self.kappa*T)

    def pricing_call(self, T):
        """ Price a European call option with maturity T and strike K=S0. """
        payoff = np.exp(-0.03*T) * np.maximum(self.trajectories[:, int(T / self.dt) - 1] - self.S0, 0)
        mean = np.mean(payoff)
        std = np.std(payoff)
        sqrt_N = np.sqrt(self.trajectories.shape[0])

        CI = (mean - 1.96 * std/sqrt_N, mean + 1.96 * std/sqrt_N)
        return mean, CI

    def __seasonal_mu(self, t, mu0, A, phi):
        """ Seasonal mu function. """
        return mu0 + A * np.cos(2 * np.pi * t / 12 + phi)

    def __opt_seasonal_mu(self, show=True):
        """ Fit a seasonal sinusoidal to the monthly average prices. """
        seasons = self.prices.groupby(self.prices.index.month).mean()
        seasons.index -= 1
        xdata = np.arange(start=0, stop=12)
        ydata = seasons.values.flatten()
        popt, pcov = curve_fit(self.__seasonal_mu, xdata, ydata)
        self.__mu0 = popt[0]
        self.__A = popt[1]
        self.__phi = popt[2]

        start_date = pd.Timestamp(data.index[-1])
        # approximate offset wrt initial date 
        offset = (start_date.month - 1 + (start_date.day + start_date.hour/24) / (365/12))/12
        self.mu = lambda t : self.__mu0 + self.__A * np.cos(2 * np.pi * (t * dt + offset) + self.__phi)
        
        if show:
            seasons.plot()
            xspace = np.linspace(0, 11)
            plt.plot(xspace, self.__seasonal_mu(xspace, popt[0], popt[1], popt[2]))
            plt.xticks(xdata, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"])
            plt.show()

    def backtest_rolling(self, train_weeks=52, test_days=7):
        """ Backtest the model using a rolling window approach. """
    
        train_hours = train_weeks * 7 * 24
        test_hours = test_days * 24
        n_periods = (len(self.prices) - train_hours) // test_hours

        all_preds = []
        all_actuals = []
        all_dates = []

        for i in range(n_periods):
            start_train = i * test_hours
            end_train = start_train + train_hours
            end_test = end_train + test_hours

            train_data = self.prices.iloc[start_train:end_train]
            test_data = self.prices.iloc[end_train:end_test]

            model = OrnsteinUhlenbeck(train_data, self.dt)
            model.fit(winsorize=True, alpha=0.05, seasonal=True, show=False)
            model.simulate(N=500, steps=test_hours) # 500 suffisent pour la moyenne

            all_preds.append(model.trajectories.mean(axis=0))
            all_actuals.append(test_data.values.flatten())
            all_dates.append(test_data.index)

        final_preds = np.concatenate(all_preds)
        final_actuals = np.concatenate(all_actuals)
        final_dates = np.concatenate(all_dates)

        plt.figure(figsize=(12, 6))
        plt.plot(final_dates, final_actuals, label="Actual (BE Day-Ahead)", alpha=0.7)
        plt.plot(final_dates, final_preds, label="Predicted (OU Mean)", color="magenta", lw=1.5)
        
        mae = np.mean(np.abs(final_actuals - final_preds))
        plt.title(f"Rolling Backtest - MAE: {mae:.2f} EUR/MWh")
        
        plt.legend()
        plt.show()
        
        return mae
            

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
    T = int(1/dt * 1)
    N = 1000
    OU = OrnsteinUhlenbeck(data, dt)

    # OU.fit(winsorize=True, alpha=0.05, seasonal=True)
    # OU.simulate(N, T)
    # OU.plot(n_hist=800, mean=True)
    
    OU.backtest_rolling(train_weeks=52, test_days=7)