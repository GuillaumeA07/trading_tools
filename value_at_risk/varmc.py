# Value at Risk Monte Carlo Simulation
import numpy as np
import yfinance as yf
import datetime
import pandas as pd


def download_data(stock, start, end):
    data = {}
    ticker = yf.download(stock, start, end)
    data['Adj Close'] = ticker['Adj Close']
    return pd.DataFrame(data)


class ValueAtRiskMonteCarlo:

    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.n) * rand)
        stock_price = np.sort(stock_price)
        percentile = np.percentile(stock_price, (1 - self.c) * 100)
        return self.S - percentile


if __name__ == "__main__":
    S = 1e6  # Investment amount
    c = 0.95  # Condifence level
    n = 1  # Keep it to 1 (days in the future for MC path)
    iterations = 100000 # Number of paths in the Monte-Carlo simulation
    start_date = datetime.datetime(2014, 1, 1) # start_date
    end_date = datetime.datetime(2022, 1, 1) # end_date
    symbol = 'GOOGL' # Symbol

    data = download_data(symbol, start_date, end_date)
    data['returns'] = data['Adj Close'].pct_change()
    mu = np.mean(data['returns'])
    sigma = np.std(data['returns'])
    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)

    print('Value at risk with Monte-Carlo simulation: $%0.2f' % model.simulation())
	
	