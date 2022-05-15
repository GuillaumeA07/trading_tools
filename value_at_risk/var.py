# Value at Risk
import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date)
    data[stock] = ticker['Adj Close']
    return pd.DataFrame(data)


# Calculate the VaR tomorrow (n=1)
def calculate_var(position, c, mu, sigma):
    var = position * (mu - sigma * norm.ppf(1-c))
    return var


# Calculate the VaR for any days in the future
def calculate_var_n(position, c, mu, sigma, n):
    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1-c))
    return var


if __name__ == '__main__':

    start = datetime.datetime(2014, 1, 1) # start_date
    end = datetime.datetime(2022, 1, 1) # end_date
    symbol = 'GOOGL' # Symbol 
    S = 1e6  # Amount of investment
    c = 0.90 # Confidence level

    stock_data = download_data(symbol, start, end)
    stock_data['returns'] = np.log(stock_data[symbol] / stock_data[symbol].shift(1))
    stock_data = stock_data[1:]
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])

    print('Value at risk is: $%0.2f' % calculate_var_n(S, c, mu, sigma, 1))



