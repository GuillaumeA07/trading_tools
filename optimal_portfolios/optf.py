import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000 # number of portfolio to test

# stocks to test
#symbols = ['AAPL', 'MSFT', 'TSLA', 'GE', 'AMZN', 'GOOG']
symbols = ['GC=F', 'SI=F','XWD.TO','EEM','IUSN.DE']
# historical data - define START and END 
start_date = '2015-01-01'
end_date = '2022-05-01'


def download_data():

    symbol_data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        symbol_data[symbol] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(symbol_data)

def port_ret(weights,log_returns):
    return np.sum(log_returns.mean() *weights) *NUM_TRADING_DAYS

def port_vol(weights,log_returns):
    return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*NUM_TRADING_DAYS,weights)))

def min_func_sharpe(weights,log_returns):
    return -port_ret(weights,log_returns) / port_vol(weights,log_returns)


if __name__ == '__main__':
    data = download_data()
    data.dropna()
    noa = len(symbols)
    # Plot stocks returnss
    #(data/data.iloc[0]*100).plot(figsize=(10,6))# Comment if needed
    #plt.show()# Comment if needed

    rets = np.log(data / data.shift(1))
    #print(rets.mean()*NUM_TRADING_DAYS)
    #print(rets.cov()*NUM_TRADING_DAYS)

    # Plot log returns
    #rets.hist(bins=50,figsize=(10,8)) # Comment if needed
    #plt.show() # Comment if needed

    prets = []
    pvols = []

    for p in range(NUM_PORTFOLIOS):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        prets.append(port_ret(weights,rets))
        pvols.append(port_vol(weights,rets))
    prets = np.array(prets)
    pvols = np.array(pvols)

    # Plot all portfolios, comment if needed
    plt.figure(figsize=(10,6))
    plt.scatter(pvols,prets,c=prets/pvols,marker='o',cmap='coolwarm')
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()
    # stop comment

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0,1) for x in range(noa))
    eweights = np.array(noa * [1./noa,])

    # Optimization part sharpe ratio
    opts = sco.minimize(min_func_sharpe, eweights,rets, method='SLSQP', bounds=bnds, constraints=cons)
    print('\nBest Sharpe Ratio Portfolio (%): ',[symbols[x] + ' ~ '+str(round(list(opts['x'])[x]*100,3))+'  ' for x in range(len(list(opts['x'])))])
    print('Volatility of selected PTF: ', round(port_vol(opts['x'],rets)*100,3), ' %')
    print('Return of selected PTF: ', round(port_ret(opts['x'],rets) / port_vol(opts['x'],rets)*100,4), ' %')

     # Optimization part volatilitw
    optv = sco.minimize(port_vol, eweights,rets, method='SLSQP', bounds=bnds, constraints=cons)
    print('\nBest Low Volatility Portfolio (%): ',[symbols[x] + ' ~ '+str(round(list(optv['x'])[x]*100,3))+'  ' for x in range(len(list(optv['x'])))])
    print('Volatility of selected PTF: ', round(port_vol(optv['x'],rets)*100,3), ' %')
    print('Return of selected PTF: ', round(port_ret(optv['x'],rets) / port_vol(optv['x'],rets)*100,4), ' %\n')

    # Efficient frontier
    cons = ({'type': 'eq', 'fun': lambda x: port_ret(x,rets)-tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bnds = tuple((0,1) for x in weights)

    trets = np.linspace(0.05,0.2,50)
    tvols= []
    for tret in trets:
        res= sco.minimize(port_vol, eweights, rets,method='SLSQP',bounds=bnds,constraints=cons)
        tvols.append(res['fun'])
    tvols = np.array(tvols)


    # Plot efficient frontier, comment if needed
    plt.figure(figsize=(10,6))
    plt.scatter(pvols,prets,c=prets/pvols,marker='.',alpha=0.8,cmap='coolwarm')
    plt.plot(tvols,trets,'b',lw=0.4)
    plt.plot(port_vol(opts['x'],rets),port_ret(opts['x'],rets),'y*',markersize=15.0)
    plt.plot(port_vol(optv['x'],rets),port_ret(optv['x'],rets),'y*',markersize=15.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()
    # stop comment
