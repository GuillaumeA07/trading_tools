import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pylab import plt,mpl
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
import yfinance as yf
import scipy.optimize as sco
np.set_printoptions(precision=5,suppress=True,formatter={'float':lambda x: f'{x:6.3f}'})

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000 # number of portfolio to test

# stocks to test
symbols = ['AAPL', 'MSFT', 'TSLA', 'GE', 'AMZN', 'GOOG']
#symbols = ['GC=F', 'SI=F','XWD.TO','EEM','IUSN.DE']
# historical data - define START and END 
start_date = 2010
end_date = 2022


def download_data():

    symbol_data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        symbol_data[symbol] = ticker.history(start=str(start_date)+'-01-01', end=str(end_date)+'-12-31')['Close']

    return pd.DataFrame(symbol_data)


def port_ret(log_returns,weights,):
    return np.sum(log_returns.mean() *weights) *NUM_TRADING_DAYS

def port_vol(log_returns,weights):
    return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*NUM_TRADING_DAYS,weights)))

def port_shape(log_returns,weights):
    return port_ret(log_returns,weights) / port_vol(log_returns,weights)


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

    weights = len(rets.columns)*[1/len(rets.columns)]
    w = np.random.random((NUM_PORTFOLIOS,len(symbols)))
    w = (w.T/w.sum(axis=1)).T

    pvr = [(port_vol(rets[symbols],weights),
        port_ret(rets[symbols],weights))
        for weights in w]
    pvr = np.array(pvr)
    psr = pvr[:,1] /pvr[:,0]

    # Plot all portfolios, comment if needed
    plt.figure(figsize=(10,6))
    fig = plt.scatter(pvr[:,0],pvr[:,1],c=psr,cmap='coolwarm')
    cb = plt.colorbar(fig)
    cb.set_label('Sharpe Ratio')
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.title(' | '.join(symbols))
    plt.show()
    # stop comment

    bnds = len(symbols) * [(0,1),]
    cons = ({'type': 'eq', 'fun': lambda weights: weights.sum() - 1})

    opt_weights = {}
    for year in range(start_date, end_date):
        rets_ = rets[symbols].loc[f'{year}-01-01':f'{year}-12-31']
        ow = sco.minimize(lambda weights: -port_shape(rets_,weights), len(symbols) * [1/len(symbols)], bounds=bnds, constraints=cons)['x']
        opt_weights[year] = ow
    
    # Print yearly PTF
    
    for k,v in opt_weights.items():
        print(k,': ', [symbols[x] + ' ~ '+str(round(list(v)[x]*100,3))+'  ' for x in range(len(list(v)))])
    

    res = pd.DataFrame()
    for year in range(start_date,end_date):
        rets_ = rets[symbols].loc[f'{year}-01-01':f'{year}-12-31']
        epv = port_vol(rets_,opt_weights[year])
        epr = port_ret(rets_,opt_weights[year])
        esr = epr / epv
        rets_ = rets[symbols].loc[f'{year+1}-01-01':f'{year+1}-12-31']
        rpv = port_vol(rets_,opt_weights[year])
        rpr = port_ret(rets_,opt_weights[year])
        rsr = rpr / rpv
        res = res.append(pd.DataFrame({'epv': epv, 'epr': epr, 'esr': esr, 'rpv':rpv, 'rpr':rpr, 'rsr': rsr},
        index=[year+1]))
    
    # Plot volatility
    res[['epv', 'rpv']].plot(kind='bar',figsize=(10,6),title= 'Expected vs Realized PTF Volatility')
    plt.show()

    # Plot return
    res[['epr', 'rpr']].plot(kind='bar',figsize=(10,6),title= 'Expected vs Realized PTF Return')
    plt.show()

    # Plot sharpe ratio
    res[['esr', 'rsr']].plot(kind='bar',figsize=(10,6),title= 'Expected vs Realized PTF Sharpe Ratio')
    plt.show()