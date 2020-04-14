# This is an exercise for Financial stock Portfolio Optimization using MonteCarlo, on the following stocks: Google, Apple, AMD, Nvidia, RGA, Berkshire Hathawa, P&G, Unilever, Disney 

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as datetime

#Fetch data from yahoo and save under DataFrame named 'data'
stock = ['GOOGL', 'AAPL', 'AMD', 'NVDA', 'RGA', 'BRK.A', 'PG', 'UN', 'DIS']
#tock = ['BAC', 'GS', 'JPM', 'MS']
yester=datetime.date.today()-pd.DateOffset(years=5)
today=datetime.date.today()-pd.DateOffset(days=1)
data = web.DataReader(stock, data_source='iex', start=yester, end=today)
stportf=data['close']

stret=stportf.pct_change()

mret = stret.mean()
covret= stret.cov()
num_iter= 1000000
sim_res = np.zeros((4+len(stock)-1,num_iter))

for i in range(num_iter):
        
#Select random weights and normalize to set the sum to 1
        weights = np.array(np.random.random(len(stock)))
        weights /= np.sum(weights)

#Calculate the return and standard deviation for every step
        portfolio_return = np.sum(mret * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(covret, weights)))

#Store all the results in a defined array
        sim_res[0,i] = portfolio_return
        sim_res[1,i] = portfolio_std_dev

#Calculate Sharpe ratio and store it in the array
        sim_res[2,i] = sim_res[0,i] / sim_res[1,i]

#Save the weights in the array
        for j in range(len(weights)):
                sim_res[j+3,i] = weights[j]
       
labels=['ret','stdev','sharpe']
labels=labels+stock
sim_frame = pd.DataFrame(sim_res.T,columns=labels)

#Spot the position of the portfolio with highest Sharpe Ratio (best return to risk ratio)
max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()]

#Spot the position of the portfolio with minimum Standard Deviation
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]

print("The portfolio for max Sharpe Ratio:\n", max_sharpe)
print("The portfolio for min risk:\n", min_std)

                
