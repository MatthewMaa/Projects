# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:06:19 2019

@author: Yazhuo Ma
"""

from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MA.csv')





def bollinger(W,k):
        df['Adj Close'].fillna(0, inplace = True)
        df['MA'] = df['Adj Close'].rolling(window=W, min_periods=1).mean()
        df['Volatility'] = df['Adj Close'].rolling(window=W, min_periods=1).std()
        
        df.index = range(len(df))
        
        profit_loss = 0
        transaction_num = 0
        position = 'no'
        shares_sell = 0
        shares_buy = 0
        for i in range(len(df['Adj Close'])):
            
            if df['Adj Close'][i] > (df['MA'][i] + k*df['Volatility'][i]):
                
                if position == 'no':
                    shares_sell = 100/df['Adj Close'][i]
                    position = 'short'
                    
                elif position == 'short':
                    continue
                
                elif position == 'long':
                    profit_loss = profit_loss + (shares_buy*df['Adj Close'][i]-100)
                    position = 'no'
                    transaction_num = transaction_num +1
                    
            elif df['Adj Close'][i] < (df['MA'][i] - k*df['Volatility'][i]):
                    
                if position == 'no':
                    shares_buy = 100/df['Adj Close'][i]
                    position = 'long'
                    
                elif position == 'long':
                    continue
                
                elif position == 'short':
                    profit_loss = profit_loss + (100-shares_sell*df['Adj Close'][i])
                    position = 'no'
                    transaction_num = transaction_num +1
                    
            elif df['Adj Close'][i] > (df['MA'][i] - k*df['Volatility'][i]) and df['Adj Close'][i] < (df['MA'][i] + k*df['Volatility'][i]):
                continue
            
        if transaction_num == 0:
            return 0
        
        return profit_loss/ transaction_num
    
def pltcolor(x):
    colors=[]
    for i in range(len(x)):
        if x[i] >= 0:
            colors.append('green')
        elif x[i] < 0:
            colors.append('red')
        else:
            colors.append('grey')
    return colors


k_list = [0.5,1,1.5,2,2.5] 



#Question 1&2
df_list = []
df = data.loc[data.loc[:,'Year'] == 2017,]

for i in k_list:
    for j in range(10,51):
        df_list.append([i,j,bollinger(j,i)])


df_1 = pd.DataFrame(df_list, columns = ['k','W','Avg Profit/Loss'])


#plot data
a = plt.figure(1)
c = pltcolor(df_1['Avg Profit/Loss'])

plt.scatter(df_1['W'], df_1['k'], s=np.abs(df_1['Avg Profit/Loss'])*5, c=c ,alpha=0.5)
plt.title('2017')
a.show()


print('')
print('From the plot we can see that the best combination of k and W in 2017 are as following:')
print(df_1[df_1['Avg Profit/Loss']== max(df_1['Avg Profit/Loss'])])
print('')

#Question 3
b = plt.figure(2)
df_list = []
df = data.loc[data.loc[:,'Year'] == 2018,]

for i in k_list:
    for j in range(10,51):
        df_list.append([i,j,bollinger(j,i)])


df_2 = pd.DataFrame(df_list, columns = ['k','W','Avg Profit/Loss'])


#plot data
c = pltcolor(df_2['Avg Profit/Loss'])

plt.scatter(df_2['W'], df_2['k'], s=np.abs(df_2['Avg Profit/Loss'])*5, c=c ,alpha=0.5)
plt.title('2018')

b.show()

print('')
print('From the plot we can see that the best combination of k and W in 2018 are as following:')
print(df_2[df_2['Avg Profit/Loss']== max(df_2['Avg Profit/Loss'])])
print('')


