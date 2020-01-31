# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:15:27 2019

@author: Matthew
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MA.csv')


def bollinger(S,L):
        
        df['MAS'] = df['Adj Close'].rolling(window=S, min_periods=1).mean()
        df['MAL'] = df['Adj Close'].rolling(window=L, min_periods=1).mean()
        
        df.index = range(len(df))
        
        profit_loss = 0
        transaction_num = 0
        position = 'no'
        shares_buy = 0
        for i in range(len(df['Adj Close'])):
            
            if position == 'no':
                
                if df['MAS'][i]>df['MAL'][i]:
                    shares_buy = 100/df['Adj Close'][i]
                    position = 'active'
                else:
                    continue
                
                    
            elif position == 'active':
                    
                if df['MAS'][i]<df['MAL'][i]:
                    profit_loss = profit_loss + (shares_buy*df['Adj Close'][i]-100)
                    shares_buy = 0
                    transaction_num = transaction_num +1
                    position = 'no' 
                else:
                    continue
                
            
        if transaction_num == 0:
            return 0
        
        else:
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


        
S_list = [5,10,15,20,25] 
L_list = [20,30,40,50,60]

df_list = []
df = data.loc[:]

for i in S_list:
    for j in L_list:
        if i < j:
            df_list.append([i,j,bollinger(i,j)])
        elif i >= j:
            continue
        
df_1 = pd.DataFrame(df_list, columns = ['S','L','Avg Profit/Loss'])      


#plot data
a = plt.figure(1)
c = pltcolor(df_1['Avg Profit/Loss'])

plt.scatter(df_1['S'], df_1['L'], s=np.abs(df_1['Avg Profit/Loss'])*5, c=c ,alpha=0.5)
plt.title('2014-2018 Trading strategy')
plt.xlabel('S')
plt.ylabel('L')
a.show()  

print('')
print('From the plot we can see that the best combination of S and L from 2014 to 2018 are as following:\n')
print(df_1[df_1['Avg Profit/Loss']== max(df_1['Avg Profit/Loss'])])
print('')


            