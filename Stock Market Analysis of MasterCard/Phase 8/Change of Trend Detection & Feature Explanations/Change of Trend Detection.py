# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:30:42 2019

@author: Matthew
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import f as fisher_f
import warnings
warnings.simplefilter('ignore')
# data preprocessing
data = pd.read_csv('MA.csv')

# 2017
df = data.loc[data['Year'] == 2017,]

df_2017 = df.dropna().reset_index(drop = True)
#2018
df = data.loc[data['Year'] == 2018,]

df_2018 = df.dropna().reset_index(drop = True)



def get_answer(group):
    
    # to find out best k
    k_list = []
    totalsse_list = []
    n = len(group)
    for k in range(2,n):
        
        X1 = np.array(range(1,k+1)).reshape(-1, 1)
        y1 = np.array(group['Adj Close'][0:k])
        X2 = np.array(range(k+1,n+1)).reshape(-1, 1)
        y2 = np.array(group['Adj Close'][k:])
        
        model = LinearRegression(fit_intercept = True)
        model.fit(X1,y1)
        
        
        squared_errors_1 = (y1 - model.predict(X1)) ** 2
        sse1 = np.sum(squared_errors_1)
        
        model.fit(X2,y2)
        
        
        squared_errors_2 = (y2 - model.predict(X2)) ** 2
        sse2 = np.sum(squared_errors_2)
        
        k_list.append(k)
        totalsse_list.append(sse1+sse2)
    
    bestsse_index = totalsse_list.index(min(totalsse_list))
    bestk = k_list[bestsse_index]
    
    # to compute F test
    X = np.array(range(1,n+1)).reshape(-1, 1)
    y = np.array(group['Adj Close'])
    X1 = np.array(range(1,bestk+1)).reshape(-1, 1)
    y1 = np.array(group['Adj Close'][0:bestk])
    X2 = np.array(range(bestk+1,n+1)).reshape(-1, 1)
    y2 = np.array(group['Adj Close'][bestk:])
    
    model = LinearRegression(fit_intercept = True)
    
    model.fit(X,y)
    squared_errors = (y - model.predict(X)) ** 2
    L = np.sum(squared_errors)
    
    model.fit(X1,y1)
    squared_errors_1 = (y1 - model.predict(X1)) ** 2
    L1 = np.sum(squared_errors_1)
    
    model.fit(X2,y2)
    squared_errors_2 = (y2 - model.predict(X2)) ** 2
    L2 = np.sum(squared_errors_2)
    
    F = (L-(L1+L2))/2 * ((L1+L2)/(n-4))**-1
    
    p_value = fisher_f.cdf(F,2,n-4)
    
    status = 'Yes'
    
    if p_value >0.1:
        status = 'Yes'
    elif p_value <= 0.1:
        status = 'No'
        
    
    return status

#Question 1

df_answer1 = df_2017.groupby('Month').apply(get_answer).reset_index()
df_answer1 = df_answer1.rename(columns = {0:'Trend'})

df_answer2 = df_2018.groupby('Month').apply(get_answer).reset_index()
df_answer2 = df_answer2.rename(columns = {0:'Trend'})

print()
print('Question 1:\n')
print('2017:')
print(df_answer1)
print('2018:')
print(df_answer2)

# Question 2
print()
print('Question 2:\n')
print("Number of months with signiticant price chnages:{}".format(sum(df_answer1.Trend == 'Yes')+sum(df_answer2.Trend == 'Yes')))



# Question 3
print()
print('Question 3:\n')
print('Both 2017 and 2018 have 12 changing months, so they are equal.')