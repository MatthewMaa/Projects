# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:09:15 2019

@author: Yazhuo Ma
"""

from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd


data = pd.read_csv('MA.csv')

year = [2014,2015,2016,2017,2018]

df_list1 = []

for i in year:
    
    dayreturn = data.loc[data.loc[:,'Year'] == i,]['Return'].values

    # Question 1
    print('Days with positive returns in ', i, ':',len(dayreturn[dayreturn>0]))
    print('Days with negative returns in ', i, ':',len(dayreturn[dayreturn<0]))

    
    # Question 2
    
    # trading days
    trdays = len(dayreturn)
    
    #u
    u = np.mean(dayreturn)
    
    # %days less than u
    less_u = "{:.2%}".format(len(dayreturn[dayreturn<u])/trdays)
    
    # %days greater than u
    more_u = "{:.2%}".format(len(dayreturn[dayreturn>u])/trdays)


    # Store in a list & append
    info1 = [i,trdays, u, less_u, more_u]
    df_list1.append(info1)

df_1 = pd.DataFrame(df_list1, columns = ['Year','Trading days','µ','% days < µ','% days > µ'])
print()
print(df_1)
print('')
print('From the table and answer of question 1 we can draw a conclusion that positive return days are more than negative return days in every year.')
print('And starting from 2015, the percentage of days with returns greater than µ is a little bit over 50%, which means the company makes money gradually instead of at one time.')

# Question 3&4

df_list2 = []
for i in year:
    
    dayreturn = data.loc[data.loc[:,'Year'] == i,]['Return'].values
    
    # trading days
    trdays = len(dayreturn)
    
    #u
    u = np.mean(dayreturn)
    
    #sigma
    sigma = np.std(dayreturn)
    
    
    #% days < µ − 2σ
    less = "{:.2%}".format(len(dayreturn[dayreturn<(u-2*sigma)])/trdays)

    #% days > µ + 2σ
    more = "{:.2%}".format(len(dayreturn[dayreturn>(u+2*sigma)])/trdays)

    # Store in a list & append
    info2 = [i, trdays, u, sigma, less, more]
    df_list2.append(info2)

df_2 = pd.DataFrame(df_list2, columns = ['Year','Trading days','µ','σ','% days < µ − 2σ','% days > µ + 2σ'])
print()
print(df_2)
print('')
print('From the table we could know the following information  about every year''s day returns distribution:')
print('2014: Not normal distribution. Skewed to the left.')
print('2015: Not normal distribution. Skewed to the left.')
print('2016: Not normal distribution. Skewed to the left.')
print('2017: Almost normal distribution. A little skewed to the left.')
print('2018: Not normal distribution. Skewed to the right.')




