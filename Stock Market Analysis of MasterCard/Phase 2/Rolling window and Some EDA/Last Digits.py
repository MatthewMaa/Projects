# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:17:26 2019

@author: Yazhuo Ma
"""


from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd


data = pd.read_csv('MA.csv')




def get_last_digit(y):
        x = str(round(float(y),2))
        x_list = x.split('.')
        fraction_str = x_list[1]
        if len(fraction_str)==1:
            return 0
        else:
            return int(fraction_str[1])




def get_every_year(year):
        
        df = data.loc[data.loc[:,'Year'] == year,]
        
        df['last digit'] = df['Open'].apply(get_last_digit)
        
        df['count'] = 1
        total = len(df)
        
        df_1 = df.groupby(['last digit'])['count'].sum()
        df_2 = df_1.to_frame()
        df_2.reset_index(level=0, inplace=True)
        df_2['digit_frequency'] = df_2['count']/total
        df_2['uniform'] = 0.1
        
        
        
        actual = df_2.loc[:,'digit_frequency'].values
        predicted = df_2.loc[:,'uniform'].values
        
        #calculate most frequent digit
        mfd = df_2[df_2['digit_frequency']== max(df_2['digit_frequency'])]['last digit'].values[0]
        
        #calculate least frequent digit
        lfd = df_2[df_2['digit_frequency']== min(df_2['digit_frequency'])]['last digit'].values[0]
        
        #calculate max absolute error
        maxae =  round(np.max(np.abs(actual - predicted)),2)
        
        #calculate mean absolute error
        mae = round(np.mean(np.abs(actual - predicted)),2)
        
        #calculate median absolute error
        mdae = round(np.median(np.abs(actual - predicted)),2)
        
        #calculate root mean squared error
        rmse = round(np.sqrt(np.mean(np.square(actual - predicted))),2)
        
        year_sum = [mfd,lfd,maxae,mae,mdae,rmse]
        
        return year_sum
    

data_list = [get_every_year(2014),get_every_year(2015),get_every_year(2016),get_every_year(2017),get_every_year(2018)]

df_3 = pd.DataFrame(data_list, columns = ['MFD','LFD','MAXAE','MAE','MDAE','RMSE'])
df_4 = df_3.set_index(pd.Index([2014,2015,2016,2017,2018]))

print('\n')
print('The table:\n')
print(df_4,'\n')
print('From the table we could find out that the most frequent digital is 0 for all five years, which means a lot of opening prices only have one decimal point or use 0 as their last digit.')