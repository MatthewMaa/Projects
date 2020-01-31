# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:35:52 2019

@author: Yazhuo Ma
"""
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MA.csv')




def pltcolor(x):
    colors=[]
    for i in range(len(x)):
        if x[i] == 'R':
            colors.append('red')
        elif x[i] == 'G':
            colors.append('green')
        else:
            colors.append('grey')
    return colors


# 2017
df = data.loc[data.loc[:,'Year'] == 2017,]

df_2017 =   df.groupby(
               ['Week_Number', 'Label']
            ).agg(
                {
                     'Return':['mean','std']   
                }
            ).reset_index()

df_2017.columns = ['Week_Number', 'Label','mu','sd']

a = plt.figure(1)
c = pltcolor(df_2017['Label'])

plt.scatter(df_2017['mu'], df_2017['sd'],c=c,s = 50,alpha=0.5)

for x,y,z in zip(df_2017['mu'],df_2017['sd'],df_2017['Week_Number']):


    plt.annotate(z, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center',
                 size = 5
                 ) # horizontal alignment can be left, right or center


plt.title('2017')
plt.xlabel('mu')
plt.ylabel('sigma')
a.show()

# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 =   df.groupby(
               ['Week_Number', 'Label']
            ).agg(
                {
                     'Return':['mean','std']   
                }
            ).reset_index()

df_2018.columns = ['Week_Number', 'Label','mu','sd']

b = plt.figure(2)
c = pltcolor(df_2018['Label'])

plt.scatter(df_2018['mu'], df_2018['sd'],c=c,s = 50,alpha=0.5)

for x,y,z in zip(df_2018['mu'],df_2018['sd'],df_2018['Week_Number']):


    plt.annotate(z, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center',
                 size = 5
                 ) # horizontal alignment can be left, right or center


plt.title('2018')
plt.xlabel('mu')
plt.ylabel('sigma')
b.show()
