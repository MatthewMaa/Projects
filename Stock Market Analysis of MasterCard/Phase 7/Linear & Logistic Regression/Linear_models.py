# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:44:38 2019

@author: Matthew
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



# Construct data frame
data = pd.read_csv('MA.csv')

# 2017
df = data.loc[data.loc[:,'Year'] == 2017,]

df_2017 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2017 = df_2017.dropna()













#Question 1
W = [i for i in range(5,13)]
d = [1,2,3]
def accuracy(W,d):
    
    accuracy_list = []
    
    for w in W:
        
        pred = []
        
        for i in range(0,len(df_2017)-w):
            X = df_2017['Adj Close'][i:i+w]
            y = df_2017['Adj Close'][i:i+w]
            weights = np.polyfit(X,y,d)
            model = np.poly1d(weights)
            if model(df_2017['Adj Close'][i+w]) > df_2017['Adj Close'][i+w-1]:
                pred.append('G')
            elif model(df_2017['Adj Close'][i+w]) < df_2017['Adj Close'][i+w-1]:
                pred.append('R')
            elif model(df_2017['Adj Close'][i+w]) == df_2017['Adj Close'][i+w-1]:
                pred.append(df_2017['Label'][i+w-1]) 
        
        answer = np.mean(pred == df_2017['Label'][w:])
        accuracy_list.append(answer)
    
    return accuracy_list
        

# d = 1,2,3
d1_values = accuracy(W,1)
d2_values = accuracy(W,2)
d3_values = accuracy(W,3)


print()
print('Question 1:')
print('From the  figure below we can know that my optimal W = 5, no matter d = 1, 2 or 3.')
plt.figure(figsize = (10, 7))
point_x = W
plt.plot(point_x, d1_values,label='d = 1')
plt.plot(point_x, d2_values,label='d = 2')
plt.plot(point_x, d3_values,label='d = 3')

plt.legend(loc='upper right')
plt.title ('Accuracy vs  W')
plt.xlabel ('W')
plt.ylabel ('Accuracy')


plt.show()

# Question 2


# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2018 = df_2018.dropna()

df_2018['Week_Number'] = range(1,len(df_2018['Week_Number'])+1)

# pick 52 weeks
df_2018 = df_2018.loc[0:len(df_2018)-2,]

# W = 5 Rebuild training data

adj_close = list(df_2017['Adj Close'][-5:]) + list(df_2018['Adj Close'])
label  = list(df_2017['Label'][-5:]) + list(df_2018['Label'])


def accu(d):
    
    w = 5
    pred = []
            
    for i in range(0,len(adj_close)-w):
        X = adj_close[i:i+w]
        y = adj_close[i:i+w]
        weights = np.polyfit(X,y,d)
        model = np.poly1d(weights)
        if model(adj_close[i+w]) > adj_close[i+w-1]:
            pred.append('G')
        elif model(adj_close[i+w]) < adj_close[i+w-1]:
            pred.append('R')
        elif model(adj_close[i+w]) == adj_close[i+w-1]:
            pred.append(label[i+w-1]) 
    
    accu1 = np.mean(pred == df_2018['Label'])
    return accu1

print()
print('Question 2:')
print('While W = 5 and d = 1, the accuracy is',round(accu(1),2))
print('While W = 5 and d = 2, the accuracy is',round(accu(2),2))    
print('While W = 5 and d = 3, the accuracy is',round(accu(3),2))    


# Question 3

def cm(d):
    
    w = 5
    pred = []
            
    for i in range(0,len(adj_close)-w):
        X = adj_close[i:i+w]
        y = adj_close[i:i+w]
        weights = np.polyfit(X,y,d)
        model = np.poly1d(weights)
        if model(adj_close[i+w]) > adj_close[i+w-1]:
            pred.append('G')
        elif model(adj_close[i+w]) < adj_close[i+w-1]:
            pred.append('R')
        elif model(adj_close[i+w]) == adj_close[i+w-1]:
            pred.append(label[i+w-1]) 
    
    matrix = confusion_matrix(label[w:],pred,labels = ['R','G']) 
    return matrix

print()
print('Question 3:')
print('While W = 5 and d = 1, the confusion matirx is:\n',cm(1))
print('While W = 5 and d = 2, the confusion matirx is:\n',cm(2))    
print('While W = 5 and d = 3, the confusion matirx is:\n',cm(3)) 


# Question 4

# to get predicted label
def predi(d):
    
    w = 5
    pred = []
            
    for i in range(0,len(adj_close)-w):
        X = adj_close[i:i+w]
        y = adj_close[i:i+w]
        weights = np.polyfit(X,y,d)
        model = np.poly1d(weights)
        if model(adj_close[i+w]) > adj_close[i+w-1]:
            pred.append('G')
        elif model(adj_close[i+w]) < adj_close[i+w-1]:
            pred.append('R')
        elif model(adj_close[i+w]) == adj_close[i+w-1]:
            pred.append(label[i+w-1]) 
    
    return pred

#true label


def Balance(df):
        
        balance = 100
        position = 'no'
        shares = 0
        balance_list = []
        for i in range(len(df['Week_Number'])-1):
            
            if df['Label'][i+1] == 'G':
                
                if position == 'no':
                    shares = balance/df['Adj Close'][i]
                    position = 'yes'
                    balance = shares*df['Adj Close'][i]
                    balance_list.append(balance)
                    
                elif position == 'yes':
                    
                    balance_list.append(shares*df['Adj Close'][i])
                    
            elif df['Label'][i+1] == 'R':
                    
                if position == 'no':
                    balance_list.append(balance)
                    
                elif position == 'yes':
                    balance = shares*df['Adj Close'][i]
                    position = 'no'
                    balance_list.append(balance)
                    shares = 0
        
        # to decide the last week
        balance_list.append(balance_list[len(balance_list)-1])
                    
        weeknum_list = list(df['Week_Number'])
        label_list = list(df['Label'])
        df1 = pd.DataFrame(list(zip(weeknum_list,label_list,balance_list)), 
               columns =['Week_Number','Label', 'Balance']) 
        return df1


df_tl = df_2018.iloc[:]

# d = 1
df_tl['Label'] = predi(1)
df_tl_answer = Balance(df_tl)
tl_values1 = df_tl_answer['Balance']

# d = 2
df_tl['Label'] = predi(2)
df_tl_answer = Balance(df_tl)
tl_values2 = df_tl_answer['Balance']

# d = 3
df_tl['Label'] = predi(3)
df_tl_answer = Balance(df_tl)
tl_values3 = df_tl_answer['Balance']

print()
print('Question 4:')
print('From the figure below we can know that True Lable strategy has the same result for d = 1, 2 or 3')

plt.figure(figsize = (10, 7))
point_x = df_2018['Week_Number']
plt.plot(point_x, tl_values1,label='d = 1')
plt.plot(point_x, tl_values2,label='d = 2')
plt.plot(point_x, tl_values2,label='d = 3')

plt.legend(loc='upper left')
plt.title ('Portfolio Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel ('Portfolio Values')


plt.show()