# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:32:53 2019

@author: Matthew
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



data = pd.read_csv('MA.csv')

# 2017 training data
df = data.loc[data.loc[:,'Year'] == 2017,]

df_2017 =   df.groupby(
               ['Week_Number', 'Label']
            ).agg(
                {
                     'Return':['mean','std']   
                }
            ).reset_index()

df_2017.columns = ['Week_Number', 'Label','mu','sd']

X_2017 = df_2017[['mu','sd']].values
Y_2017 = df_2017['Label'].values

# 2018 validation data
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 =   df.groupby(
               ['Week_Number', 'Label']
            ).agg(
                {
                     'Return':['mean','std']   
                }
            ).reset_index()

df_2018.columns = ['Week_Number', 'Label','mu','sd']

df_2018 = df_2018.dropna()
df_2018['Week_Number'] = range(1,len(df_2018['Week_Number'])+1)

X_2018 = df_2018[['mu','sd']].values
Y_2018 = df_2018['Label'].values

NB_classifier = GaussianNB().fit(X_2017,Y_2017)
pred = NB_classifier.predict(X_2018)
acc = NB_classifier.score(X_2018, Y_2018)

# Question 1

print()
print('Question 1')
print('Accuracy of naive bayes model:',round(acc,2))



# Question 2

matrix = confusion_matrix(Y_2018,pred, labels = ['R','G'])

print()
print('Question 2')
print('Confusion Matrix:')
print(np.matrix(matrix))

#Question 3&4

recall = matrix[1][1]/(matrix[1][1] + matrix[1][0])
specificity = matrix[0][0]/(matrix[0][0] + matrix[0][1])

print()
print('Question 3&4')
print('Recall:',round(recall,2))
print('specificity:',round(specificity,2))


# Question 5

# 2018

# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018_balance = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2018_balance = df_2018_balance.dropna()

df_2018_balance['Week_Number'] = range(1,len(df_2018_balance['Week_Number'])+1)


df_2018_balance = df_2018_balance.iloc[0:-1,]


df_tl = df_2018_balance.copy()

pred_list = list(pred)

df_tl['Label'] = pred_list

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

df_tl_answer = Balance(df_tl)
tl_values = df_tl_answer['Balance']

pl = tl_values-100

print()
print('Question 5:')
print('From the figure below we can know the True Lable profit/loss based on Naive Bayes Model.')

plt.figure(figsize = (10, 7))
point_x = df_2018_balance['Week_Number']

plt.plot(point_x, pl,label='True Label')

plt.legend(loc='upper left')
plt.title ('Profit/Loss(Naive Bayes) VS Week Number')
plt.xlabel ('Profit/Loss')


plt.show()