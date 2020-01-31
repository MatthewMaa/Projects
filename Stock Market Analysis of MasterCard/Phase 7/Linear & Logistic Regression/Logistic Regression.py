# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:59:31 2019

@author: Matthew
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,classification_report
data = pd.read_csv('MA.csv')

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

X_2017 = df_2017 [[ 'mu', 'sd']].values
Y_2017 = df_2017 [ 'Label'].apply( lambda x: 1 if x== 'G' else 0)
Y_2017 = Y_2017.values

scaler = StandardScaler().fit (X_2017)
X_2017 = scaler.transform(X_2017)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit (X_2017,Y_2017)

#Question 1
print()
print('Question 1:')

equation = 'y = ' + str(log_reg_classifier.intercept_[0]) +' + ('+ str(log_reg_classifier.coef_[0][0])+')x1'+' + ('+ str(log_reg_classifier.coef_[0][1])+')x2'
print(equation)

#Question 2
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

df_2018 = df_2018.dropna()
df_2018['Week_Number'] = range(1,len(df_2018['Week_Number'])+1)

X_2018 = df_2018 [[ 'mu', 'sd']].values
Y_2018 = df_2018 [ 'Label'].apply( lambda x: 1 if x== 'G' else 0)
Y_2018 = Y_2018.values

scaler = StandardScaler().fit (X_2018)
X_2018 = scaler.transform(X_2018)

accuracy = log_reg_classifier.score (X_2018, Y_2018)
print()
print('Question 2:')
print('Accuracy for Year 2:',round(accuracy,2))

#Question 3
pred = log_reg_classifier.predict(X_2018)
matrix = confusion_matrix(Y_2018,pred) 

print()
print('Question 3:')
print('Matrix for year2:')
print(np.matrix(matrix))

#Question 4

recall = matrix[1][1]/(matrix[1][1] + matrix[1][0])
specificity = matrix[0][0]/(matrix[0][0] + matrix[0][1])

print()
print('Question 4:')
print('Recall:',round(recall,2))
print('specificity:',round(specificity,2))

#Question 5

# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018_balance = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2018_balance = df_2018_balance.dropna()

df_2018_balance['Week_Number'] = range(1,len(df_2018_balance['Week_Number'])+1)


df_2018_balance = df_2018_balance.loc[0:len(df_2018)-1,]

# buy and hold 
df_bnh = df_2018_balance
share = 100/df_bnh['Adj Close'][0]
bnh_values = []
for i in range(len(df_bnh)):
    balance  = share * df_bnh['Adj Close'][i]
    bnh_values.append(balance)

# True label
    
#true label

df_tl = df_2018_balance

pred_list = list(pred)
pred_y = np.array(['G' if v == 1 else 'R' for v in pred_list])
df_tl['Label'] = pred_y

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

print()
print('Question 5:')
print('From the figure below we can know that True Lable strategy has a larger amount at the end of the year.')

plt.figure(figsize = (10, 7))
point_x = df_2018_balance['Week_Number']
plt.plot(point_x, bnh_values,label='Buy and hold: '+'sd = '+str(round(np.std(bnh_values),2))+','+' mu = '+str(round(np.mean(bnh_values),2)))
plt.plot(point_x, tl_values,label='True Label: '+'sd = '+str(round(np.std(tl_values),2))+','+' mu = '+str(round(np.mean(tl_values),2)))

plt.legend(loc='upper left')
plt.title ('Portfolio Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel ('Portfolio Values')


plt.show()