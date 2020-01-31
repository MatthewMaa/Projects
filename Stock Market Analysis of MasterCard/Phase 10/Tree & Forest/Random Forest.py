# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:21:21 2019

@author: Matthew
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



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

class_labels_dict = {'G': 1, 'R': 0}
label_color_dict = {1: 'G', 0: 'R'}
df_2017['class_labels']= df_2017['Label'].map(class_labels_dict)


# Question 1
print()
print('Question 1\n')
score_list = []

for i in range(1,11):
    for j in range(1,6):
               
        model = RandomForestClassifier(n_estimators=i, max_depth =j,random_state = 1,criterion ='entropy')
        model.fit(X_2017,Y_2017)
        score_list.append([i,j,(1-model.score(X_2018,Y_2018))])

error_rate_list = [i[2] for i in score_list]   

fig = plt.figure(figsize = (10, 7))
ax1 = fig.add_subplot(111)
ax1.plot(error_rate_list)
plt.title ('Error rate VS Index')
plt.xlabel ('Index')
plt.ylabel ('Error rate')

plt.show()

error_min = min(error_rate_list)
index_min = error_rate_list.index(error_min)

print(f'From the plot we can know that error rate is lowest({round(error_min,2)}) when N = {score_list[index_min][0]} and d = {score_list[index_min][1]}')

# Question 2
print()
print('Question 2\n')

n = score_list[index_min][0]
d = score_list[index_min][1]

model = RandomForestClassifier(n_estimators=n, max_depth = d,random_state = 1,criterion ='entropy')
model.fit(X_2017,Y_2017)

pred = model.predict(X_2018)

matrix = confusion_matrix(Y_2018,pred, labels = ['R','G'])

print('Confusion Matrix:')
print(np.matrix(matrix))

#Question 3&4

recall = matrix[1][1]/(matrix[1][1] + matrix[1][0])
specificity = matrix[0][0]/(matrix[0][0] + matrix[0][1])


print()
print('Question 3&4\n')
print('Recall:',round(recall,2))
print('specificity:',round(specificity,2))


# Question 5

print()
print('Question 5\n')

# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2018 = df_2018.dropna()

df_2018['Week_Number'] = range(1,len(df_2018['Week_Number'])+1)


df_2018 = df_2018.loc[0:len(df_2018)-2,]

# Compute portfolio growth 

# buy and hold 
df_bnh = df_2018.copy()
share = 100/df_bnh['Adj Close'][0]
bnh_values = []
for i in range(len(df_bnh)):
    balance  = share * df_bnh['Adj Close'][i]
    bnh_values.append(balance)

#Decision Tree


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

df_forest = df_2018.copy()

df_forest['Label'] = pred

df_forest_answer = Balance(df_forest)
random_forest_values = df_forest_answer['Balance']



# plot all lines

plt.figure(figsize = (10, 7))
point_x = df_2018['Week_Number']
plt.plot(point_x, bnh_values,label='Buy and hold: '+'sd = '+str(round(np.std(bnh_values),2))+','+' mu = '+str(round(np.mean(bnh_values),2)))
plt.plot(point_x, random_forest_values,label='Random Forest: '+'sd = '+str(round(np.std(random_forest_values),2))+','+' mu = '+str(round(np.mean(random_forest_values),2)))


plt.legend(loc='upper left')
plt.title ('Portfolio Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel ('Portfolio Values')


plt.show()

print("From the graph on the top we can know that Random Froest strategy gives me the highest final portfolio value.")