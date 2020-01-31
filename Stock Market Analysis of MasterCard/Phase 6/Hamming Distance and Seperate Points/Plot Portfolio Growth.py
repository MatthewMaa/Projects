# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:45:24 2019

@author: Matthew
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('MA.csv')






# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2018 = df_2018.dropna()

df_2018['Week_Number'] = range(1,len(df_2018['Week_Number'])+1)


df_2018 = df_2018.loc[0:len(df_2018)-2,]

# Compute portfolio growth 

# buy and hold 
df_bnh = df_2018
share = 100/df_bnh['Adj Close'][0]
bnh_values = []
for i in range(len(df_bnh)):
    balance  = share * df_bnh['Adj Close'][i]
    bnh_values.append(balance)

#true label

df_tl = df_2018

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

#Knn p = 1, k =9
# use 2017 to build model 
df = data.loc[data.loc[:,'Year'] == 2017,]

df_2017_knn =   df.groupby(
               ['Week_Number', 'Label']
            ).agg(
                {
                     'Return':['mean','std']   
                }
            ).reset_index()

df_2017_knn.columns = ['Week_Number', 'Label','mu','sd']
df_2017_knn = df_2017_knn.dropna()

X = df_2017_knn [[ 'mu', 'sd']].values
Y = df_2017_knn [ 'Label'].values

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
X_train = X
Y_train = Y


knn_classifier = KNeighborsClassifier ( n_neighbors = 9, p = 1)
knn_classifier.fit( X_train , Y_train.ravel())


# predict 2018 label

df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018_knn =   df.groupby(
               ['Week_Number', 'Label']
            ).agg(
                {
                     'Return':['mean','std']   
                }
            ).reset_index()

df_2018_knn.columns = ['Week_Number', 'Label','mu','sd']

df_2018_knn = df_2018_knn.dropna()

df_2018_knn['Week_Number'] = range(1,len(df_2018_knn['Week_Number'])+1)


X_2018 = df_2018_knn [[ 'mu', 'sd']].values
Y_2018 = df_2018_knn [ 'Label'].values



X_test = X_2018
Y_test = Y_2018

pred_k = knn_classifier.predict (X_test)

# replace old label and calculate values

df_knn1 = df_2018

df_knn1['Label'] = pred_k

df_knn_answer1 = Balance(df_knn1)
knn_values1 = df_knn_answer1['Balance']

#Knn p = 1.5, k =9

knn_classifier = KNeighborsClassifier ( n_neighbors = 9, p = 1.5)
knn_classifier.fit( X_train , Y_train.ravel())
pred_k = knn_classifier.predict (X_test)

# replace old label and calculate values

df_knn2 = df_2018

df_knn2['Label'] = pred_k

df_knn_answer2 = Balance(df_knn2)
knn_values2 = df_knn_answer2['Balance']

#Knn p = 2, k =9

knn_classifier = KNeighborsClassifier ( n_neighbors = 9, p = 2)
knn_classifier.fit( X_train , Y_train.ravel())
pred_k = knn_classifier.predict (X_test)

# replace old label and calculate values

df_knn3 = df_2018

df_knn3['Label'] = pred_k

df_knn_answer3 = Balance(df_knn2)
knn_values3 = df_knn_answer3['Balance']


# plot all lines

plt.figure(figsize = (10, 7))
point_x = df_2018['Week_Number']
plt.plot(point_x, bnh_values,label='Buy and hold: '+'sd = '+str(round(np.std(bnh_values),2))+','+' mu = '+str(round(np.mean(bnh_values),2)))
plt.plot(point_x, tl_values,label='True Label: '+'sd = '+str(round(np.std(tl_values),2))+','+' mu = '+str(round(np.mean(tl_values),2)))
plt.plot(point_x, knn_values1,label="Knn(p=1): "+'sd = '+str(round(np.std(knn_values1),2))+','+' mu = '+str(round(np.mean(knn_values1),2)))
plt.plot(point_x, knn_values2,label="Knn(p=1.5): "+'sd = '+str(round(np.std(knn_values2),2))+','+' mu = '+str(round(np.mean(knn_values2),2)))
plt.plot(point_x, knn_values3,label="Knn(p=2): "+'sd = '+str(round(np.std(knn_values3),2))+','+' mu = '+str(round(np.mean(knn_values3),2)))

plt.legend(loc='upper left')
plt.title ('Portfolio Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel ('Portfolio Values')


plt.show()

print("From the graph on the top we can know that True Label strategy gives me the highest final portfolio value.")