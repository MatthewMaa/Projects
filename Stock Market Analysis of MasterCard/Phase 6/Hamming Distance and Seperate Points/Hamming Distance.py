# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:26:52 2019

@author: Matthew
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('MA.csv')






# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


df_2018 = df_2018.dropna()

df_2018['Week_Number'] = range(1,len(df_2018['Week_Number'])+1)


df_2018 = df_2018.loc[0:len(df_2018)-2,]

# Compute hamming values(1 and 0) for each strategy

# buy and hold 
df_bnh = df_2018
share = 100/df_bnh['Adj Close'][0]
bnh_values = []
for i in range(len(df_bnh)):
    value = 1
    bnh_values.append(value)

#true label

df_tl = df_2018
a = list(df_tl['Label']=='G')

tl_values = list(map(int, a))


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

pred_k1 = knn_classifier.predict (X_test)

# Calculate hamming values
a = list(pred_k1=='G')

knn_values1 = list(map(int, a))



#Knn p = 1.5, k =9

knn_classifier = KNeighborsClassifier ( n_neighbors = 9, p = 1.5)
knn_classifier.fit( X_train , Y_train.ravel())
pred_k2 = knn_classifier.predict (X_test)


# Calculate hamming values
a = list(pred_k2=='G')

knn_values2 = list(map(int, a))


#Knn p = 2, k =9

knn_classifier = KNeighborsClassifier ( n_neighbors = 9, p = 2)
knn_classifier.fit( X_train , Y_train.ravel())
pred_k3 = knn_classifier.predict (X_test)


# Calculate hamming values
a = list(pred_k3=='G')

knn_values3 = list(map(int, a))

# function to calculate hamming distance
def Hamming(x,y):
    distance = 0
    if len(x) == len(y):
        for i in range(len(x)):
            if x[i] != y[i]:
                distance += 1
            else:
                pass
    else:
        return('length error')
    return distance

                

    

# Question 1
print()
print('Question 1 :')
plt.figure(figsize = (10, 7))
point_x = df_2018['Week_Number']
plt.plot(point_x, bnh_values,label='Buy and Hold ' + 'Distance to True Label: ' + str(Hamming(bnh_values,tl_values)))
plt.plot(point_x, tl_values,label='True Label '+'Distance to True Label: ' + str(Hamming(tl_values,tl_values)))
plt.plot(point_x, knn_values1,label="Knn(p=1) "+'Distance to True Label: ' + str(Hamming(knn_values1,tl_values)))
plt.plot(point_x, knn_values2,label="Knn(p=1.5) "+'Distance to True Label: ' + str(Hamming(knn_values2,tl_values)))
plt.plot(point_x, knn_values3,label="Knn(p=2) "+'Distance to True Label: ' + str(Hamming(knn_values3,tl_values)))

plt.legend(loc='upper left')
plt.title ('Trajectory Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel (' Trajectory Values')
plt.ylim(0,2)

plt.show()

#Question 2
A = [[Hamming(bnh_values,bnh_values), Hamming(bnh_values,tl_values), Hamming(bnh_values,knn_values1),Hamming(bnh_values,knn_values2),Hamming(bnh_values,knn_values3)], 
     [Hamming(tl_values,bnh_values), Hamming(tl_values,tl_values), Hamming(tl_values,knn_values1),Hamming(tl_values,knn_values2),Hamming(tl_values,knn_values3)],
     [Hamming(knn_values1,bnh_values), Hamming(knn_values1,tl_values), Hamming(knn_values1,knn_values1), Hamming(knn_values1,knn_values2),Hamming(knn_values1,knn_values3)],
     [Hamming(knn_values2,bnh_values), Hamming(knn_values2,tl_values), Hamming(knn_values2,knn_values1), Hamming(knn_values2,knn_values2),Hamming(knn_values2,knn_values3)],
     [Hamming(knn_values3,bnh_values), Hamming(knn_values3,tl_values), Hamming(knn_values3,knn_values1), Hamming(knn_values3,knn_values2),Hamming(knn_values3,knn_values3)]]

Y_test = df_2018['Label'].apply( lambda x: 1 if x== 'G' else 0)
Y_test = Y_test.values

pred_bnh = [1 for i in Y_test]
cm_1 = confusion_matrix(Y_test,pred_bnh)
cm_2 = confusion_matrix(Y_test,tl_values)
cm_3 = confusion_matrix(Y_test,knn_values1)
cm_4 = confusion_matrix(Y_test,knn_values2)
cm_5 = confusion_matrix(Y_test,knn_values3)
print()
print('Question 2 :')
print()
print(np.matrix(A))
print()
print(np.matrix(cm_1))
print()
print(np.matrix(cm_2))
print()
print(np.matrix(cm_3))
print()
print(np.matrix(cm_4))
print()
print(np.matrix(cm_5))
print()
print("Conclusion: Because of too many 0 in my matrix, it makes me hard to find relationship between these matrices."+
      'However, it seems that FPs(15,0,15,15,15) and TN(0,15,0,0,0) from the cunfusion matrices could build this 5x5 matrix either horizontally or vertically.')
