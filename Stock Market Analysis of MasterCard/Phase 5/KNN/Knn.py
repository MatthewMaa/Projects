# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:25:52 2019

@author: Matthew
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import warnings
warnings.simplefilter('ignore')

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

X = df_2017 [[ 'mu', 'sd']].values
Y = df_2017 [[ 'Label']].values

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
X_train = X
X_test = X
Y_train = Y
Y_test = Y



# Question 1

error_rate = []

for k in range(3,12,2):
    knn_classifier = KNeighborsClassifier ( n_neighbors =k)
    knn_classifier.fit( X_train , Y_train.ravel())
    pred_k = knn_classifier.predict ( X_test )
    error_rate.append(np. mean ( pred_k != Y_test ))

print('')
print('Question 1: ')
print('See figure below!')

a = plt.figure(figsize = (10, 7))
plt.plot ( range (3 ,12 ,2) , error_rate , color ='red', linestyle ='dashed',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Error Rate vs. k for 2017')
plt.xlabel ('number of neighbors : k')
plt.ylabel ('Error Rate')
a.show()

#Question 2

k_array = np.asarray(range(3,12,2))
optimal = k_array[np.where(error_rate == min(error_rate))]
print('')
print('Question 2: ')
print('The optimal value of k for 2017 is:',*optimal)


# Question 3

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

X_2018 = df_2018 [[ 'mu', 'sd']].values
Y_2018 = df_2018 [[ 'Label']].values



X_test = X_2018
Y_test = Y_2018


error_rate = []

for k in optimal:
    knn_classifier = KNeighborsClassifier ( n_neighbors = k)
    knn_classifier.fit( X_train , Y_train.ravel())
    pred_k = knn_classifier.predict ( X_test )
    error_rate.append(np.mean ( pred_k != Y_test ))
    
accuracy = [1 - x for x in error_rate]
print('')
print('Question 3: ')
print('The accuracy is:',np.round(accuracy,2),'for k = ',optimal)

# Question 4

# for k = 9

knn_classifier = KNeighborsClassifier ( n_neighbors = 9)
knn_classifier.fit( X_train , Y_train.ravel())
pred_k = knn_classifier.predict ( X_test )

cm_9 = confusion_matrix(Y_test,pred_k)

# for k = 11

knn_classifier = KNeighborsClassifier ( n_neighbors = 11)
knn_classifier.fit( X_train , Y_train.ravel())
pred_k = knn_classifier.predict ( X_test )

cm_11 = confusion_matrix(Y_test,pred_k)

print('')
print('Question 4: ')
print('The confusion matrix for 2018 when k = 9 :')
print(cm_9)
print('The confusion matrix for 2018 when k = 11 :')
print(cm_11)


#QUestion 5

le = LabelEncoder ()
Y_test = le.fit_transform(Y_test)
pred_k = le.fit_transform(pred_k)

recall = recall_score(Y_test,pred_k)

total = sum(sum(cm_9))

specificity = cm_9[0][0]/(cm_9[0][0] + cm_9[0][1])

print('')
print('Question 5: ')
print('True positive rate (sensitivity or recall) for 2018:',recall)
print('True negative rate (specificity) for 2018:',specificity)
