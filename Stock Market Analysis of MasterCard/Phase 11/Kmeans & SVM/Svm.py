# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:35:07 2019

@author: Matthew
1. implement a linear SVM. What is the accuracy of your SVM
for year 2?
2. compute the confusion matrix for year 2
3. what is true positive rate and true negative rate for year 2?
4. implement a Gaussian SVM and compute its accuracy for
year 2? Is it better than linear SVM (use default values for
parameters)
5. implement polynomial SVM for degree 2 and compute its
accuracy? Is it better than linear SVM?
6. implement a trading strategy based on your labels (from
linear SVM) for year 2 and compare the performance with
the ”buy-and-hold” strategy. Which strategy results in a
larger amount at the end of the year?

"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
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
scaler = StandardScaler()
scaler.fit(X_2017)
X_2017 = scaler.transform (X_2017)
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
scaler.fit(X_2018)
X_2018 = scaler.transform (X_2018)
Y_2018 = df_2018['Label'].values


# Question 1

svm_classifier = svm.SVC(kernel ='linear')
svm_classifier.fit (X_2017,Y_2017)
pred = svm_classifier.predict(X_2018)
accuracy_1 = svm_classifier.score (X_2018, Y_2018)
print()
print('Question 1\n')
print(f'Accuracy of linear SVM: {round(accuracy_1,2)}')

# Q2

matrix = confusion_matrix(Y_2018,pred, labels = ['R','G'])
print()
print('Question 2\n')
print('Confusion Matrix:')
print(np.matrix(matrix))


# Q3

recall = matrix[1][1]/(matrix[1][1] + matrix[1][0])
specificity = matrix[0][0]/(matrix[0][0] + matrix[0][1])


print()
print('Question 3\n')
print('Recall:',round(recall,2))
print('specificity:',round(specificity,2))

# Q4

svm_classifier = svm.SVC(kernel ='rbf')
svm_classifier.fit (X_2017,Y_2017)

accuracy_2 = svm_classifier.score (X_2018, Y_2018)
print()
print('Question 4\n')
print(f'Accuracy of Gaussian SVM: {round(accuracy_2,2)}')

# Q5

svm_classifier = svm.SVC(kernel ='poly', degree = 2)
svm_classifier.fit (X_2017,Y_2017)

accuracy_3 = svm_classifier.score (X_2018, Y_2018)
print()
print('Question 5\n')
print(f'Accuracy of polynomial SVM for degree 2: {round(accuracy_3,2)}')
print('The acuracies of Gaussian, Linear and Polynomial(d = 2) SVM are equal to each other.')

#6

print()
print('Question 6\n')


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

#linear SVM


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

df_svm = df_2018.copy()

df_svm['Label'] = pred

df_svm_answer = Balance(df_svm)
svm_values = df_svm_answer['Balance']



# plot all lines

plt.figure(figsize = (10, 7))
point_x = df_2018['Week_Number']
plt.plot(point_x, bnh_values,label='Buy and hold: '+'sd = '+str(round(np.std(bnh_values),2))+','+' mu = '+str(round(np.mean(bnh_values),2)))
plt.plot(point_x, svm_values,label='Linear SVM: '+'sd = '+str(round(np.std(svm_values),2))+','+' mu = '+str(round(np.mean(svm_values),2)))


plt.legend(loc='upper left')
plt.title ('Portfolio Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel ('Portfolio Values')


plt.show()

print("From the graph on the top we can know that Buy and Hold ives me the highest final portfolio value.")