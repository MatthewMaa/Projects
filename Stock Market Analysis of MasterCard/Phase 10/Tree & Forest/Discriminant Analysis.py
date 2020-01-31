# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:36:03 2019

@author: Matthew
1. what is the equation for linear and quadratic classifier found
from year 1 data?
2. what is the accuracy for year 2 for each classifier. Which
classifier is ”better”?
3. compute the confusion matrix for year 2 for each classifier
4. what is true positive rate (sensitivity or recall) and true
negative rate (specificity) for year 2?
5. implement trading strategyies based on your labels for year
2 (for both linear and quadratic) and compare the performance with the ”buy-and-hold” strategy. Which strategy
results in a larger amount at the end of the year?
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

scaler = StandardScaler()
scaler.fit(X_2017)
X_2017 = scaler.transform (X_2017)
scaler.fit(X_2018)
X_2018 = scaler.transform (X_2018)


# Question 1

lda_classifier = LDA(n_components =2)
lda_classifier.fit (X_2017,Y_2017)

print()
print('Question 1\n')
print(f'Equation for linear calssifier: ({lda_classifier.coef_[0][0]})x1 + ({lda_classifier.coef_[0][1]})x2 + ({lda_classifier.intercept_[0]}) = 0')

qda_classifier = QDA()
qda_classifier.fit (X_2017,Y_2017)

# Question 2
lda_accuracy = lda_classifier.score (X_2017, Y_2018)
qda_accuracy = qda_classifier.score (X_2017, Y_2018)
print()
print('Question 2\n')
print(f'Accuracy for lda and qda: {round(lda_accuracy,2)},{round(qda_accuracy,2)}')
print('So QDA is better.')

# Question 3
lda_pred = lda_classifier.predict(X_2018)
qda_pred = qda_classifier.predict(X_2018)


lda_matrix = confusion_matrix(Y_2018,lda_pred, labels = ['R','G'])
qda_matrix = confusion_matrix(Y_2018,qda_pred, labels = ['R','G'])

print()
print('Question 3\n')
print('LDA Confusion Matrix:')
print(np.matrix(lda_matrix))
print('QDA Confusion Matrix:')
print(np.matrix(qda_matrix))

#Question 4

lda_recall = lda_matrix[1][1]/(lda_matrix[1][1] + lda_matrix[1][0])
lda_specificity = lda_matrix[0][0]/(lda_matrix[0][0] + lda_matrix[0][1])
qda_recall = qda_matrix[1][1]/(qda_matrix[1][1] + qda_matrix[1][0])
qda_specificity = qda_matrix[0][0]/(qda_matrix[0][0] + qda_matrix[0][1])

print()
print('Question 4\n')
print('LDA Recall:',round(lda_recall,2))
print('LDA specificity:',round(lda_specificity,2))
print('QDA Recall:',round(qda_recall,2))
print('QDA specificity:',round(qda_specificity,2))


#Question 5
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

#LDA


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

df_lda = df_2018.copy()

df_lda['Label'] = lda_pred

df_lda_answer = Balance(df_lda)
lda_values = df_lda_answer['Balance']

#QDA

df_qda = df_2018.copy()

df_qda['Label'] = qda_pred

df_qda_answer = Balance(df_qda)
qda_values = df_qda_answer['Balance']


# plot all lines

plt.figure(figsize = (10, 7))
point_x = df_2018['Week_Number']
plt.plot(point_x, bnh_values,label='Buy and hold: '+'sd = '+str(round(np.std(bnh_values),2))+','+' mu = '+str(round(np.mean(bnh_values),2)))
plt.plot(point_x, lda_values,label='LDA: '+'sd = '+str(round(np.std(lda_values),2))+','+' mu = '+str(round(np.mean(lda_values),2)))
plt.plot(point_x, qda_values,label="QDA: "+'sd = '+str(round(np.std(qda_values),2))+','+' mu = '+str(round(np.mean(qda_values),2)))

plt.legend(loc='upper left')
plt.title ('Portfolio Values VS Week Number')
plt.xlabel ('Week Number')
plt.ylabel ('Portfolio Values')


plt.show()

print("From the graph on the top we can know that QDA strategy gives me the highest final portfolio value.")