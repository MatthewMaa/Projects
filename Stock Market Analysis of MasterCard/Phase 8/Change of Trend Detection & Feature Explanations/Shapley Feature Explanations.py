# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:17:14 2019

@author: Matthew
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder,PolynomialFeatures,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
# data preprocessing
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


#Question 1

#Logistic Regression


def logistic(columns):
    
    X_train = df_2017 [columns].values
    Y_train = df_2017 [ 'Label'].apply( lambda x: 1 if x== 'G' else 0)
    Y_train = Y_train.values
    
    scaler = StandardScaler().fit (X_train)
    X_train = scaler.transform(X_train)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit (X_train,Y_train)
    
    X_test = df_2018 [columns].values
    Y_test = df_2018 [ 'Label'].apply( lambda x: 1 if x== 'G' else 0)
    Y_test = Y_test.values
    
    scaler = StandardScaler().fit (X_test)
    X_test = scaler.transform(X_test)
    
    accuracy = log_reg_classifier.score (X_test, Y_test)
    
    return accuracy

    
log_acc_list = [logistic(['mu','sd']),logistic(['sd']),logistic(['mu'])]


# knn

def knn(columns):
    X_train = df_2017 [columns].values
    Y_train = df_2017 [ 'Label'].values#.apply( lambda x: 1 if x== 'G' else 0)
    #Y_train = Y_train.values
    
    scaler = StandardScaler().fit (X_train)
    X_train = scaler.transform(X_train)
    knn_classifier = KNeighborsClassifier ( n_neighbors = 9)
    knn_classifier.fit( X_train , Y_train)
    
    X_test = df_2018 [columns].values
    Y_test = df_2018 [ 'Label'].values#.apply( lambda x: 1 if x== 'G' else 0)
    #Y_test = Y_test.values
    
    scaler = StandardScaler().fit (X_test)
    X_test = scaler.transform(X_test)
    
    accuracy = knn_classifier.score (X_test, Y_test)
    
    return accuracy
    
knn_acc_list = [knn(['mu','sd']),knn(['sd']),knn(['mu'])]

#d=1 linear model

def linear(columns):
    
    d = 1
    X_train = df_2017 [columns].values
    poly = PolynomialFeatures(degree=d)
    X_train = poly.fit_transform(X_train)
    
    Y_train = df_2017 [ 'Label'].apply( lambda x: 1 if x== 'G' else 0)
    Y_train = Y_train.values
    
    
    poly.fit(X_train, Y_train)
    line = LinearRegression() 
    line.fit(X_train, Y_train)
    
    
    X_test = df_2018 [columns].values
    X_test = poly.fit_transform(X_test)
    Y_test = df_2018 [ 'Label'].apply( lambda x: 1 if x== 'G' else 0)
    Y_test = Y_test.values
    
    
    pred = line.predict(X_test)
    accuracy = np.mean(np.array([int(round(x)) for x in pred]) == Y_test)
    
    return accuracy

linear_acc_list = [linear(['mu','sd']),linear(['sd']),linear(['mu'])]

answer  = pd.DataFrame({'Logistic Regression':[round((log_acc_list[0]-log_acc_list[1]),2),round((log_acc_list[0]-log_acc_list[2]),2)],
                        'Knn':[round((knn_acc_list[0]-knn_acc_list[1]),2),round((knn_acc_list[0]-knn_acc_list[2]),2)],
                        'Linear Model':[round((linear_acc_list[0]-linear_acc_list[1]),2),round((linear_acc_list[0]-linear_acc_list[2]),2)]},
                        index = ['mu','sd'])

print()
print('Question 1:\n')
print(answer)
print()
print('From the table we can know that for different models there are different influences for reducing mu or sigma.')
print('For example, mu has a postive effect on Knn model whereas sigma has a negative effect on the same model.')


# Question 2

# preporcessing data
dataset= load_iris()
data=pd.DataFrame(dataset['data'],columns=['Sepal length','Sepal Width','Petal Length','Petal Width'])
data['Species']=dataset['target']
data['Species']=data['Species'].apply(lambda x: dataset['target_names'][x])
OH_encoder = OneHotEncoder (handle_unknown ='ignore',sparse = False)
encoded = pd.DataFrame(OH_encoder.fit_transform(data [['Species']]))
encoded = encoded.rename(columns = {0:'Setosa',1:'Versicolor',2:'Virginica'})

iris_data = pd.concat([data,encoded], axis =  1)

def log_iris(columns,species):
    
    X = iris_data [columns].values
    Y = iris_data [species].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5,random_state =1)
    
    
    scaler = StandardScaler().fit (X_train)
    X_train = scaler.transform(X_train)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit (X_train,Y_train)
    
    scaler = StandardScaler().fit (X_test)
    X_test = scaler.transform(X_test)
    
    accuracy = log_reg_classifier.score (X_test, Y_test)
    
    return accuracy
    

column_name = ['Sepal length','Sepal Width','Petal Length','Petal Width']
specie_name = ['Setosa','Versicolor','Virginica']

acc_list =[]

for i in specie_name:
    temp_list = []
    for j in column_name:
        columns = list(column_name)
        columns.remove(j)
        temp_list.append(log_iris(column_name,i)-log_iris(columns,i))
    acc_list.append(temp_list)
        
    
answer2 = pd.DataFrame({'Setosa':np.round(acc_list[0],2),
                        'Versicolor':np.round(acc_list[1],2),
                        'Virginica':np.round(acc_list[2],2)},
                        index = [i+' Delta' for i in column_name] )
     
print()
print('Question 2:\n')
print(answer2)
print()
print('From the table we can know that for different species there are different impacts for reducing one of flower features.')
print('Interstingly, for Setoca removing any one of features will not affect accuracy of model.')
   