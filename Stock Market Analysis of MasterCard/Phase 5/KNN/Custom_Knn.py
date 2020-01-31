# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:01:41 2019

@author: Matthew
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import collections as cl
import warnings
warnings.simplefilter('ignore')


class Custom_knn():
    
    def __init__ (self , number_neighbots_k , distance_parameter_p):
        
        self.number_neighbots_k = number_neighbots_k
        self.distance_parameter_p = distance_parameter_p
        
    def __str__ (self):
        pass
        
        
    def fit (self, X, Labels):

        self.X = X
        self.Y = Labels


        

        
        
    def predict (self , new_x ):
        
        
        distances = []
        labels = []
    
        for i in range(len(self.X)):
            # compute and store L2 distance
            distances.append([np.linalg.norm (new_x-self.X[i], ord = self.distance_parameter_p), i])
    
        # sort the list
        distances = sorted(distances)
    
        # find out all labels
        for i in range(self.number_neighbots_k):
            index = distances[i][1]
            labels.append(self.Y[index])
            
        # return most labels
        return cl.Counter(labels).most_common(1)[0][0]
        
        
    def draw_decision_boundary (self , new_x ):
        
        point_x = []
        point_y = []
        distances = []
        week_number = []
        colors = []
        
        for i in range(len(self.X)):
            # compute and store L2 distance
            distances.append([np.linalg.norm (new_x-self.X[i], ord = self.distance_parameter_p), i])
        
        distances = sorted(distances)
        
        # color function
        def pltcolor(x):
            
            colors=[]
            for i in range(len(x)):
                if x[i] == 'R':
                    colors.append('red')
                elif x[i] == 'G':
                    colors.append('green')
                else:
                    colors.append('yellow')
            return colors

        # find out nearest k points
        for i in range(self.number_neighbots_k):
            
            index = distances[i][1]
            point_x.append(self.X[index][0])
            point_y.append(self.X[index][1])
            week_number.append(index)
            colors.append(self.Y[index])
        
        point_x.append(new_x[0])
        point_y.append(new_x[1])
        colors.append('None')
        # to plot
        c = pltcolor(colors)
        
        plt.figure(figsize = (10, 7))
        plt.scatter(point_x, point_y,c=c,alpha=0.5,s = 200)
            
        for x,y,z in zip(point_x, point_y,week_number):
        
        
            plt.annotate(z, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,0), # distance from text to points (x,y)
                         ha='center',
                         size = 9
                         ) # horizontal alignment can be left, right or center


        plt.title ('Neareaset 9 Neighbors')
        plt.xlabel ('mu')
        plt.ylabel ('sigma')
    
        # return all points
        return plt.show()
    

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

X_2017 = df_2017 [['mu', 'sd']].values
Y_2017 = df_2017 ['Label'].values


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

X_2018 = df_2018 [['mu', 'sd']].values
Y_2018 = df_2018 ['Label'].values

# Question 1

error_rate = []
p_list = [1,1.5,2]

for p in p_list:
    
    pred_k = []
    c_knn = Custom_knn(9,p)
    c_knn.fit(X_2017,Y_2017)
    
    for i in range(len(X_2017)):
        
        pred_k.append(c_knn.predict(X_2017[i]))
        
    error_rate.append(np.mean ( pred_k != Y_2017 ))

#plot
print('')
print('Question 1: ')
print('From the figure below we can know that p = 1.5 or 2 gives me that highest accuracy.')

plt.plot ( p_list , error_rate , color ='red', linestyle ='dashed',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Error Rate vs. p for 2017')
plt.xlabel ('number of p ')
plt.ylabel ('Error Rate')
plt.show()
    



# Question 2


error_rate = []
p_list = [1,1.5,2]

for p in p_list:
    
    pred_k = []
    c_knn = Custom_knn(9,p)
    c_knn.fit(X_2017,Y_2017)
    
    for i in range(len(X_2018)):
        
        pred_k.append(c_knn.predict(X_2018[i]))
        
    error_rate.append(np.mean ( pred_k != Y_2018))

#plot

print('')
print('Question 2: ')
print('From the figure below we can know that p = 2 gives me that highest accuracy.')

plt.plot ( p_list , error_rate , color ='red', linestyle ='dashed',
marker ='o', markerfacecolor ='black', markersize =10)
plt.title ('Error Rate vs. p for 2018')
plt.xlabel ('number of p ')
plt.ylabel ('Error Rate')
plt.show()


# Question 3

pred_k = []
c_knn = Custom_knn(9,1.5)
c_knn.fit(X_2017,Y_2017)
    
for i in range(len(X_2018)):
    pred_k.append(c_knn.predict(X_2018[i]))

week_index = np.where(pred_k != Y_2018)
week_number = [i for i in week_index]
print('')
print('Question 3: ')
print('Pick two weeks from the weeks below: 5, 40')
print(*week_number)
print()
print("Neighber's id and color for week 5:")
c_knn.draw_decision_boundary(X_2018[5])
print()
print("Neighber's id and color for week 40:")
c_knn.draw_decision_boundary(X_2018[40])


# Question 4


matrix = []

for p in p_list:
    
    pred_k = []
    c_knn = Custom_knn(9,p)
    c_knn.fit(X_2017,Y_2017)
    
    for i in range(len(X_2018)):
        
        pred_k.append(c_knn.predict(X_2018[i]))
        
    matrix.append(confusion_matrix(Y_2018,pred_k))
    
print('')
print('Question 4: ')
print('Confusion matrix for p = 1:')
print(matrix[0])
print('Confusion matrix for p = 1.5:')
print(matrix[1])
print('Confusion matrix for p = 2:')
print(matrix[2])

# Question 5

#p = 1
recall = matrix[0][1][1]/(matrix[0][1][1]+matrix[0][1][0])
specificity = matrix[0][0][0]/(matrix[0][0][0] + matrix[0][0][1])
print('')
print('Question 5: ')
print('For p = 1:')
print('Recall:',round(recall,2),'Sepcificity:',round(specificity,2))
#p = 1.5
recall = matrix[1][1][1]/(matrix[1][1][1]+matrix[1][1][0])
specificity = matrix[1][0][0]/(matrix[1][0][0] + matrix[1][0][1])
print('For p = 1.5:')
print('Recall:',round(recall,2),'Sepcificity:',round(specificity,2))
#p = 2
recall = matrix[2][1][1]/(matrix[2][1][1]+matrix[2][1][0])
specificity = matrix[2][0][0]/(matrix[2][0][0] + matrix[2][0][1])
print('For p = 2:')
print('Recall:',round(recall,2),'Sepcificity:',round(specificity,2))
print('They are all slightly different from each other.')



#Question 6

# Use stargegy in Assignment 4

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



# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018_balance = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()


# calculate values of portfolio for different p
values = []

for p in p_list:
    
    pred_k = []
    c_knn = Custom_knn(9,p)
    c_knn.fit(X_2017,Y_2017)
    
    for i in range(len(X_2018)):
        
        pred_k.append(c_knn.predict(X_2018[i]))
    
    df_copy = df_2018_balance.iloc[0:len(pred_k),]
    df_copy['Label'] = pred_k
    df_2018_answer = Balance(df_copy)
    values.append(np.max(df_2018_answer['Balance'].tail(1)))
    
    
# buy and hold
df_bnh = df_2018_balance.iloc[0:-1,]
share = 100/df_bnh['Adj Close'][0]
bnh_values = [share*(df_bnh['Adj Close'][len(df_bnh)-1])]*3

print('')
print('Question 6: ')
print('By using strategy in assignment 4 , I found p VS portfolio value & buy and hold portfolio value as following: ')
#plot p VS portfolio value
line_up, = plt.plot ( p_list , values , color ='red', linestyle ='dashed',
marker ='o', markerfacecolor ='black', markersize =10,label='buy and hold portfolio value')
line_down,  = plt.plot ( p_list , bnh_values , color ='blue', linestyle ='dashed',
marker ='o', markerfacecolor ='black', markersize =10,label='Assignment4 method portfolio value')
plt.title ('p VS portfolio value in 2018')
plt.xlabel ('number of p ')
plt.ylabel ('final portfolio value')
plt.legend([line_up, line_down], [ 'Assignment4 method portfolio value','buy and hold portfolio value'])
plt.show()

print('So when p = 1 I got the largest portfolio value:',round(values[0],2))
print('It seems that the strategy in assignment 4 is not better than buy and hold strategy.')