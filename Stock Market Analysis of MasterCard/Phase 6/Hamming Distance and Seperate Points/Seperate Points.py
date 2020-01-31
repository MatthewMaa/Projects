# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:09:00 2019

@author: Matthew
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

df_2017 = df_2017.dropna()





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


# Question 1

print('')
print('Question 1: ')
print('From the figure below we can know what are the points needs to be removed:')
# to plot
colors = df_2017 ['Label'].values
point_x = df_2017 ['mu'].values
point_y = df_2017 ['sd'].values
week_number = df_2017 ['Week_Number'].values

c = pltcolor(colors)

plt.figure(figsize = (10, 7))
plt.scatter(point_x, point_y,c=c,alpha=0.5,s = 100)
    
for x,y,z in zip(point_x, point_y,week_number):


    plt.annotate(z, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center',
                 size = 9
                 ) # horizontal alignment can be left, right or center

plt.title ('Weeks and Labels')
plt.xlabel ('mu')
plt.ylabel ('sigma')

plt.show()

print('After removing some of the points,draw a line in the plot to seperate points:')
print('The equation of the line is showed in the upper left corner.')

selected_weeks = [1,21,22,25,29,28,41,43,6,50,48,24,46,9,42,27,18,10,19,4]

df_2017_new = df_2017.loc[df_2017['Week_Number'].isin(selected_weeks)].reset_index()

# to plot
colors = df_2017_new ['Label'].values
point_x = df_2017_new ['mu'].values
point_y = df_2017_new ['sd'].values
week_number = df_2017_new ['Week_Number'].values

c = pltcolor(colors)

plt.figure(figsize = (10, 7))
plt.scatter(point_x, point_y,c=c,alpha=0.5,s = 100)
    
for x,y,z in zip(point_x, point_y,week_number):


    plt.annotate(z, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center',
                 size = 9
                 ) # horizontal alignment can be left, right or center

plt.title ('Labels and Seperate line')
plt.xlabel ('mu')
plt.ylabel ('sigma')


x = np.linspace(min(point_x),max(point_x),100)
y = 1.83*x+0.002
plt.plot(x, y, 'blue', label='y = 1.83x+0.002')
plt.legend(loc='upper left')
plt.show()


# Question 2
print('')
print('Question 2: ')
print('The plot of 2018 after prediction:')
# use the line to predict year 2
def predict(x,y):
    labels = []
    for i in range(len(x)):

        if y[i] > (1.83*x[i]+0.002):
            labels.append('R')
        elif y[i] < (1.83*x[i]+0.002):
            labels.append('G')
        else:
            labels.append('Y')
    return labels

point_x = df_2018['mu'].values
point_y = df_2018['sd'].values
week_number = df_2018['Week_Number'].values
            
c = pltcolor(predict(point_x,point_y))

plt.figure(figsize = (10, 7))
plt.scatter(point_x, point_y,c=c,alpha=0.5,s = 100)
    
for x,y,z in zip(point_x, point_y,week_number):


    plt.annotate(z, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,0), # distance from text to points (x,y)
                 ha='center',
                 size = 9
                 ) # horizontal alignment can be left, right or center

plt.title ('Prediction of 2018')
plt.xlabel ('mu')
plt.ylabel ('sigma')
plt.show()


#Accuracy
prediction = predict(point_x,point_y)
Y_2018 = df_2018['Label'].values
acc = np.mean (prediction == Y_2018)
print('Accuracy rate is:','{:.2%}'.format(acc))

#Question 3

print('')
print('Question 3: ')
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

# replace actual label with predcition label and apply true labels strategy
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018_balance = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()
df_2018_balance = df_2018_balance.loc[0:len(df_2018_balance)-2,]

df_2018_balance['Week_Number'] = range(1,len(df_2018_balance['Week_Number'])+1)
df_2018_balance['Label'] = prediction

df_predict = Balance(df_2018_balance)

# plot
plt.figure(figsize = (10, 7))
plt.plot(df_predict['Week_Number'],df_predict['Balance'])
plt.title('Result of applying trading strategy for 2018 with predicted labels')
plt.xlabel('Week_Number')
plt.ylabel('Balance')
plt.show()