# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:01:50 2019

@author: Yazhuo Ma
"""

import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MA.csv')



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
# 2017
df = data.loc[data.loc[:,'Year'] == 2017,]

df_2017 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()

df_2017_answer = Balance(df_2017)

print('2017')
#Question 1
print('')
print('Q1:')
print('The average weekly balance is:',round(np.mean(df_2017_answer['Balance']),2))
print('The volatility of weekly balance is:',round(np.std(df_2017_answer['Balance']),2))

#Question 2
print('')
print('Q2:')
print('See the plot at the end!!')
a = plt.figure(1)
plt.plot(df_2017_answer['Week_Number'],df_2017_answer['Balance'])
plt.title('2017')
plt.xlabel('Week_Number')
plt.ylabel('Balance')
a.show()

#Question 3
print('')
print('Q3:')
print('The max value of account is:',round(np.max(df_2017_answer['Balance']),2),'in Week',df_2017_answer['Week_Number'][np.max(df_2017_answer['Balance'])==df_2017_answer['Balance']].values)
print('The min value of account is:',round(np.min(df_2017_answer['Balance']),2),'in Week',df_2017_answer['Week_Number'][np.min(df_2017_answer['Balance'])==df_2017_answer['Balance']].values)

#Question 4
print('')
print('Q4:')
print('The final value of account is:',round(np.max(df_2017_answer['Balance'].tail(1)),2))

#Question 5

def LogestIncreasing(lis):
    
    max_length = 1
    current_length = 1
    maxIndex = 0
    max_list = []
    for i in range(1, len(lis)):
        if lis[i] > lis[i-1] :
            
            current_length = current_length + 1
            
        else :
            if max_length < current_length:
                max_length = current_length
                maxIndex = i - max_length
            
            else:
                current_length = 1	
        
    # To handle the last element
    if (max_length < current_length) :
        max_length = current_length
        maxIndex = len(lis) - max_length
        
    for i in range(maxIndex, (max_length+maxIndex)):
        max_list.append(lis[i])
        
    return max_list 


def LogestDecreasing(lis):
    
    max_length = 1
    current_length = 1
    maxIndex = 0
    max_list = []
    for i in range(1, len(lis)):
        if lis[i] < lis[i-1] :
            
            current_length = current_length + 1
            
        else :
            if max_length < current_length:
                max_length = current_length
                maxIndex = i - max_length
            
            else:
                current_length = 1	
        
    # To handle the last element
    if (max_length < current_length) :
        max_length = current_length
        maxIndex = len(lis) - max_length
        
    for i in range(maxIndex, (max_length+maxIndex)):
        max_list.append(lis[i])
        
    return max_list 

		


print('')
print('Q5:')
print('Maximum duration (in weeks) that the account was growing is:',len(LogestIncreasing(df_2017_answer['Balance'])),'weeks')
print('Maximum duration (in weeks) that the account was decreasing is:',len(LogestDecreasing(df_2017_answer['Balance'])),'weeks')

# 2018
df = data.loc[data.loc[:,'Year'] == 2018,]

df_2018 = df.groupby(['Week_Number', 'Label']).tail(1).reset_index()

df_2018_answer = Balance(df_2018)

print('')
print('2018')
#Question 1
print('')
print('Q1:')
print('The average weekly balance is:',round(np.mean(df_2018_answer['Balance']),2))
print('The volatility of weekly balance is:',round(np.std(df_2018_answer['Balance']),2))

#Question 2
print('')
print('Q2:')
print('See the plot at the end!!')
b = plt.figure(2)
plt.plot(df_2018_answer['Week_Number'],df_2018_answer['Balance'])
plt.title('2018')
plt.xlabel('Week_Number')
plt.ylabel('Balance')
b.show()
#Question 3
print('')
print('Q3:')
print('The max value of account is:',round(np.max(df_2018_answer['Balance']),2),'in Week',df_2018_answer['Week_Number'][np.max(df_2018_answer['Balance'])==df_2018_answer['Balance']].values)
print('The min value of account is:',round(np.min(df_2018_answer['Balance']),2),'in Week',df_2018_answer['Week_Number'][np.min(df_2018_answer['Balance'])==df_2018_answer['Balance']].values)

#Question 4
print('')
print('Q4:')
print('The final value of account is:',round(np.max(df_2018_answer['Balance'].tail(1)),2))

#Question 5

print('')
print('Q5:')
print('Maximum duration (in weeks) that the account was growing is:',len(LogestIncreasing(df_2018_answer['Balance'])),'weeks')
print('Maximum duration (in weeks) that the account was decreasing is:',len(LogestDecreasing(df_2018_answer['Balance'])),'weeks')

