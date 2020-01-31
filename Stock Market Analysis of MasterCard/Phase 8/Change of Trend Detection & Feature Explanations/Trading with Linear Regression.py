# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:08:08 2019

@author: Matthew
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter('ignore')

# data preprocessing
data = pd.read_csv('MA.csv')

# 2017
df = data.loc[data['Year'] == 2017,]

df_2017 = df.dropna().reset_index(drop = True)



# function to calculate profit loss per trade for different W

W = [i for i in range(5,31)]
        

def profit_loss_pertrade(W,data):
    

    
    pl_list = []
    
    for w in W:
        
        pred = []
        
        for i in range(0,len(data)-w):
            X = np.array(range(1,(w+1))).reshape(-1, 1)
            y = np.array(data['Adj Close'][i:i+w])
            
            model = LinearRegression(fit_intercept = True)
            model.fit(X,y)
            pred.append(model.predict(np.array(w+1).reshape(-1, 1)))
            
        profit_loss = 0
        transaction_num = 0
        position = 'no'
        shares = 0
        for i in range(len(pred)):
            
            if pred[i]>data['Adj Close'][i+w-1]:
                
                if position == 'no':
                    
                    shares = 100/data['Adj Close'][i+w-1]
                    position = 'long'
                    
                elif position == 'long':
                    position = 'long'
                
                elif position == 'short':
                    
                    profit_loss += (100-shares*data['Adj Close'][i+w-1])
                    
                    shares = 0
                    position = 'no'
                    transaction_num +=1
                    
            elif pred[i]<data['Adj Close'][i+w-1]:
                
                if position == 'no':
                    
                    shares = 100/data['Adj Close'][i+w-1]
                    position = 'short'
                    
                elif position == 'short':
                    position = 'short'
                
                elif position == 'long':
                    
                    profit_loss += (shares*data['Adj Close'][i+w-1]-100)
                    shares = 0
                    position = 'no'
                    transaction_num +=1
                    
            elif pred[i]==data['Adj Close'][i+w-1]:
                pass
                
                    
        #append profit loss to pl_list
        pl_list.append(profit_loss/transaction_num)
        
    return pl_list
    

        

# Question 1

opt_w_df = pd.DataFrame({'W':W, 'P/L':profit_loss_pertrade(W,df_2017)})
opt_w = opt_w_df.W[opt_w_df['P/L'].idxmax()]
print()
print('Question 1:')
print('From the plot we can know that optimal W =',opt_w)

plt.figure(figsize = (14,6))
sns.lineplot(x = opt_w_df['W'], y = opt_w_df['P/L'] )
plt.title ('P/L vs  W')


plt.show()

#Question 2




# 2018
df = data.loc[data['Year'] == 2018,]

df_2018 = df.dropna().reset_index(drop = True)

# concaate 2017's tail(w = 28) with 2018

df_2018_new = pd.concat([df_2017.iloc[-29:-1,],df_2018]).reset_index()

pred = []

r_squared_list =[]


for i in range(0,len(df_2018_new)-28):
    X = np.array(range(1,(28+1))).reshape(-1, 1)
    y = np.array(df_2018_new['Adj Close'][i:i+28])
    y_old = []
    yhat = []
    model = LinearRegression(fit_intercept = True)
    model.fit(X,y)
    pred.append(model.predict(np.array(28+1).reshape(-1, 1)))
    y = np.append(y,model.predict(np.array(28+1).reshape(-1, 1)))
    y_old = pd.Series(df_2018_new['Adj Close'][i:i+29]).reset_index(drop = True)
    yhat = pd.Series(y)
    SS_Residual = sum((y_old-yhat)**2)
    SS_Total = sum((y_old-np.mean(y_old))**2)
    r_squared = 1 - SS_Residual/SS_Total
    r_squared_list.append(r_squared)





print()
print('Question 2:')

plt.figure(figsize = (14,6))
sns.lineplot(x = range(1,len(df_2018)+1), y = r_squared_list )
plt.title ('r^2 vs Days in Year 2018')
plt.xlabel ('Days in Year 2018')
plt.ylabel ('r^2')


plt.show()

print('The average r square is: ',round(np.mean(r_squared_list),2))
print('This value price movements are relatively smoothy.')

#Question 3&4

def long_position(data):
    

    
    pl_list = []        
    pred = []
    w = 28
    for i in range(0,len(data)-w):
        X = np.array(range(1,(w+1))).reshape(-1, 1)
        y = np.array(data['Adj Close'][i:i+w])
        
        model = LinearRegression(fit_intercept = True)
        model.fit(X,y)
        pred.append(model.predict(np.array(w+1).reshape(-1, 1)))
        
    profit_loss = 0
    transaction_num = 0
    position = 'no'
    shares = 0
    for i in range(len(pred)):
        
        if pred[i]>data['Adj Close'][i+w-1]:
            
            if position == 'no':
                
                shares = 100/data['Adj Close'][i+w-1]
                position = 'long'
                
            elif position == 'long':
                position = 'long'
            
            elif position == 'short':
                
                #profit_loss += (100-shares*data['Adj Close'][i+w-1])
                
                shares = 0
                position = 'no'
                #transaction_num +=1
                
        elif pred[i]<data['Adj Close'][i+w-1]:
            
            if position == 'no':
                
                shares = 100/data['Adj Close'][i+w-1]
                position = 'short'
                
            elif position == 'short':
                position = 'short'
            
            elif position == 'long':
                
                profit_loss += (shares*data['Adj Close'][i+w-1]-100)
                shares = 0
                position = 'no'
                transaction_num +=1
        elif pred[i]==data['Adj Close'][i+w-1]:
                pass
                
    #append profit loss to pl_list
    #pl_list.append(profit_loss/transaction_num)
        
    return [profit_loss,transaction_num]

def short_position(data):
    

    
    pl_list = []        
    pred = []
    w = 28
    for i in range(0,len(data)-w):
        X = np.array(range(1,(w+1))).reshape(-1, 1)
        y = np.array(data['Adj Close'][i:i+w])
        
        model = LinearRegression(fit_intercept = True)
        model.fit(X,y)
        pred.append(model.predict(np.array(w+1).reshape(-1, 1)))
        
    profit_loss = 0
    transaction_num = 0
    position = 'no'
    shares = 0
    for i in range(len(pred)):
        
        if pred[i]>data['Adj Close'][i+w-1]:
            
            if position == 'no':
                
                shares = 100/data['Adj Close'][i+w-1]
                position = 'long'
                
            elif position == 'long':
                position = 'long'
            
            elif position == 'short':
                
                profit_loss += (100-shares*data['Adj Close'][i+w-1])
                
                shares = 0
                position = 'no'
                transaction_num +=1
                
        elif pred[i]<data['Adj Close'][i+w-1]:
            
            if position == 'no':
                
                shares = 100/data['Adj Close'][i+w-1]
                position = 'short'
                
            elif position == 'short':
                position = 'short'
            
            elif position == 'long':
                
                #profit_loss += (shares*data['Adj Close'][i+w-1]-100)
                shares = 0
                position = 'no'
                #transaction_num +=1
        elif pred[i]==data['Adj Close'][i+w-1]:
                pass
                
    #append profit loss to pl_list
    #pl_list.append(profit_loss/transaction_num)
        
    return [profit_loss,transaction_num]

long = long_position(df_2018_new)
short = short_position(df_2018_new)

print()
print('Question 3:')
print('Long position transaction: ',long[1])
print('Short position transaction: ',short[1])

print()
print('Question 4:')
print('Long position average profit/loss per trade: ',round((long[0]/long[1]),2))
print('Short position average profit/loss per trade: ',round((short[0]/short[1]),2))

# Question 5

def days(data):
    

    
    pl_list = []        
    pred = []
    w = 28
    for i in range(0,len(data)-w):
        X = np.array(range(1,(w+1))).reshape(-1, 1)
        y = np.array(data['Adj Close'][i:i+w])
        
        model = LinearRegression(fit_intercept = True)
        model.fit(X,y)
        pred.append(model.predict(np.array(w+1).reshape(-1, 1)))
        
    profit_loss = 0
    transaction_num = 0
    position = 'no'
    shares = 0
    longdays = 0
    shortdays = 0
    for i in range(len(pred)):
        
        if pred[i]>data['Adj Close'][i+w-1]:
            
            if position == 'no':
                
                shares = 100/data['Adj Close'][i+w-1]
                position = 'long'
                longdays +=1
                
            elif position == 'long':
                position = 'long'
                longdays +=1
                
            elif position == 'short':
                
                profit_loss += (100-shares*data['Adj Close'][i+w-1])
                
                shares = 0
                position = 'no'
                transaction_num +=1
                
        elif pred[i]<data['Adj Close'][i+w-1]:
            
            if position == 'no':
                
                shares = 100/data['Adj Close'][i+w-1]
                position = 'short'
                shortdays +=1
            elif position == 'short':
                position = 'short'
                shortdays +=1
            
            elif position == 'long':
                
                profit_loss += (shares*data['Adj Close'][i+w-1]-100)
                shares = 0
                position = 'no'
                transaction_num +=1
        elif pred[i]==data['Adj Close'][i+w-1]:
            
            if position == 'long':
                
                longdays +=1
            elif position == 'short':
                
                shortdays +=1
                
    #append profit loss to pl_list
    #pl_list.append(profit_loss/transaction_num)
        
    return [longdays,shortdays]

days_list = days(df_2018_new)
longdays = days_list[0]
shortdays = days_list[1]

print()
print('Question 5:')
print('Long position average days per trade: ',round((longdays/long[1]),2))
print('Short position average days per trade: ',round((shortdays/short[1]),2))


#Question 6
print()
print('Question 6:')
print('Below are results for 2017:')
print('These results are very similar to 2018.')
#avg r squre for 2017
pred = []

r_squared_list =[]


for i in range(0,len(df_2017)-28):
    X = np.array(range(1,(28+1))).reshape(-1, 1)
    y = np.array(df_2017['Adj Close'][i:i+28])
    y_old = []
    yhat = []
    model = LinearRegression(fit_intercept = True)
    model.fit(X,y)
    pred.append(model.predict(np.array(28+1).reshape(-1, 1)))
    y = np.append(y,model.predict(np.array(28+1).reshape(-1, 1)))
    y_old = pd.Series(df_2017['Adj Close'][i:i+29]).reset_index(drop = True)
    yhat = pd.Series(y)
    SS_Residual = sum((y_old-yhat)**2)
    SS_Total = sum((y_old-np.mean(y_old))**2)
    r_squared = 1 - SS_Residual/SS_Total
    r_squared_list.append(r_squared)
print()
print('The average r square is: ',round(np.mean(r_squared_list),2))

#
long = long_position(df_2017)
short = short_position(df_2017)

print()

print('Long position transaction: ',long[1])
print('Short position transaction: ',short[1])

print()

print('Long position average profit/loss per trade: ',round((long[0]/long[1]),2))
print('Short position average profit/loss per trade: ',round((short[0]/short[1]),2))

days_list = days(df_2017)
longdays = days_list[0]
shortdays = days_list[1]

print()

print('Long position average days per trade: ',round((longdays/long[1]),2))
print('Short position average days per trade: ',round((shortdays/short[1]),2))