# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:44:23 2019

@author: Yazhuo Ma
"""
import pandas as pd
import math
import numpy as np 
from pandas import DataFrame

df = pd.read_csv('BreadBasket_DMS_output.csv')

# Question 1
#a

# Get the num of the transactions per hour
#hour = df.groupby('Hour',as_index=False)['Transaction'].count()
hour = df.groupby('Hour')['Transaction'].nunique().reset_index()
# get the hour
print('Q1.a:\n')
print('We can get the busiest hour with cooresponding transaction number as folloiwing:\n')
print(hour[hour['Transaction'] == max(hour['Transaction'])].to_string(index = False))
print()

#b

# Get the num of the transactions per weekday
week = df.groupby('Weekday')['Transaction'].nunique().reset_index()
# get the weekday
print('Q1.b:\n')
print('We can get the busiest day of the week with cooresponding transaction number as folloiwing:\n')
print(week[week['Transaction'] == max(week['Transaction'])].to_string(index = False))
print()

#c

# Get the num of the transactions per period
period = df.groupby('Period')['Transaction'].nunique().reset_index()
# get the period
print('Q1.c:\n')
print('We can get the busiest period with cooresponding transaction number as folloiwing:\n')
print(period[period['Transaction'] == max(period['Transaction'])].to_string(index = False))
print()

# Question 2


#a
# Get the sum of the revenue per hour
hour_revenue = df.groupby('Hour')['Item_Price'].sum().reset_index(name = 'Revenue')
# get the hour
print('Q2.a:\n')
print('We can get the most profitable hour with cooresponding revenue as folloiwing:\n')
print(hour_revenue[hour_revenue['Revenue'] == max(hour_revenue['Revenue'])].to_string(index = False))
print()

#b
# Get the sum of the revenue per weekday
week_revenue = df.groupby('Weekday')['Item_Price'].sum().reset_index(name = 'Revenue')
# get the weekday
print('Q2.b:\n')
print('We can get the most profitable weekday with cooresponding revenue as folloiwing:\n')
print(week_revenue[week_revenue['Revenue'] == max(week_revenue['Revenue'])].to_string(index = False))
print()

#c
# Get the sum of the revenue per period
period_revenue = df.groupby('Period')['Item_Price'].sum().reset_index(name = 'Revenue')
# get the period
print('Q2.c:\n')
print('We can get the most profitable period with cooresponding revenue as folloiwing:\n')
print(period_revenue[period_revenue['Revenue'] == max(period_revenue['Revenue'])].to_string(index = False))
print()

#Question 3

item = df.groupby('Item')['Transaction'].count().reset_index(name = 'Count')

print('Q3:\n')
print('We can get the most popular item with cooresponding transaction as folloiwing:\n')
print(item[item['Count'] == max(item['Count'])].to_string(index = False))
print()
print('We can get the least popular item with cooresponding transaction as folloiwing:\n')
print(item[item['Count'] == min(item['Count'])].to_string(index = False))
print()

#Question 4

weekday = df.groupby(['Year','Month', 'Day','Weekday'])['Weekday'].count().reset_index(name = 'Count')
weekday1 = weekday.groupby('Weekday')['Weekday'].count().reset_index(name = 'Count')
weekday1['Transaction'] = week['Transaction']
# calculate barrista needed per weekday
weekday1['Barristas_Number'] = np.ceil(weekday1['Transaction'].values/weekday1['Count'].values/50)

print('Q4:\n')
print('The barristas needed per weekday is as following:\n')
print(weekday1)
print()

#Question 5

food = ['Alfajores','Bacon','Baguette','Bakewell','Bare Popcorn','Bread','Bread Pudding','Brioche and salami',
        'Brownie','Cake','Caramel bites','Cherry me Dried fruit','Chicken Stew','Chimichurri Oil',
        'Chocolates','Cookies', 'Crepes','Crisps','Duck egg','Dulce de Leche','Eggs','Empanadas',
        'Extra Salami or Feta','Focaccia','Frittata','Fudge','Gingerbread syrup','Granola','Hack the stack',
        'Hearty & Seasonal','Honey','Jam','Jammie Dodgers','Kids biscuit','Lemon and coconut','Medialuna',
        'Mighty Protein','Muesli','Muffin','Olum & polenta','Panatone','Pastry','Pintxos','Polenta','Raspberry shortbread sandwich',
        'Raw bars','Salad', 'Sandwich','Scone','Spanish Brunch','Spread','Tacos/Fajita','Tartine','Toast',
        'Truffles', 'Vegan Feast', 'Vegan mincepie', 'Victorian Sponge'
 ]
foodIndex = df[df.isin({'Item':food})['Item']].index
df.loc[foodIndex, 'Group'] = 'Food'
#item['Item'][item.isin({'Item':food})['Item'] == False]
drink = ['Coffee','Coke','Ella''s Kitchen Pouches','Hot chocolate','Juice','Mineral water','My-5 Fruit Shoot','Smoothies','Soup','Tea']
drinkIndex = df[df.isin({'Item':drink})['Item']].index
df.loc[drinkIndex, 'Group'] = 'Drink'

nullIndex = df[df['Group'].isnull()].index
df.loc[nullIndex, 'Group'] = 'Unknown'


groups = df.groupby('Group', as_index = False)['Item_Price'].mean()

print('Q5:\n')
print('The average price of a drink and a food item are as following:\n')
print(np.round(groups,2).to_string(index = False))
print()

# Question 6

group_revenue = df.groupby('Group', as_index = False)['Item_Price'].sum()
print('Q6:\n')
print('From the table below we can know that the coffee shop makes more money from selling drink :\n')
print(group_revenue.to_string(index = False))
print()

#Question 7
mostpopular = df.groupby(['Weekday','Item']).agg({'Item':'count'})
l = mostpopular['Item'].groupby(level=0, group_keys=False).nlargest(5)


print('Q7:\n')
print('From the table below we can know the top 5 most popular items for each day of the week:\n')
print(l)
print('\nThis list almost stays the same from day to day.\n')

#Question 8
leastpopular = df.groupby(['Weekday', 'Item']).agg({'Item':'count'})
s = leastpopular['Item'].groupby(level=0, group_keys=False).nsmallest(5)


print('Q8:\n')
print('From the table below we can know the  bottom 5 least popular items for each day of the week:\n')
print(s)
print('\nThis list does not stay the same from day to day.\n')


#Question 9
ndrink = len(df[df['Group']== 'Drink']['Group'])
ntransaction = sum(hour['Transaction'])
print('Q9:\n')
print('Drinks per transaction:',round(ndrink/ntransaction,2))
