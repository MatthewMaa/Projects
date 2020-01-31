# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:06:30 2019

@author: Matthew
"""
"""
1. what is the average tip (as a percentage of meal cost) for
for lunch and for dinner?
2. what is average tip for each day of the week (as a percentage
of meal cost)?
3. when are tips highest (which day and time)?
4. compute the correlation between meal prices and tips
5. is there any relationship between tips and size of the group?
6. what percentage of people are smoking?
7. assume that rows in the tips.csv file are arranged in time.
Are tips increasing with time in each day?
8. is there any difference in correlation between tip amounts
from smokers and non-smokers?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('tips.csv')


df = data.copy()

# Question 1
df['tip_peccent_of_meal_cost'] = df['tip']/df['total_bill']*100

avg_tip_lunch = df.loc[df.time == 'Lunch'].tip_peccent_of_meal_cost.mean()
avg_tip_dinner = df.loc[df.time == 'Dinner'].tip_peccent_of_meal_cost.mean()

print()
print('Question 1:')
print("Average tip for lunch:{}%".format(round(avg_tip_lunch,2)))
print("Average tip for dinner:{}%".format(round(avg_tip_dinner,2)))


#Question 2
avg_tip_weekday = df.groupby(['day'])['tip_peccent_of_meal_cost'].mean().reset_index()
avg_tip_weekday['tip_peccent_of_meal_cost'] = ['{}%'.format(np.round(i,2)) for i in avg_tip_weekday['tip_peccent_of_meal_cost']]

print()
print('Question 2:')
print("Average tip for each day of week:\n", avg_tip_weekday)

#Question 3
highest = df.groupby(['day','time'])['tip_peccent_of_meal_cost'].mean().reset_index()
highest['tip_peccent_of_meal_cost'] == highest['tip_peccent_of_meal_cost'].max()
highest_day_time = highest.loc[highest['tip_peccent_of_meal_cost'] == highest['tip_peccent_of_meal_cost'].max(),]
print()
print('Question 3:')
print("Tips highest day and time:\n", highest_day_time[['day','time']])

# Question 4
x = df['total_bill'].values
y = df['tip'].values

print()
print('Question 4:')
print("Correlation between meal prices and tips:", round(np.corrcoef(x, y)[0][1],2))

#Question 5
x = df['tip'].values
y = df['size'].values

print()
print('Question 5:')
print("Correlation between tips percent and group size:", round(np.corrcoef(x, y)[0][1],2))
print("So there is a positive relationship between tips and group size." )

# Question 6
print()
print('Question 6:')
print("Percentage of people are smoking: {}%".format(round(sum(df['smoker'] == 'Yes')/len(df['smoker'])*100,2)))

# Question 7
print()
print('Question 7:')
plt.figure(figsize = (9, 6))
df_day_time = df.groupby(['day'])['tip'].plot(legend = True)
plt.show()
print('From the plot we can know that tips for each weekday are not increasing with time.')


#Question 8
x = [1 if i == 'Yes' else 0 for i in df['smoker'] ]
y = list(df['tip'].values)

print()
print('Question 8:')
print("Correlation between tip amount and smoker:", round(np.corrcoef(x, y)[0][1],3))
print("So there is almost no relationship between tip amount and smoker or non-smoker." )
