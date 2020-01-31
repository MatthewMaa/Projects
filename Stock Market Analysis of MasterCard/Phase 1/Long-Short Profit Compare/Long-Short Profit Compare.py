# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:51:27 2019

@author: Yazhuo Ma
"""

import csv


#read data from csv file
rows = []

with open ('MA.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    
    next(reader)
    
    
    for row in reader:
        rows.append(row)
        
        
    
        
             
#create empty list to store data
openprice = []
adclose = []
#set the first day's overnight return as 0
returnrate = [0]


for i in range(756,reader.line_num-1):
    openprice.append(rows[i][8])
    adclose.append(rows[i][13])
    

#convert string into float 
openprice = [float (i) for i in openprice]
adclose = [float (i) for i in adclose]

#calculate overnight return
for i in range(1, len(openprice)):
    rr = (openprice[i]-adclose[i-1])/adclose[i-1]
    returnrate.append(rr)


# Question 1
totalprofit = 0

for i in range(0, len(returnrate)):
    if returnrate[i] >= 0:
        totalprofit = totalprofit + (100/openprice[i])*(adclose[i]-openprice[i])
    elif returnrate[i] < 0:
        totalprofit = totalprofit + (100/openprice[i])*(openprice[i]-adclose[i])
    else:
        print('error in returnrate')

answer = round((totalprofit/len(returnrate)),2)

print('the average daily profit is $',answer)
print()
# Question 2

longprofit = 0
shortprofit = 0

for i in range(0, len(returnrate)):
    if returnrate[i] >= 0:
        longprofit = longprofit + (100/openprice[i])*(adclose[i]-openprice[i])
    elif returnrate[i] < 0:
        shortprofit = shortprofit + (100/openprice[i])*(openprice[i]-adclose[i])
    else:
        print('error in returnrate')
        
print('Profit from long positions:', round(longprofit,2))
print()
print('Profit from short positions:', round(shortprofit,2))
print()
print('So short position is more profitable.')
print()

# Question 3
x = [x / 100.0 for x in range(0, 100, 1)]
profitlist = []

for i in range(0, 100):
    
    totalprofit = 0
    
    for j in range(0, len(returnrate)):
        
        if abs(returnrate[j]) >= x[i] and returnrate[j] >= 0:
            totalprofit = totalprofit + (100/openprice[j])*(adclose[j]-openprice[j])
        
        elif abs(returnrate[j]) >= x[i] and returnrate[j] < 0:
            totalprofit = totalprofit + (100/openprice[j])*(openprice[j]-adclose[j])
        else:
            continue
            
    profitlist.append(totalprofit)
  
    
profitlist = [ round(elem/len(returnrate), 2) for elem in profitlist ]

print('let''s compare x with cooresponding profit:')
print(x)
print()
print(profitlist)
print()
print('So the optimal value for x is 0.04(4%), which brings in $0.00 average profit per day. This means the company should set a threshold for overnight reutrn as 4%')
print()

# Question 4

x = [x / 100.0 for x in range(0, 100, 1)]
longprofitlist = []
shortprofitlist = []

for i in range(0, 100):
    
    longprofit = 0
    shortprofit = 0
    
    
    for j in range(0, len(returnrate)):
        
        if abs(returnrate[j]) >= x[i] and returnrate[j] >= 0:
            longprofit = longprofit + (100/openprice[j])*(adclose[j]-openprice[j])
        
        elif abs(returnrate[j]) >= x[i] and returnrate[j] < 0:
            shortprofit = shortprofit + (100/openprice[j])*(openprice[j]-adclose[j])
        else:
            continue
            
    longprofitlist.append(longprofit)
    shortprofitlist.append(shortprofit)
  
    
longprofitlist = [ round(elem/len(returnrate), 2) for elem in longprofitlist ]
shortprofitlist = [ round(elem/len(returnrate), 2) for elem in shortprofitlist ]

print('let''s compare x with cooresponding profit:')
print('x:',x)
print()
print('longpositions:',longprofitlist)
print()
print('shortpositions:',shortprofitlist)
print()
print('It is not hard to conclude that for long positions the optimal value of x is 0.03(3%) and for short positions the optimal value of x is 0.0(0%).')