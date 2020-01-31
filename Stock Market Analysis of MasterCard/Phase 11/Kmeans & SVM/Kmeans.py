# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:11:41 2019

@author: Matthew
1. take k = 3 and use k-means sklearn library routing for kmeans (random initialization and use the defaults). Take
k = 1, 2, . . . 7, 8 and compute the distortion vs. k. Use the
”knee” method to find out the best k.
2. for this optimal k, examine your clusters and for each cluster compute the percentage of ”green” and ”red” weeks in
that cluster.
3. does your k-means clustering find any ”pure” clusters (percent of red or green weeks in a cluster is more than, say,
90% of all weeks in that cluster)?

"""
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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


total_df = pd.concat([df_2017, df_2018]).reset_index()

# Q1
print()
print('Question 1\n')

x = total_df[['mu','sd']].values

inertia_list = []
for k in range (1 ,9):
    kmeans_classifier = KMeans (n_clusters =k, init = 'random')
    y_kmeans = kmeans_classifier.fit_predict(x)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)
fig,ax = plt.subplots(1,figsize =(7 ,5))
plt.plot(range(1,9), inertia_list, marker ='o', color ='green')
plt.xlabel ('number of clusters : k')
plt.ylabel ('inertia')
plt.tight_layout ()
plt.show ()

print('The optimal k is 2')

# Q2
print()
print('Question 2\n')

kmeans_classifier = KMeans (n_clusters = 2, init = 'random', random_state = 1)
y_kmeans = kmeans_classifier.fit_predict(x)
colmap = {0: 'blue', 1: 'grey'}
cols = [colmap[k] for k in y_kmeans]

fig = plt.figure(figsize = (10, 7))
ax1 = fig.add_subplot(111)
ax1.scatter(total_df['mu'].values,total_df['sd'].values, c = cols)
plt.title ('Cluster = 2')
plt.xlabel ('mu')
plt.ylabel ('sd')

plt.show()

total_df['Cluster'] = y_kmeans

percent_0_G = len(total_df.loc[(total_df.Cluster == 0) & (total_df.Label == 'G')])/len(total_df.loc[total_df.Cluster == 0])
percent_1_G = len(total_df.loc[(total_df.Cluster == 1) & (total_df.Label == 'G')])/len(total_df.loc[total_df.Cluster == 1])

print(f'Percentage of "Green" in Cluster 0: {round(percent_0_G,2)}')
print(f'Percentage of "Red" in Cluster 0: {round(1-percent_0_G,2)}')
print(f'Percentage of "Green" in Cluster 1: {round(percent_1_G,2)}')
print(f'Percentage of "Red" in Cluster 0: {round(1-percent_1_G,2)}')


# Q3
print()
print('Question 3\n')
print('No pure clusters, the best is 71% percent.')

