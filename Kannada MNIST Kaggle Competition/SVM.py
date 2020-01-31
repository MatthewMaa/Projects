# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:51:46 2019

@author: 62382
"""

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# read the data

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



#explore the data
print(train_data.shape)
train_data.head()

print(test_data.shape)
test_data.head()

##NA?
train_data.isnull().any().sum()
test_data.isnull().any().sum()
print('No misssing values.')

#Data Preprocessing

seed = 2
# training data is too large, for now lets just use the fisrst 10000 rows
train_data_temp = train_data.sample(n = 10000,random_state=seed)
X_train = train_data_temp.iloc[:,1:].values.astype('float32')
y_train = train_data_temp['label'].values.astype('int')
X_test = test_data.iloc[:,1:].values.astype('float32')

# distribution  of numbers in train data
sns.countplot(y_train)

# preview the images first
plt.figure(figsize=(12,10))
col, row = 10, 5
for i in range(50):  
    plt.subplot(row, col, i+1)
    plt.imshow(X_train[i].reshape((28,28)), cmap = 'gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()




# Normalize the data
#Data is not normalized so we divide each image to 255 that is basic normalization for images.

X_train = X_train/255.0

X_test = X_test/255.0



#Split data into train and validation
X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_train, y_train, test_size = 0.1, random_state=seed)
print("Train Data Shape: ",X_train_1.shape)
print("Train Label Shape: ",y_train_1.shape)
print("Test Data Shape: ",X_valid_1.shape)
print("Test Label Shape: ",y_valid_1.shape)


'''
SVM classifier
'''
#C-Support Vector Classification.
classifier = svm.SVC(random_state=seed) 




'''
Plan A:  Gray Scale Images
'''
start_time_A = time.time()
classifier.fit(X_train_1, y_train_1)
fittime_A = time.time() - start_time_A
print("Time consumed to fit grayscale model: ",time.strftime("%H:%M:%S", time.gmtime(fittime_A)))

score_A = classifier.score(X_valid_1,y_valid_1)
print("Accuracy for grayscale: ",score_A)



'''
Plan B:  Binary Images
To simplify the problem, convert images to black and white from gray scale by replacing all values > 0 to 1.
'''
X_train_2 = X_train_1.copy()
X_valid_2 = X_valid_1.copy()
X_train_2[X_train_2 > 0] = 1
X_valid_2[X_valid_2 > 0] = 1
y_train_2 = y_train_1.copy()
y_valid_2 = y_valid_1.copy()

start_time_B = time.time()
classifier.fit(X_train_2, y_train_2)
fittime_B = time.time() - start_time_B

print("Time consumed to fit binary model: ",time.strftime("%H:%M:%S", time.gmtime(fittime_B)))

score_B = classifier.score(X_valid_2,y_valid_2)
print("Accuracy for binary: ",score_B)


'''
Plan C:  Gray Scale Images + Dimension Reduction - PCA

'''
#standardscale data
ss = StandardScaler().fit(X_train_1)
X_std_train = ss.transform(X_train_1)
X_std_valid = ss.transform(X_valid_1)

pca = sklearnPCA().fit(X_std_train)


#Percentage of variance explained by number of selected componenets.

var_percent = pca.explained_variance_ratio_
cum_var_percent = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(16,9))
index = np.array(range(len(var_percent)))
plt.bar(index,var_percent)
plt.xlabel('N_Components')
plt.ylabel('Variance_Ratio')
plt.title('Variance_Ratio vs N_Components ')
plt.show()

plt.figure(figsize=(16,9))
index = np.array(range(len(cum_var_percent)))
plt.bar(index,cum_var_percent)
plt.xlabel('N_components')
plt.ylabel('Cumulative_Variance_Ratio')
plt.title('Cumulative_Variance_Ratio vs N_Components ')
plt.show()

#Keeping 95% of information by choosing components falling within 0.95 cumulative.

N_Components=len(cum_var_percent[cum_var_percent <= 0.95])
print("Keeping 95% Infomation with ",N_Components," components")

# train pca again with N_Components
pca = sklearnPCA(n_components=N_Components)
X_train_3 = pca.fit_transform(X_std_train)
X_valid_3 = pca.transform(X_std_valid)
y_train_3 = y_train_1.copy()
y_valid_3 = y_valid_1.copy()

print("Shape before PCA for Train: ",X_std_train.shape)
print("Shape after PCA for Train: ",X_train_3.shape)
print("Shape before PCA for Validation: ",X_std_valid.shape)
print("Shape after PCA for Validation: ",X_valid_3.shape)

start_time_C = time.time()
classifier.fit(X_train_3, y_train_3)
fittime_C = time.time() - start_time_C
print("Time consumed to fit grayscale(PCA) model: ",time.strftime("%H:%M:%S", time.gmtime(fittime_C)))

score_C = classifier.score(X_valid_3,y_valid_3)
print("Accuracy for grayscale(PCA): ",score_C)

'''
Plan D:  Binary Images + Dimension Reduction - PCA

'''
#standardscale data
ss = StandardScaler().fit(X_train_2)
X_std_train = ss.transform(X_train_2)
X_std_valid = ss.transform(X_valid_2)

pca = sklearnPCA().fit(X_std_train)

#Percentage of variance explained by number of selected componenets.

var_percent = pca.explained_variance_ratio_
cum_var_percent = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(16,9))
index = np.array(range(len(var_percent)))
plt.bar(index,var_percent)
plt.xlabel('N_Components')
plt.ylabel('Variance_Ratio')
plt.title('Variance_Ratio vs N_Components ')
plt.show()

plt.figure(figsize=(16,9))
index = np.array(range(len(cum_var_percent)))
plt.bar(index,cum_var_percent)
plt.xlabel('N_components')
plt.ylabel('Cumulative_Variance_Ratio')
plt.title('Cumulative_Variance_Ratio vs N_Components ')
plt.show()

#Keeping 95% of information by choosing components falling within 0.95 cumulative.

N_Components=len(cum_var_percent[cum_var_percent <= 0.95])
print("Keeping 95% Infomation with ",N_Components," components")

# train pca again with N_Components
pca = sklearnPCA(n_components=N_Components)
X_train_4 = pca.fit_transform(X_std_train)
X_valid_4 = pca.transform(X_std_valid)
y_train_4 = y_train_1.copy()
y_valid_4 = y_valid_1.copy()

print("Shape before PCA for Train: ",X_std_train.shape)
print("Shape after PCA for Train: ",X_train_4.shape)
print("Shape before PCA for Validation: ",X_std_valid.shape)
print("Shape after PCA for Validation: ",X_valid_4.shape)

start_time_D = time.time()
classifier.fit(X_train_4, y_train_4)
fittime_D = time.time() - start_time_D
print("Time consumed to fit binary(PCA) model: ",time.strftime("%H:%M:%S", time.gmtime(fittime_D)))

score_D = classifier.score(X_valid_4,y_valid_4)
print("Accuracy for binary(PCA): ",score_D)

'''
Compare four plans
'''

acc_list = [score_A,score_B,score_C,score_D]
name_list = ['Grayscale','Binary','Grayscale(PCA)','Binary(PCA)']
time_list = [fittime_A,fittime_B,fittime_C,fittime_D]
plt.figure(figsize=(16,9))
plt.subplot(2, 1, 1)
plt.plot(name_list,acc_list)
plt.xlabel('Plans')
plt.ylabel('Accuracy')
plt.title('Accuracuy vs Plans ')

plt.subplot(2, 1, 2)
plt.plot(name_list,time_list)
plt.xlabel('Plans')
plt.ylabel('Training time')
plt.title('Training time vs N_Components ')

plt.show()


'''
Parameter Selection for binary(pca) by GridSearchCV
'''

print("Default Parameters of classifer are: \n",classifier.get_params)

parameters = {'gamma': [1, 0.1, 0.01, 0.001],
             'C': [1000, 100, 10, 1]} 

X_train_5 =X_train_4[0:999]
y_train_5 = y_train_4[0:999]
ps = GridSearchCV(classifier , param_grid=parameters, cv=5)
ps.fit(X_train_5,y_train_5)
print("\nBest C and Gamma Combination: ",ps.best_params_)
print("\nBest Accuracy acheieved: ",ps.best_score_)

#train model again with best parameter
c=ps.best_params_['C']
gamma=ps.best_params_['gamma']
classifier = svm.SVC(C=c,gamma = gamma,random_state=seed) 
classifier.fit(X_train_4, y_train_4)
score_E = classifier.score(X_valid_4,y_valid_4)
print("\n Accuracy before Training model with best parameter:",score_D)
print("\n Accuracy after Training model with best parameter:",score_E)

'''
Conclusion 

We shoul use plan D and best parameter to build our model with all the training data, which
shold give us the best model.
'''
