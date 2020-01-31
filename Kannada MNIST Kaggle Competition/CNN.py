# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:18:59 2019

@author: 62382
"""

import numpy as np
import pandas as pd
pd.options.display.max_columns = 8
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler,EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# read the data

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
extend_data = pd.read_csv("Dig-MNIST.csv")


#explore the data
print('Training data:')
print(train_data.shape)
train_data.head()


print('Testing data:')
print(test_data.shape)
test_data.head()

##NA?
print('Any missing values?')
train_data.isnull().any().sum()
test_data.isnull().any().sum()
print('No misssing values.')


X_train = train_data.iloc[:,1:].values.astype('float32')
y_train = train_data['label'].values.astype('int')
X_test = test_data.iloc[:,1:].values.astype('float32')

# distribution  of numbers in train data

sns.countplot(y_train)
plt.title('Distribution  of classes in train data')

# preview the images first
plt.figure(figsize=(12,10))
col, row = 10, 5
for i in range(50):  
    plt.subplot(row, col, i+1)
    plt.imshow(X_train[i].reshape((28,28)), cmap = 'gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()


#Data Preprocessing



# Normalize the data
#Data is not normalized so we divide each image to 255 that is basic normalization for images.

X_train = X_train/255.0

X_test = X_test/255.0

# reshape data  into (28x28x1) 3D matrices.
# reshape(examples, height, width, channels) channels = 1 because it is gray scale, for RGB channels = 3
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# encode 10 classes with One Hot Encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)


'''
CNN
Find out one best CNN first
'''

#Split data into train and validation
seed = 2019
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=seed)

# Building ConvNet


cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=3,activation='relu',input_shape=(28, 28, 1)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(32, kernel_size = 3, activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(32, kernel_size=5, activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.4))

cnn_model.add(Conv2D(64, kernel_size=5,activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, kernel_size = 7, activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, kernel_size=7, activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.4))

cnn_model.add(Conv2D(128, kernel_size=4, activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.4))
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))




# Compiling the model
cnn_model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


#Data Augmentation
data_generator = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        ) 
 
data_generator.fit(X_train)


# fit the model
batch_size = 128
epochs = 15
reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=7, verbose=1, mode='max', restore_best_weights=True)

history1 = cnn_model.fit_generator(data_generator.flow(X_train, y_train, batch_size = batch_size),
                                  validation_data = (X_valid, y_valid),
                                  epochs=epochs,steps_per_epoch = X_train.shape[0]//batch_size,
                                  callbacks=[reduce_lr,early_stopping], verbose=1)


#Evaluating the Model


# plot loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()

# plot accuracy

plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'])
plt.show()


# prediction of validation data
y_valid_pred = np.argmax(cnn_model.predict(X_valid),axis=1)
y_valid_label = np.argmax(y_valid,axis=1)

#confusion matrix

cm = confusion_matrix(y_valid_label,y_valid_pred)

df_cm = pd.DataFrame(cm, columns=np.unique(y_valid_label), index = np.unique(y_valid_label))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size



'''
10 CNNs

Uncomment code if you wanna run this part
Warning: Running at least 1 hour if you have enabled your GPU.
'''

# start of 10 CNNs

'''
# Building ConvNet
nets = 10
cnn_model = [0] *nets
for i in range(nets):
    cnn_model[i] = Sequential()
    cnn_model[i].add(Conv2D(32, kernel_size=3,activation='relu',input_shape=(28, 28, 1)))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Conv2D(32, kernel_size = 3, activation='relu'))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Conv2D(32, kernel_size=5, activation='relu'))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Dropout(0.4))
    
    cnn_model[i].add(Conv2D(64, kernel_size=5,activation='relu'))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Conv2D(64, kernel_size = 7, activation='relu'))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Conv2D(64, kernel_size=7, activation='relu'))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Dropout(0.4))
    
    cnn_model[i].add(Conv2D(128, kernel_size=4, activation='relu'))
    cnn_model[i].add(BatchNormalization())
    cnn_model[i].add(Flatten())
    cnn_model[i].add(Dropout(0.4))
    cnn_model[i].add(Dense(128, activation='relu'))
    cnn_model[i].add(Dense(num_classes, activation='softmax'))

    # Compiling the model
    cnn_model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Data Augmentation
data_generator = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        ) 
 



# train the model
batch_size = 128
epochs = 15
reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=7, verbose=1, mode='max', restore_best_weights=True)
history = [0] * nets

for i in range(nets):
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_train, y_train, test_size = 0.1, random_state = i*10+1)
    history[i] = cnn_model[i].fit_generator(data_generator.flow(X_train_1,y_train_1, batch_size=batch_size),
        epochs = epochs, steps_per_epoch = X_train_1.shape[0]//batch_size,  
        validation_data = (X_valid_1,y_valid_1), callbacks=[reduce_lr,early_stopping], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        i+1,epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))
    
    
# ENSEMBLE PREDICTIONS 
predictions = np.zeros( (X_test.shape[0],10) ) 
for i in range(nets):
    predictions += cnn_model[i].predict(X_test)
    

pred_labels = np.argmax(predictions,axis=1)
''' 
# end of 10 CNNS






'''
# I have not find any good influence of this method. leave this part of code alone for now
#Pseudo-Labelling

X_train_final = np.concatenate((X_train,X_test), axis=0)
y_train_final = np.concatenate((y_train,pred_1), axis=0)

X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X_train_final, y_train_final, test_size=0.1,random_state=seed)

keras.backend.clear_session()

# Building ConvNet again

cnn_model2 = Sequential()
cnn_model2.add(Conv2D(32, kernel_size=(3, 3),
                 strides=1,
                 activation='relu',
                 input_shape=(28, 28, 1)))

cnn_model2.add(Dropout(0.2))
cnn_model2.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'))
cnn_model2.add(Dropout(0.2))
cnn_model2.add(Flatten())
cnn_model2.add(Dense(128, activation='relu'))
cnn_model2.add(Dense(num_classes, activation='softmax'))


# Compiling the model


cnn_model2.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


#Data Augmentation
data_generator = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        ) 
 
data_generator.fit(X_train_2)



history2 = cnn_model2.fit_generator(data_generator.flow(X_train_2, y_train_2, batch_size = batch_size),
                                  validation_data = (X_valid_2, y_valid_2),steps_per_epoch=X_train.shape[0] // batch_size,
                                  epochs=epochs)

#Evaluating the Model

cnn_model2.evaluate(X_valid_2, y_valid_2)


# plot loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()

# plot accuracy

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Valid'])
plt.show()
# prediction
pred_2=cnn_model2.predict(X_test)
pred_2_score = cnn_model2.evaluate(X_valid_2, y_valid_2)[1]


if pred_2_score > pred_1_score:
    pred_labels = np.argmax(pred_2,axis=1)
if pred_2_score <= pred_1_score:
    pred_labels = np.argmax(pred_1,axis=1)

pred_labels = np.argmax(predictions,axis=1)
'''

