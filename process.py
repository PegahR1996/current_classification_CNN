# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:58:08 2022

@author: Pegah
"""

import pandas as pd

# for i in range(2, 7, 1):
#     for j in [1000, 1500]:
#         file_address_normal = str(i) + ' NM\\' + str(j) + ' RPM\\NORMAL ' + str(i) + 'NM-' + str(j) + ' RPM.csv'
#         file_address_full = str(i) + ' NM\\' + str(j) + ' RPM\\FULL DE ' + str(i) + 'NM-' + str(j) + '.csv'
#         print(file_address_full)
#         df = pd.read_csv(file_address_normal)
    
#         print(df[0:5])
        
#         data = df.to_numpy()

import os
import numpy as np
size = 196
normal = np.empty((size,0))
full = np.empty((size,0))
all_data = np.empty((size,0))
label = []
             
             
for root, dirs, files in os.walk(r'C:\Users\User\Downloads\Compressed\persent\persent'):
     print('root')
     print(root)
     print('dirs')
     print(dirs)
     for file in files:
         if ('90_' in file) or ('FULL' in file):
             df = pd.read_csv(root + '\\' + file)
             # print(df[0:5])
             data = df.to_numpy()
             normal = np.append(normal, data[0:size, 1:2], axis=1)
             all_data = np.append(all_data, data[0:size, 1:2], axis=1)
             label.append(0)
         elif ('70_' in file) or ('75_' in file):
             df = pd.read_csv(root + '\\' + file)
             # print(df[0:5])
             data = df.to_numpy()
             normal = np.append(normal, data[0:size, 1:2], axis=1)
             all_data = np.append(all_data, data[0:size, 1:2], axis=1)
             label.append(1)
         elif ('40_' in file) or ('50_' in file):
             df = pd.read_csv(root + '\\' + file)
             # print(df[0:5])
             data = df.to_numpy()
             normal = np.append(normal, data[0:size, 1:2], axis=1)
             all_data = np.append(all_data, data[0:size, 1:2], axis=1)
             label.append(2)
         elif ('1_' in file) or ('1.5_' in file) or ('2_' in file) or ('2.5_' in file) or ('3_' in file) or ('3.5_' in file) or ('Normal' in file) or ('normal' in file):
             # print(file)
             df = pd.read_csv(root + '\\' + file)
             # print(df[0:5])
             data = df.to_numpy()
             normal = np.append(normal, data[0:size, 1:2], axis=1)
             all_data = np.append(all_data, data[0:size, 1:2], axis=1)
             label.append(3)
             
             
from sklearn.model_selection import train_test_split             
from sklearn import svm
from sklearn.utils import shuffle
X = all_data.T
y = label
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = svm.SVC()
regr.fit(X_train, y_train)

predict = regr.predict(X_test)
print(predict)

# predict = np.where(predict > 0.5, 1, predict)
# predict = np.where(predict <= 0.5, 0, predict)

correct = (predict == y_test)
accuracy = correct.sum() / correct.size

print(accuracy)
print(y_test)
print(predict)


































def create_model_api_LSTMCNN_dilate(input_shape):
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, LSTM, Flatten, Dense, SimpleRNN
    from tensorflow.keras.layers import TimeDistributed, Activation, LeakyReLU, Bidirectional
    from tensorflow.keras.models import Sequential, Model
    
    x_in = Input(input_shape)
    x = x_in
    x = Conv1D(16, 3, kernel_initializer="random_uniform",
                            bias_initializer="random_normal", dilation_rate = 8,
                            padding="causal", activation='relu')(x)
    x = LeakyReLU()(x)
    x = MaxPooling1D()(x)
    
    x = Conv1D(32, 3, kernel_initializer="random_uniform",
                            bias_initializer="random_normal", dilation_rate = 4,
                            padding="causal", activation='relu')(x)
    # x = Activation('relu')(x)
    x = LeakyReLU()(x)
    x = MaxPooling1D()(x)
    
    x = Flatten()(x)
        
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(4)(x)
    y = Activation('softmax')(x)
    
    model = Model(inputs=x_in, outputs=y)
    
    model.summary()
    
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['acc'])
    
    return model


input_shape = (X_train.shape[1], 1)
model = create_model_api_LSTMCNN_dilate(input_shape)

import tensorflow

class AccuracyHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))




from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
seed=7
np.random.seed(seed)


history = AccuracyHistory()

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))
# X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
# X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))



# train_features,test_features,train_labels, test_labels = train_test_split(data_final, label, test_size=0.10, random_state=seed)
# train_features,valid_features,train_labels, valid_labels = train_test_split(train_features, train_labels, test_size=0.10, random_state=seed)
y_true = y_test
train_labels = np.asarray(pd.get_dummies(y_train), dtype = np.int8)
test_labels = np.asarray(pd.get_dummies(y_test), dtype = np.int8)
# valid_labels = np.asarray(pd.get_dummies(valid_labels), dtype = np.int8)

print(X_train.shape)
print(train_labels.shape)

model.fit(X_train, train_labels,
          batch_size=32,
          epochs=50,
          verbose=1, callbacks=[history])


score = model.evaluate(X_test, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
print(confusion_matrix(y_true, y_pred))