# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:48:18 2019

@author: Ayan Dasgupta
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Generating the dataset
dataset = pd.read_csv('Bitcoin_Price.csv')
dataset['Date'] = pd.to_datetime(dataset.Date,format='%d-%m-%Y')
dataset.head()

#Declaring hyperparameters
DATASET_SIZE = 3370
TRAINING_SIZE = 0.8
LOOKBACK = 60
NEURONS = 200
LSTM_ACTIVATION = 'sigmoid'
DENSE_ACTIVATION = 'linear'
KERNEL_INIT = 'he_uniform'
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mean_squared_error'
METRICS = 'mse'
EPOCHS = 100
BATCH_SIZE = 30
DROPOUT = 0.2
PATIENCE = int(0.2*EPOCHS)

#Extracting training and validation sets
training_set = dataset.iloc[0:int(TRAINING_SIZE*DATASET_SIZE), 1:2].values
validation_set = dataset.iloc[int(TRAINING_SIZE*DATASET_SIZE):DATASET_SIZE, 1:2].values

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
validation_set_scaled = sc.fit_transform(validation_set)

#Sequencing training data
X_train = []
y_train = []
for i in range(LOOKBACK, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-LOOKBACK:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Sequencing validation data
X_valid = []
y_valid = []
for i in range(LOOKBACK, len(validation_set_scaled)):
    X_valid.append(validation_set_scaled[i-LOOKBACK:i, 0])
    y_valid.append(validation_set_scaled[i, 0])
X_valid, y_valid = np.array(X_valid), np.array(y_valid)
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

#Building the LSTM Neural Network
from keras.models import Sequential
from keras.layers import Dense, LSTM #, Dropout 
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(LSTM(units = NEURONS, return_sequences = True, kernel_initializer = KERNEL_INIT, activation = LSTM_ACTIVATION, input_shape = (X_train.shape[1], 1)))
#model.add(Dropout(DROPOUT))
model.add(LSTM(units = NEURONS, kernel_initializer = KERNEL_INIT, activation = LSTM_ACTIVATION, return_sequences = True))
#model.add(Dropout(DROPOUT))
model.add(LSTM(units = NEURONS, kernel_initializer = KERNEL_INIT, activation = LSTM_ACTIVATION, return_sequences = True))
#model.add(Dropout(DROPOUT))
model.add(LSTM(units = NEURONS, kernel_initializer = KERNEL_INIT, activation = LSTM_ACTIVATION))
#model.add(Dropout(DROPOUT))
model.add(Dense(units = 1, kernel_initializer = KERNEL_INIT, activation = DENSE_ACTIVATION))
model.compile(optimizer = OPTIMIZER, loss = LOSS_FUNCTION, metrics = [METRICS])
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = PATIENCE)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_mean_squared_error', mode = 'min', save_best_only = True)
history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_valid, y_valid))
model.save('LSTM_Bitcoin_Price_v3.h5')

#Visualizing Training and Validation Loss
plt.plot(history.history['loss'], color = 'blue', label = 'Training Loss')
plt.plot(history.history['val_loss'], color = 'orange', label = 'Validation Loss')
plt.legend(loc = 0)
plt.title('Training Loss v/s Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#Testing the model
inputs = dataset.iloc[int(TRAINING_SIZE*DATASET_SIZE)-LOOKBACK:DATASET_SIZE, 1:2].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range (LOOKBACK, (LOOKBACK+int(((1-TRAINING_SIZE)*DATASET_SIZE)))):
    X_test.append(inputs[i-LOOKBACK:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

# # Snippet from Nilesh: Rolling prediction using past prediction
# test_x = dataset['Price'][1500-60:1500].to_numpy().reshape(1,-1)
# test_x = sc.transform(test_x).reshape(1,-1,1)
# outputs = []

# for i in range (1500, 2400):
#     out = model.predict(test_x)
#     outputs.append(out[0])
#     test_x = np.concatenate([test_x[0,1:], out]).reshape(1,-1,1)
    
# outputs = sc.inverse_transform(outputs)

real_price = dataset.iloc[(int(TRAINING_SIZE*DATASET_SIZE)-LOOKBACK):DATASET_SIZE, 1:2].values
real_price_lookback = real_price[LOOKBACK:-1 ,:]

#Evaluating the model
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
print('R^2 Score: %.3f' % r2_score(real_price_lookback, predicted_price))
print('RMSE: %.3f' % sqrt(mean_squared_error(real_price_lookback, predicted_price)))

#Plotting the prediction results
#plt.plot(real_price, color = 'red', label = 'Actual Bitcoin Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Bitcoin Price', linestyle = '--')
plt.plot(real_price_lookback, color = 'red', label = 'Actual Bitcoin Price')
plt.legend(loc = 0)
plt.title('Bitcoin Price Prediction Using LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.xticks([0, 200, 400, 600], ['28-11-2017', '16-06-2018', '02-01-2019', '21-07-2019'])
plt.show()

#Visualizing the deviation of predicted price from actual price
delta = []
for i in range (0, int(((1-TRAINING_SIZE)*DATASET_SIZE))):
    delta.append((real_price_lookback[i, 0]-predicted_price[i, 0])/(real_price_lookback[i, 0]))

plt.plot(delta, color ='blue', label = 'Delta')
plt.xlabel('Time')
plt.ylabel('Deviation')
plt.title('Deviation of Predicted Price from Actual Price (Delta)')
plt.legend(loc = 0)
plt.xticks([0, 200, 400, 600], ['28-11-2017', '16-06-2018', '02-01-2019', '21-07-2019'])
plt.show()

#Optimizing the model through hyperparameter tuning
"""
from sklearn.model_selection import GridSearchCV
parameters = [{'units': [30, 60, 90, 120], 'activation': ['tanh']},
              {'units': [30, 60, 90, 120], 'activation': ['sigmoid']},
              {'units': [30, 60, 90, 120], 'activation': ['relu']}
        ]
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_mse = grid_search.best_score_
best_parameters = grid_search.best_params_
"""
