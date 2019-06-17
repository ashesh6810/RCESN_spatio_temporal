#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
#import json
import matplotlib.pyplot as plt
from numpy import genfromtxt
#from eccodes import *

np.random.seed(42)

dataf = pd.read_csv('3tier_lorenz_v3.csv',header=None)
data = np.array(dataf)
print(np.shape(data))


# In[ ]:


# global variables
shift_k = 0
#shift_k=int(shift_k)
res_params = {
             'train_length': 500000,
             'predict_length': 2000
              }


# In[ ]:


# train reservoir
train = data[shift_k:shift_k+res_params['train_length'],:]
label = data[1+shift_k:1+shift_k+res_params['train_length'],:]
print('np.shape(train)', np.shape(train))
print('np.shape(label)', np.shape(label))
#print(train)
#print(label)


# In[ ]:


y_train = label - train
x_train = train
print('np.shape(y_train)', np.shape(y_train))
print('np.shape(x_train)', np.shape(x_train))


# In[ ]:


model = Sequential()
model.add(Dense(8, input_dim=8, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(8, activation='tanh'))

model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
model.fit(x_train, y_train,nb_epoch=200,batch_size=128,validation_split=0.2)
model.save_weights("./weights_Shift"+str(shift_k)+"K")


# In[ ]:

print ('Read reference state')
# Load the reference state
ref_state = np.transpose(data[shift_k+res_params['train_length']:shift_k+res_params['train_length']+res_params['predict_length'],:])
#print('np.shape(ref_state)',np.shape(ref_state))

train_y = data[1+shift_k:1+shift_k+res_params['train_length'],:]
#print('np.shape(train_y)',np.shape(train_y))

n_dummy = np.shape(ref_state)
n_forecasts = 1
n_steps =  n_dummy[1]

fore_state = np.zeros((n_forecasts*(n_steps+1),8))
state = np.zeros(8)
state_n = np.zeros((1,8))

out0 = np.zeros((8,1))
out1 = np.zeros((8,1))
out2 = np.zeros((8,1))
out3 = np.zeros((8,1))


# In[32]:

# Get the last point from the training as the starting point of forcasting
state[:] = train_y[res_params['train_length']-1,:]
fore_state[0,:] = state[:]
for j in range(n_steps):
    out3=out2
    out2=out1
    state_n[0,:] = state

    out1 = model.predict(state_n,batch_size=1)
    if j==0:
        out0 = out1
    if j==1:
        out0 = 1.5*out1-0.5*out2
    if j>1:
        out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
    state[:] = state[:] + out0
    fore_state[1*(0)+j+1,:] = state[:]


# In[33]:

np.savetxt('output_ANN'+ 'shift'+str(shift_k)+ 'trainN' + str(res_params['train_length'])+'.csv',fore_state,delimiter=',')
np.savetxt('truth_ANN'+ 'shift'+str(shift_k)+ 'trainN' + str(res_params['train_length'])+'.csv',ref_state,delimiter=',')
print('done')
