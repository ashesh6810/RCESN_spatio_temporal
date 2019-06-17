#!/usr/bin/env python
# coding: utf-8

# ## LSTM for Lorenz96 
# - simple one layer architecture with X_{lookback-t:t} to X_{t+1}
import numpy as np
import pandas as pd
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# ## Read the raw data
# - 10 million samples from lorenz
# - only 8 X_i variables for t = 1 through 10 million

data = pd.read_csv('./3tier_lorenz_v3.csv',header=None)
print(data.shape)
data.head()


# ## Prepare train, validation and test sets for LSTMS
# - train is 500K (first 500K samples)
# - val is 10K (next 10K)
# - test  is 2K (next 2K)
# training size parameters
train_size = 500000
val_size = 10000
test_size = 2000
 
# lookback

lookback = 3

def make_LSTM_datasets(data,train_size,val_size,test_size):
    samples = train_size + val_size + test_size
    nfeatures = data.shape[1]
    sdata = np.transpose(data.values)[:,:samples]

    Xtemp = {}
    for i in range(lookback):    
        Xtemp[i] = sdata[:,i:samples-(lookback-i-1)]

    X = Xtemp[0]
    for i in range(lookback-1):
        X = np.vstack([X,Xtemp[i+1]])

    X = np.transpose(X)
    Y = np.transpose(sdata[:,lookback:samples])

    Xtrain = X[:train_size,:]
    Ytrain = Y[:train_size,:]

    Xval = X[train_size:train_size+val_size,:]
    Yval = Y[train_size:train_size+val_size,:]

    Xtest = X[train_size+val_size:,:]
    Ytest = Y[train_size+val_size:,:]

    # reshape inputs to be 3D [samples, timesteps, features] for LSTM

    Xtrain = Xtrain.reshape((Xtrain.shape[0], lookback, nfeatures))
    Xval = Xval.reshape((Xval.shape[0], lookback,nfeatures))
    Xtest = Xtest.reshape((Xtest.shape[0], lookback,nfeatures))
    print("Xtrain shape = ", Xtrain.shape, "Ytrain shape = ", Ytrain.shape)
    print("Xval shape =   ", Xval.shape, "  Yval shape =   ", Yval.shape)
    print("Xtest shape =  ", Xtest.shape, " Ytest shape =  ", Ytest.shape)
    
    return Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,nfeatures


# ## Setup and train the LSTM
# design network

# LSTM parameters
nhidden = 50

def make_and_train_LSTM_model(Xtrain,Ytrain,nfeatures,nhidden):
    model = Sequential()
    model.add(LSTM(nhidden, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dense(nfeatures))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(Xtrain, Ytrain, epochs=10, batch_size=72, verbose=2, shuffle=False)
    
    return model,history


# ## Test the model on test data

# test model on set aside test set (actually validation set)

def model_predict(model,Xval):
    ypred = np.zeros((Xval.shape[0],nfeatures))

    for i in range(Xval.shape[0]):  
        if i ==0:
            tt = Xval[0,:,:].reshape((1,lookback,nfeatures))
            ypred[i,:] = model.predict(tt) 
        elif i < lookback:
            tt = Xval[i,:,:].reshape((1,lookback,nfeatures))
            u = ypred[:i,:]
            tt[0,(lookback-i):lookback,:] = u
            ypred[i,:] = model.predict(tt)
        else:
            tt = ypred[i-lookback:i,:].reshape((1,lookback,nfeatures))
            ypred[i,:] = model.predict(tt)
    return ypred


# ## Run everything

Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,nfeatures = make_LSTM_datasets(data,train_size,val_size,test_size)
model,history = make_and_train_LSTM_model(Xtrain,Ytrain,nfeatures,nhidden)
ypred = model_predict(model,Xval)
np.savetxt('ypred.csv',ypred,delimiter=',')     
np.savetxt('ytest.csv',Yval,delimiter=',')





