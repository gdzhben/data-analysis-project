#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras import regularizers


# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()


# In[4]:


x = boston.data
x.shape


# In[5]:


y = boston.target


# In[6]:


print(x)


# In[7]:


x[0]


# In[8]:


print(y)


# In[9]:


['{:f}'.format(x) for x in x[0]]


# In[10]:


type(boston.data)


# In[11]:


len(x[:,1])


# In[12]:


x.shape


# In[13]:


# respounder = x[:,1]


# In[14]:


# predictors = np.delete(x, 0, 1) 


# In[15]:


# print(predictors)


# In[16]:


# predictors.shape


# In[17]:


# x=np.reshape(x, (-1,1))
# y=np.reshape(y, (-1,1))
# scaler = MinMaxScaler()
# print(scaler.fit(x))
# print(scaler.fit(y))
# xscale=scaler.transform(x)
# yscale=scaler.transform(y)


# In[18]:

prar = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
for i in prar:
    model = Sequential()
    model.add(Dense(13, input_dim=13,use_bias=True, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, activation='relu',use_bias=True, kernel_regularizer=regularizers.l1(l=i)))
    model.add(Dense(1,use_bias=True, activation='linear'))
    model.summary()


    # In[19]:


    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])


    # In[20]:


    history = model.fit(x,y, epochs=1000, batch_size=50,  verbose=1, validation_split=0.2)


    # In[21]:


    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


    # In[22]:


    Xnew = np.array([['0.006320','18.000000',
    '2.310000',
    '0.000000',
    '0.538000',
    '6.575000',
    '65.200000',
    '4.090000',
    '1.000000',
    '296.000000',
    '15.300000',
    '396.900000',
    '4.980000']])
    ynew=model.predict(Xnew)
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))


# In[ ]:





# In[ ]:





# In[ ]:




