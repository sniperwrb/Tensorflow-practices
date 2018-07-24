# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:00:07 2018

@author: hf6sb
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

a=[1,0,1,5,5,1,0,1,3,1,0,1,-2,3,2,-1,0,1,2,
   1,0,1,5,5,6,7,6,3,6,7,8,5,3,3,4,3,-2,2,1,1]
a=np.array(a)
m=7
l=len(a)
b=np.zeros((l,m))
for i in range(l):
    a[i]=a[i]%m
    b[i][a[i]]=1
c=np.reshape(b,(l,1,m))

model=keras.Sequential()
model.add(keras.layers.LSTM(m))
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
model.fit(c,b,batch_size=l,epochs=1000,shuffle=False, verbose=0)

test_loss, test_acc = model.evaluate(c,b,batch_size=l)
print('Test accuracy:', test_acc)
predictions = model.predict(c,batch_size=l)
a_pred=np.argmax(predictions,axis=1)
print(a);print(a_pred)