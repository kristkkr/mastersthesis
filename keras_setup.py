#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:23:40 2018

This file serves the purpose of exploring keras setup.

@author: kristoffer
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
#from keras.optimizers import Adadelta
from keras.regularizers import l1, l2
from keras.datasets import mnist


class Autoencoder:
    def __init__(self):
        #self.name = 'kriss sin autoencoder' 
        self.model = []
        
    def create_autoencoder(self, input_shape):
            
        #input_dim = np.prod(input_shape)
        encoding_dim = 32
        
        input_img = Input(shape=input_shape)
        x = Conv2D(filters=4, kernel_size=2, strides=2, activation='relu', padding='same')(input_img)
        x = Conv2D(filters=8, kernel_size=2, strides=2, activation='relu', padding='same')(x)
        conv_shape = x._keras_shape[1:]
        conv_dim = np.prod(conv_shape)
        
        x = Flatten()(x)
        x = Dense(encoding_dim, activation='relu')(x)
        #BOTTLENECK#
        x = Dense(conv_dim, activation='relu')(x)
        x = Reshape(conv_shape)(x)
        x = Conv2DTranspose(filters=4, kernel_size=2, strides=2, activation='sigmoid', padding='same')(x)
        x = Conv2DTranspose(filters=1, kernel_size=2, strides=2, activation='sigmoid', padding='same')(x)
        
                
        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer='adadelta', loss='mae')
    
        return autoencoder
    
    def train(self, x_train, x_val):
        
        x_train = x_train.astype('float32') / 255.
        x_val = x_val.astype('float32') / 255.
        x_train = np.expand_dims(x_train,3)
        x_val = np.expand_dims(x_val,3)
                        
        self.model.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, x_val))
        return self.model
    
    
def main():
    (x_train, _), (x_val, _) = mnist.load_data()
    
    autoencoder = Autoencoder()
    autoencoder.model = autoencoder.create_autoencoder(input_shape = x_train.shape[1:]+(1,))
    print(autoencoder.model.summary())
    autoencoder.train(x_train, x_val)
    return

main()

    
    
    
    