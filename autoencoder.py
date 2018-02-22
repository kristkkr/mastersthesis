#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer
"""

from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class Autoencoder:
    def __init__(self):
        self.input_shape = (1920,2560,3)
        self.model = []
        
       
    def create_autoencoder(self):
        """
        Creates the autoencoder.
        Could be called in __init__ ?
        """

        # conv layer parameters
        conv_kernel_size1 = 4
        conv_strides1 = 2
        #conv_kernel_size2 = 4
        #conv_strides2 = (3,4)
        
        filters = [4,8,16,32,64,128,256,512]
        
        input_image = Input(shape=self.input_shape)
        
        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(input_image)
        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        ### BOTTLENECK ###    
        x = Conv2DTranspose(filters=filters[0], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = Conv2DTranspose(filters=3, kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'sigmoid', padding = 'same')(x)
    
        autoencoder = Model(input_image, x)
        autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')
    
        self.model = autoencoder
        
    def train(self, dataset, epochs, batch_size, val_split):
        """
        Train by the use of train_on_batch().
        Uses Dataloader.load_batch()
        Callbacks are not used.
        
        NOT UPDATED
        """
        
        ds = dataset
        
        batches_per_epoch = len(ds.timestamp_list)//(batch_size//12)
        train_batches = int((1-val_split)*batches_per_epoch)
        
        for epoch in range(epochs):
            print('Epoch', epoch+1, '/',epochs)
            
            # train
            for train_batch in range(train_batches):
                print('Training batch '+str(train_batch+1)+'/'+str(train_batches)+'. Total batches '+str(batches_per_epoch))
                print(ds.timestamp_list[train_batch:train_batch+batch_size//12])
                
                x_batch = ds.load_batch(ds.timestamp_list[train_batch:train_batch+batch_size//12])
                                
                self.model.train_on_batch(x_batch,x_batch)
            
            # validate
            for val_batch in range(train_batches,batches_per_epoch):
                print('ValBatch', val_batch+1, '/','total batches', batches_per_epoch)
                
                x_batch = ds.load_batch(ds.timestamp_list[val_batch:val_batch+batch_size//12])

                self.model.test_on_batch(x_batch,x_batch)
                

    def train_generator(self, dataset, epochs, batch_size, val_split):
        """
        Train by the use of fit_generator(). 
        Uses Dataloader.generate_batches()
        Allows for callbacks.
        
        Status: needs validation
        """        
        ds = dataset
        
        assert(batch_size % ds.IMAGES_PER_TIMESTAMP == 0)
                
        batches_per_epoch = len(ds.timestamp_list)//(batch_size//ds.IMAGES_PER_TIMESTAMP) ## use len(ds.timestamp_list) instead of ds. size
        train_batches = int((1-val_split)*batches_per_epoch)
        val_batches = batches_per_epoch-train_batches
        
        #val_loss=0
        
        callback_list = [EarlyStopping(monitor='train_loss', patience=1), 
                         ModelCheckpoint('results/model.hdf5', verbose=1, save_best_only=True),
                         TensorBoard(log_dir='log/./logs', batch_size=batch_size, write_images=True)]
                        
        train_gen = ds.generate_batches(batch_size)
        h = self.model.fit_generator(train_gen, batches_per_epoch, epochs, callbacks = callback_list)  
        
            
if __name__ == "__main__":
    
    from datahandler import Dataset

    # initialize model
    ae = Autoencoder()
    ae.create_autoencoder()
    ae.model.summary()
    
    # initialize data
    ds = Dataset()
    ds.read_timestamps_file('timestamps2017-10-22-11-sampled')
    #ds.get_timestamp_list(randomize=True)
    #ds.size = len(ds.timestamp_list)
    
    # hyperparameters
    epochs = 100
    batch_size = 12*2
    val_split = 0.1
    ae.train_generator(ds, epochs, batch_size, val_split)

