#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer

STATUS:
    leaky relu does not import. 
"""
import numpy as np

from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, PReLU
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from figures import create_simple_reconstruction_plot


class Autoencoder:
    def __init__(self):
        self.input_shape = (1920,2560,3)
        self.model = None
        self.path_results = '/home/kristoffer/Documents/mastersthesis/results/'
        
        
    def create_autoencoder(self):
        # conv layer parameters
        conv_kernel_size1 = 3
        conv_strides1 = 2
        conv_kernel_size1 = 5
        conv_strides2 = (3,2)
        
        filters = [4,8,16,32,64,128,256,512]
        
        input_image = Input(shape=self.input_shape) # change to ds.IMAGE_SHAPE?
        
        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(input_image)
        #x = LeakyReLu(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[2], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[3], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        x = Conv2D(filters=filters[4], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)           
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)  
        x = Conv2D(filters=filters[7], kernel_size=conv_kernel_size1, strides=conv_strides2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        ### BOTTLENECK ###    
        x = Conv2DTranspose(filters=filters[6], kernel_size=conv_kernel_size1, strides=conv_strides2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=filters[4], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=filters[3], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=filters[2], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=filters[1], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        
        x = Conv2DTranspose(filters=filters[0], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
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
                

    def train_on_generator(self, dataset, epochs, batch_size, split_frac):
        """
        Train by the use of fit_generator(). 
        Uses Dataloader.generate_batches()
        Allows for callbacks.
        
        """        
        ds = dataset
        
        assert(batch_size % ds.images_per_timestamp == 0)
                
        batches_per_epoch = len(ds.timestamp_list_train)//(batch_size//ds.images_per_timestamp)
        val_batches = len(ds.timestamp_list_val)//(batch_size//ds.images_per_timestamp)
        
        print('Train and val batches per epoch:',batches_per_epoch,val_batches)
        
        #val_loss=0
        
        callback_list = [EarlyStopping(monitor='val_loss', patience=10), 
                         ModelCheckpoint(self.path_results+'model.hdf5',monitor='val_loss', verbose=1, save_best_only=False),
                         TensorBoard(log_dir='log/./logs', batch_size=batch_size, write_images=True)]
        
        
           
        h = self.model.fit_generator(generator = ds.generate_batches(ds.timestamp_list_train, batch_size), 
                                     steps_per_epoch = batches_per_epoch, 
                                     epochs = epochs, 
                                     callbacks = callback_list, 
                                     validation_data = ds.generate_batches(ds.timestamp_list_val, batch_size), 
                                     validation_steps = val_batches)  
    
    def reconstruct(self, dataset, images_per_figure, steps):
        
        ds = dataset
        
        generator = ds.generate_batches(ds.timestamp_list_val, images_per_figure)
        #original,_ = next(generator)
        
        
        originals = np.empty((steps*images_per_figure,)+ds.IMAGE_SHAPE, np.float32)
        reconstructions = originals
        
        for step in range(steps):
            originals[step*images_per_figure:(step+1)*images_per_figure,:,:,:], _ = next(generator)
            reconstructions[step*images_per_figure:(step+1)*images_per_figure,:,:,:] = self.model.predict_on_batch(originals[step*images_per_figure:(step+1)*images_per_figure,:,:,:])
        """
        reconstructions = self.model.predict_generator(generator = ds.generate_batches(ds.timestamp_list_val, images_per_figure),
                                                       steps = steps,
                                                       verbose=1)
        """
        #originals = originals*255.0 #, int(reconstructions*255)
        
        n_images,_,_,_ = originals.shape
        n_fig = n_images//images_per_figure
        for i in range(n_fig):
            plot = create_simple_reconstruction_plot(originals[i*images_per_figure:(i+1)*images_per_figure,:,:,:], reconstructions[i*images_per_figure:(i+1)*images_per_figure,:,:,:], images_per_figure)
            plot.savefig(self.path_results+'output'+str(i)+'.jpg')
        print('Plots saved')
        
        
            
if __name__ == "__main__":
    
    from datahandler import Dataset

    # initialize model
    ae = Autoencoder()
    #ae.create_autoencoder()
    #ae.model.summary()
    """ 
    # initialize data
    ds = Dataset()
    ds.get_timestamp_list(sampling_period=60*6, randomize=False)
    print('Length of timestamp_list:',len(ds.timestamp_list))
    split_frac = (0.8,0.2,0.0)
    ds.split_list(split_frac)
    """
    #[(0,1), (0,2), (1,1), (2,1), (3,1)]
    ds = Dataset(cams_lenses = 'all')
    ds.read_timestamps_file('datasets/all/interval_60sec/timestamps')
    split_frac = (0.8,0.2,0.0)
    ds.split_list(split_frac, shuffle_order=True)
    # hyperparameters
    epochs = 100
    batch_size = 8
    
    
    #ae.train_on_generator(ds, epochs, batch_size, split_frac)
    
    ae.model = load_model(ae.path_results+'model.hdf5')
    ae.reconstruct(ds, images_per_figure=12, steps=1)
    
    
    

