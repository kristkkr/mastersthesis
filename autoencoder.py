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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback

from figures import create_simple_reconstruction_plot, save_plot_loss_history, save_to_directory


class Autoencoder:
    def __init__(self, path_results):
        self.input_shape = (1920,2560,3)
        self.model = None
        self.path_results = path_results
        
        
    def create_autoencoder(self):
        # conv layer parameters
        conv_kernel_size1 = 3
        conv_strides1 = 2
        conv_kernel_size1 = 5
        conv_strides2 = (3,4)
        conv_strides3 = (4,4)
        
        filters = [8,16,32,64,128,256,512]
        
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
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size1, strides=conv_strides2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)  
        """
        x = Conv2D(filters=filters[7], kernel_size=conv_kernel_size1, strides=conv_strides2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        
        ### BOTTLENECK ###    
        x = Conv2DTranspose(filters=filters[6], kernel_size=conv_kernel_size1, strides=conv_strides2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        """
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides2, activation = 'relu', padding = 'same')(x)
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
        
    def train(self, dataset, epochs, batch_size): #val_split
        """
        Train by the use of train_on_batch().
        Uses Dataset.load_batch()
        Callbacks are not used.
        
        """
        
        ds = dataset
        np.save(self.path_results+'data_timestamp_list_train', ds.timestamp_list_train)
        np.save(self.path_results+'data_timestamp_list_val', ds.timestamp_list_val)
        assert(batch_size % ds.images_per_timestamp == 0)
        
        batches_per_epoch = len(ds.timestamp_list)*ds.images_per_timestamp//batch_size
        train_batches = len(ds.timestamp_list_train)*ds.images_per_timestamp//batch_size
        val_batches = len(ds.timestamp_list_val)*ds.images_per_timestamp//batch_size
        
        train_val_ratio = train_batches//val_batches
        #print(train_val_ratio)
        
        loss_history = LossHistory()
        loss_history.on_train_begin()
        #loss_train, loss_val = [],[]
        
        print('Total, train and val batches per epoch:', batches_per_epoch, train_batches, val_batches)
        print('Batch size:', batch_size)

        for epoch in range(epochs):
            print('Epoch '+str(epoch+1)+'/'+str(epochs))
            val_batch = 0
            
            # train
            for train_batch in range(train_batches):
                print('Training batch '+str(train_batch+1)+'/'+str(train_batches)+'. ', end='')
                
                
                x_batch = ds.load_batch(ds.timestamp_list[train_batch:train_batch+batch_size//ds.images_per_timestamp])
                loss = self.model.train_on_batch(x_batch,x_batch)
                #loss_train.append(loss)
                loss_history.on_train_batch_end(loss)
                print('Training loss: '+str(loss))
                # validate
                if (train_batch+1) % train_val_ratio == 0:           
                    print('Validate batch '+str(val_batch+1)+'/'+str(val_batches)+'. ', end='')
                    
                    x_batch = ds.load_batch(ds.timestamp_list[val_batch:val_batch+batch_size//ds.images_per_timestamp])
                    loss = self.model.test_on_batch(x_batch,x_batch)
                    #loss_val.append(loss)
                    loss_history.on_val_batch_end(loss)                    
                    print('Validate loss: '+str(loss))    
                    
                    val_batch += 1
                    
                    save_to_directory(self, loss_history, epoch, (train_batch+1), train_val_ratio, model_freq=train_val_ratio*10, loss_freq=train_val_ratio, n_move_avg=1)
                    #save_plot_loss_history(self.path_results, loss_history, train_val_ratio, n=1)
                

    def train_on_generator(self, dataset, epochs, batch_size): #split_frac removed from arguments
        """
        Train by the use of fit_generator(). 
        Allows for keras callbacks.
        
        """        
        ds = dataset
        
        assert(batch_size % ds.images_per_timestamp == 0)
                
        batches_per_epoch = len(ds.timestamp_list_train)//(batch_size//ds.images_per_timestamp)
        val_batches = len(ds.timestamp_list_val)//(batch_size//ds.images_per_timestamp)
        
        print('Train and val batches per epoch:',batches_per_epoch,val_batches)
        print('Batch size:', batch_size)
        #val_loss=0
        
        callback_list = [EarlyStopping(monitor='val_loss', patience=10), 
                         ModelCheckpoint(self.path_results+'model.hdf5',monitor='val_loss', verbose=1, save_best_only=False, period=0.01),
                         TensorBoard(log_dir=self.path_results+'log/./logs', batch_size=batch_size, write_images=True, )]
        
        
           
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
        

class LossHistory(Callback):
    def on_train_begin(self, log={}):
        self.train_loss = []
        self.val_loss = []

    def on_train_batch_end(self, loss):
        self.train_loss.append(loss)

    def on_val_batch_end(self, loss):
        self.val_loss.append(loss)
        
            
if __name__ == "__main__":
    
    from datahandler import Dataset

    # initialize model
    path_results = '/home/kristoffer/Documents/mastersthesis/results/ex2/'
    ae = Autoencoder(path_results)
    ae.create_autoencoder()
    ae.model.summary()
    """ 
    # initialize data
    ds = Dataset()
    ds.get_timestamp_list(sampling_period=60*6, randomize=False)
    print('Length of timestamp_list:',len(ds.timestamp_list))
    split_frac = (0.8,0.2,0.0)
    ds.split_list(split_frac)
    """
    #[(0,1), (0,2), (1,1), (2,1), (3,1)]
    ds = Dataset(cams_lenses = [(1,1),(3,1)])
    ds.read_timestamps_file('datasets/all/interval_60sec/timestamps')
    
    ds.split_list(split_frac = (0.9,0.1,0.0), shuffle_order=True)
    # hyperparameters
    epochs = 2
    batch_size = 8
    
    ae.train(ds, epochs, batch_size)
    #ae.train_on_generator(ds, epochs, batch_size)
    
    #ae.model = load_model(ae.path_results+'model.hdf5')
    #ae.reconstruct(ds, images_per_figure=12, steps=1)
    
    
    

