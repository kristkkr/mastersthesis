#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer

"""
import numpy as np

from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback

from figures import create_reconstruction_plot, save_to_directory


class Autoencoder:
    def __init__(self, dataset, path_results, dataset_reconstruct=None): 
        self.input_shape = dataset.IMAGE_SHAPE
        self.model = None #self.create_model()
        self.path_results = path_results
        self.dataset = dataset
        #self.dataset_reconstruct = dataset_reconstruct # from now on only 'dataset' is used, since 'dataset' include train, val and test. 
        # alternative solution is to have a dataset instance for train and another instance for val.
            
    def create_model(self):
        
        # conv layer parameters
        conv_kernel_size1 = 5
        conv_strides0 = 1
        conv_strides1 = 2
        conv_strides2 = (3,2)
        conv_strides3 = (1,2)
        
        #filters = [8,16,32,64,128,256,512,1024] 
        #filters = [2**n for n in range(3,16)] 
        filters = [8, 8*3, 8*3**2, 8*3**3, 8*3**4, 8*3**5]
        
        input_image = Input(shape=self.input_shape) # change to ds.IMAGE_SHAPE?
        
        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(input_image)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[2], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[3], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   

        x = Conv2D(filters=filters[4], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)           
        """        
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)  
        
        ### BOTTLENECK ###            
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=filters[4], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        """     
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
        
    def train(self, epochs, batch_size): #val_split
        """
        Train by the use of train_on_batch() and Dataset.load_batch()
        """
        
        ds = self.dataset
        np.save(self.path_results+'data_timestamp_list_train', ds.timestamp_list_train)
        np.save(self.path_results+'data_timestamp_list_val', ds.timestamp_list_val)
        assert(batch_size % ds.images_per_timestamp == 0)
        
        batches_per_epoch = len(ds.timestamp_list)*ds.images_per_timestamp//batch_size
        train_batches = len(ds.timestamp_list_train)*ds.images_per_timestamp//batch_size
        val_batches = len(ds.timestamp_list_val)*ds.images_per_timestamp//batch_size
        
        train_val_ratio = round(train_batches/val_batches) # better to use round()?
        assert(train_val_ratio == round(len(ds.timestamp_list_train)/len(ds.timestamp_list_val)))
        
        loss_history = LossHistory()
        loss_history.on_train_begin(self.path_results)
        
        print('Total, train and val batches per epoch:', batches_per_epoch, train_batches, val_batches)
        print('Batch size:', batch_size)
        failed_im_load = []
        
        for epoch in range(epochs):
            print('Epoch '+str(epoch+1)+'/'+str(epochs))
            val_batch = 0 # keeps track of what number of val_batch we're currently at
            train_timestamp_index = 0 # keeps track of index in timestamp_list_train
            val_timestamp_index = 0 # keeps track of index in timestamp_list_val
            
            # train
            for train_batch in range(train_batches):
                print('Training batch '+str(train_batch+1)+'/'+str(train_batches)+'. ', end='')
                x_batch = []
                x_batch, failed_im_load = ds.load_batch(ds.timestamp_list_train[train_timestamp_index:train_timestamp_index+batch_size//ds.images_per_timestamp], failed_im_load)
                if x_batch == []:
                    continue
                
                loss = self.model.train_on_batch(x_batch,x_batch)
                loss_history.on_train_batch_end(loss)
                print('Training loss: '+str(loss))
                
                train_timestamp_index += batch_size//ds.images_per_timestamp
                
                # validate
                if (train_batch+1) % train_val_ratio == 0:           
                    print('Validate batch '+str(val_batch+1)+'/'+str(val_batches)+'. ', end='')
                    x_batch = []
                    x_batch, failed_im_load = ds.load_batch(ds.timestamp_list_val[val_timestamp_index:val_timestamp_index+batch_size//ds.images_per_timestamp], failed_im_load)
                    if x_batch == []:
                        continue
                    loss = self.model.test_on_batch(x_batch,x_batch)
                    loss_history.on_val_batch_end(loss)                    
                    print('Validate loss: '+str(loss))    
                    
                    val_batch += 1
                    val_timestamp_index += batch_size//ds.images_per_timestamp
                    
                save_to_directory(self, loss_history, failed_im_load, epoch, train_batch, train_val_ratio, model_freq=10*train_val_ratio, loss_freq=train_val_ratio, reconstruct_freq=5*train_val_ratio, n_move_avg=1)
                
                

    def test(self, what_data, numb_of_timestamps, epoch, batch):
        """
        Test autoencoder.model on images from the train, val or test dataset.
        Saves figure to file.
        To be used during training or after fully trained.
        """
        
        if what_data == 'train':
            timestamps = self.dataset.timestamp_list_train[:numb_of_timestamps+1]
        elif what_data == 'val':
            timestamps = self.dataset.timestamp_list_val[:numb_of_timestamps+1]
        elif what_data == 'test':
            timestamps = self.dataset.timestamp_list_test[:numb_of_timestamps+1]
        else:
            print('Invalid data argument, no reconstruction possible.')
        
        images_per_figure = len(self.dataset.cams_lenses)
            
        i = 0
        while i < numb_of_timestamps:
            # x_batch = ds.load_batch(ds.timestamp_list[val_batch:val_batch+batch_size//ds.images_per_timestamp])
            x_batch,_ = self.dataset.load_batch(timestamps[i:i+images_per_figure//self.dataset.images_per_timestamp], failed_im_load=[])
            y_batch = self.model.predict_on_batch(x_batch)
            plot = create_reconstruction_plot(x_batch, y_batch, images_per_figure)
            plot.savefig(self.path_results+'reconstruction-'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data+str(i+1)+'.jpg')
            i += images_per_figure//self.dataset.images_per_timestamp
            print('Reconstruction saved')
            

class LossHistory(Callback):
    def on_train_begin(self, path_results, log={}):
        try:
            self.train_loss = list(np.load(path_results+'loss_history_train.npy'))
            self.val_loss = list(np.load(path_results+'loss_history_val.npy'))
        except:
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
    
    
    

