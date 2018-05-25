#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer

"""
import os

import numpy as np
from scipy.ndimage import imread, label, find_objects
from scipy.misc import imresize
#import scipy.ndimage
import json


from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, BatchNormalization, Lambda, LeakyReLU, Flatten, Reshape, add, Activation
from keras.models import Model#, Sequential
from keras.callbacks import Callback#, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend import tf
from keras.optimizers import Adam, Adadelta
#from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator

from group_norm import GroupNormalization

import xml.etree.ElementTree as ET


from figures import create_reconstruction_plot, create_reconstruction_plot_single_image, plot_loss_history, insert_leading_zeros, show_detections

class AutoencoderModel:

    IMAGE_SHAPE = (144,192,3)#(192,256,3) #
    
    def create_ae_dilated(self):
        
        k1,k2, = 5,3
        dk1 = 4
        
        s1, s2 = 1, 2
                        
        filters = [32,64,128,256] 

        dilation = [2,4,8,16]
                
        input_image = Input(shape=self.IMAGE_SHAPE) 
        x = input_image
        resized_shape = (192,256) #(384, 512)	
        resize_method = tf.image.ResizeMethod.BICUBIC
        
        #x = Lambda(lambda image: tf.image.resize_images(image, resized_shape, method = resize_method))(input_image)	
        
        x = Conv2D(filters=filters[1], kernel_size=k1, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[2], kernel_size=k2, strides=s2, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=filters[2], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=filters[3], kernel_size=k2, strides=s2, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=filters[3], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=filters[3], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        for d in dilation:
            x = Conv2D(filters=filters[3], kernel_size=k1, strides=s1, activation='relu', padding = 'same', dilation_rate=d)(x)
            x = BatchNormalization()(x)            
        
        x = Conv2D(filters=filters[3], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=filters[3], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2DTranspose(filters=filters[2], kernel_size=dk1, strides=s2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[2], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2DTranspose(filters=filters[1], kernel_size=dk1, strides=s2, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[0], kernel_size=k2, strides=s1, activation='relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=3, kernel_size=k2, strides=s1, activation='sigmoid', padding = 'same')(x)        
        #x = Lambda(lambda image: tf.image.resize_images(image, self.dataset.IMAGE_SHAPE[:2], method = resize_method))(x)
        #x = Activation('sigmoid')(x)
        #x = add([input_image,x])
        
        autoencoder = Model(input_image, x)
        optimizer = Adadelta()
        autoencoder.compile(optimizer=optimizer, loss='mean_absolute_error')
    
        self.model = autoencoder

    def create_fca(self):
        
        # conv layer parameters
        conv_kernel_size = 5
        conv_strides = 2
                        
        filters = [8,16,32,64,128,256,512] 
        
        conv_layers_pure = 2
        dilation = 1
        
        input_image = Input(shape=self.IMAGE_SHAPE) 
        skip0 = input_image
        x = input_image
        
        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)        

        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip2 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip4 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[4], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)     
        
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip6 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)      
        
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        for i in range(conv_layers_pure):
            x = Conv2D(filters=filters[-1], kernel_size=conv_kernel_size, strides=1, padding = 'same', dilation_rate=dilation)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
        ### BOTTLENECK ###            
        for i in range(conv_layers_pure):
            x = Conv2DTranspose(filters=filters[-1], kernel_size=conv_kernel_size, strides=1, activation = 'relu', padding = 'same', dilation_rate=dilation)(x)
            x = BatchNormalization()(x)
        
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = add([skip6, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)
        
        x = Conv2DTranspose(filters=filters[4], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = add([skip4, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = add([skip2, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=3, kernel_size=conv_kernel_size, strides=conv_strides, activation = 'sigmoid', padding = 'same')(x)
        
        x = add([skip0, x])
        
        autoencoder = Model(input_image, x)
        optimizer = Adam(lr=0.001)
        autoencoder.compile(optimizer=optimizer, loss='mean_absolute_error')
    
        self.model = autoencoder
    

    def create_ce(self):
        """	
        Model inspired by the paper Context Encoders.	
        """	
        # conv layer parameters
        conv_kernel_size = 5
        conv_strides = 2
        
        conv_layers_pure = 0
        dilation = 1

        filters = [128, 256, 512, 1024] #64, 128, 	
       	
        input_image = Input(shape=self.IMAGE_SHAPE)	
        x = input_image
        skip0 = input_image

        resized_shape = (384, 512)	
        resize_method = tf.image.ResizeMethod.BILINEAR
        #x = Lambda(lambda image: tf.image.resize_images(image, resized_shape, method = resize_method))(input_image)	
        
        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)        

        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip2 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip3 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip4 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        """
        x = Conv2D(filters=filters[4], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        skip5 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)     
        
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        #skip6 = x
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)      
        
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        """
        for i in range(conv_layers_pure):
            x = Conv2D(filters=filters[-1], kernel_size=conv_kernel_size, strides=1, padding = 'same', dilation_rate=dilation)(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
        ### BOTTLENECK ###            
        for i in range(conv_layers_pure):
            x = Conv2DTranspose(filters=filters[-1], kernel_size=conv_kernel_size, strides=1, activation = 'relu', padding = 'same', dilation_rate=dilation)(x)
            x = BatchNormalization()(x)
        """
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = add([skip6, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)
        
        x = Conv2DTranspose(filters=filters[4], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = add([skip5, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)
        
        x = Conv2DTranspose(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = add([skip4, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)
        """
        x = Conv2DTranspose(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        #x = add([skip3, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        #x = add([skip2, x])
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU(alpha=0.0)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=3, kernel_size=conv_kernel_size, strides=conv_strides, activation = 'sigmoid', padding = 'same')(x)
        
        #x = Lambda(lambda image: tf.image.resize_images(image, self.dataset.IMAGE_SHAPE[:2], method = resize_method))(x)
        #x = add([skip0, x])
        
        autoencoder = Model(input_image, x)
        optimizer = Adam(lr=0.001)
        autoencoder.compile(optimizer=optimizer, loss='mean_absolute_error')
    
        self.model = autoencoder
        
    def channel_wise_dense_layer_tensorflow(self, x, name): # bottom: (7x7x512)
        """ 
        Based on Context Encoder implementation in TensorFlow.
        """
        _, height, width, n_feat_map = x.get_shape().as_list()
        input_reshape = tf.reshape( x, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,49,49)
                    initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.matmul(input_transpose, W)
        
        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape
    
    """
    def channel_wise_dense_layer(self,x):
       # pseudo code that might work directly in keras
        
        _, height, width, n_filters = K.int_shape(x)
        x = Reshape((height*width, n_filters))(x)
        
        for f in range(n_filters):
            x[:,:,f] = Dense(units = height*width)(x[:,:,f]) must slice differently
    """
   
class Autoencoder(AutoencoderModel):
    def __init__(self, path_results, dataset, dataset_reconstruct=None): 
        #self.input_shape = dataset.IMAGE_SHAPE
        self.model = None #self.create_model()
        self.path_results = path_results
        self.dataset = dataset # datasets train and val for training and train+val loss
        self.dataset_reconstruct = dataset_reconstruct # dataset val for validation/reconstruction during training.
                        
    def train(self, epochs, batch_size, inpainting_grid=None, single_im=False):
        """
        Train by the use of train_on_batch() and Dataset.load_batch()
        'single_im=True' if deep-image-prior. only training, no validation.
        fails if more than one of cams_lenses.
        """
        ds = self.dataset

        batches_per_epoch = len(ds.timestamp_list)*ds.images_per_timestamp//batch_size
        train_batches = len(ds.timestamp_list_train)*ds.images_per_timestamp//batch_size
        val_batches = len(ds.timestamp_list_val)*ds.images_per_timestamp//batch_size
        
        #if not inpainting_grid==None:
        #    batches_per_epoch, train_batches, val_batches = batches_per_epoch*np.product(inpainting_grid), train_batches*np.product(inpainting_grid), val_batches*np.product(inpainting_grid)
        
        train_val_ratio = int(round(train_batches/val_batches)) #train_batches//val_batches #
                
        loss_history = LossHistory()
        loss_history.on_train_begin(self.path_results)
        loss = 0
        
        if not inpainting_grid==None:
            print('Inpainting autoencoder')
        else:
            print('Regular reconstructing autoencoder')
        
        print('Total, train and val batches per epoch:', batches_per_epoch, train_batches, val_batches)
        print('Batch size:', batch_size)
        failed_im_load = []

        self.indexing_iterator = None # timestamps per batch?
        #if not inpainting_grid==None:
        #    self.indexing_iterator = 1
        #else:
        self.indexing_iterator = batch_size//ds.images_per_timestamp
        
        
        for epoch in range(epochs):
            print('Epoch '+str(epoch+1)+'/'+str(epochs))
            val_batch = 0 # keeps track of what number of val_batch we're currently at
            train_timestamp_index = 0 # keeps track of index in timestamp_list_train
            val_timestamp_index = 0 # keeps track of index in timestamp_list_val
            
            # train
            for train_batch in range(train_batches):
                print('Training batch '+str(train_batch+1)+'/'+str(train_batches)+'. ', end='')
                if train_batch % 6000 ==0:
                    self.model.optimizer.lr = self.model.optimizer.lr*0.1  
                
                x = []
                x, failed_im_load = ds.load_batch(ds.timestamp_list_train[train_timestamp_index:train_timestamp_index+self.indexing_iterator], failed_im_load)
                
                if x == []:
                    continue
                
                if not inpainting_grid==None:
                    #x_batch_masked, x_batch = ds.mask_batch(x, inpainting_grid)
                    x_batch_masked, x_batch = ds.mask_batch_randomly(x, inpainting_grid)
                    loss = self.model.train_on_batch(x_batch_masked, x_batch)
                else:
                    loss = self.model.train_on_batch(x, x)
                
                loss_history.on_train_batch_end(loss)
                print('Training loss: '+str(loss))
                
                train_timestamp_index += self.indexing_iterator
                
                # validate
                if (not single_im) and (train_batch+1) % train_val_ratio == 0:           
                    print('Validate batch '+str(val_batch+1)+'/'+str(val_batches)+'. ', end='')
                    x = []
                    x, failed_im_load = ds.load_batch(ds.timestamp_list_val[val_timestamp_index:val_timestamp_index+self.indexing_iterator], failed_im_load)
                    #print(x.shape)
                    if x == []:
                        continue
                    if not inpainting_grid==None:
                        #x_batch_masked, x_batch = ds.mask_batch(x, inpainting_grid)
                        x_batch_masked, x_batch = ds.mask_batch_randomly(x, inpainting_grid)
                        loss = self.model.test_on_batch(x_batch_masked, x_batch)
                    else:
                        loss = self.model.test_on_batch(x, x)
                    
                    loss_history.on_val_batch_end(loss)                    
                    print('Validate loss: '+str(loss))    
                    
                    val_batch += 1
                    val_timestamp_index += self.indexing_iterator
                    
                self.save_to_directory(loss_history, 
                                       failed_im_load, 
                                       epoch, train_batch, 
                                       train_timestamp_index-self.indexing_iterator, 
                                       val_timestamp_index-self.indexing_iterator, 
                                       train_val_ratio, 
                                       model_freq=2*100*train_val_ratio, 
                                       loss_freq=train_val_ratio,  
                                       reconstruct_freq_train=1000000, 
                                       reconstruct_freq_val=20*train_val_ratio, 
                                       inpainting_grid=inpainting_grid, 
                                       single_im=single_im)
                
    def save_to_directory(self, loss_history, failed_im_load, epoch, batch, train_timestamp_index, val_timestamp_index, train_val_ratio, model_freq, loss_freq, reconstruct_freq_train, reconstruct_freq_val, inpainting_grid=None, single_im=False):
        """
        Saves results to directory during training.
        """
        epoch_str = insert_leading_zeros(epoch+1,5)
        batch_str = insert_leading_zeros(batch+1,6)
        
        if single_im:
            freq_counter = epoch
        else:
            freq_counter = batch
        
        if (freq_counter+1)%loss_freq == 0:
            np.save(self.path_results+'loss_history_train', loss_history.train_loss)
            np.save(self.path_results+'loss_history_val', loss_history.val_loss)
            plot_loss_history(self.path_results, train_val_ratio, single_im, n=1)
            np.save(self.path_results+'failed_imageload_during_training', failed_im_load)
            with open(self.path_results+'failed_imageload_during_training.txt', 'w') as text: 
                print('Length:{}'.format(len(failed_im_load)), file=text)
                print('Timestamps and cam_lens of failed batches:\n{}'.format(failed_im_load), file=text)
    
        if (freq_counter+1)%model_freq == 0:
            try:
                self.model.save(self.path_results+'epoch'+epoch_str+'_batch'+batch_str+'.hdf5')
                print('Model saved')
            except:
                self.model.save_weights(self.path_results+'epoch'+epoch_str+'_batch'+batch_str+'.hdf5')
                print('Model weights saved')
            
        if (freq_counter+1)%reconstruct_freq_train == 0: #used when reconstucting training data
            self.test(dataset = self.dataset, what_data_split='train', timestamp_index=train_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid, single_im_batch=False)
        
        if (freq_counter+1)%reconstruct_freq_val == 0: #reconstruct_freq_val needs to be a multiplier of train_val_ratio
            self.test(dataset = self.dataset, what_data_split='val', timestamp_index=val_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid, single_im_batch=False)
            
            if not inpainting_grid == None:
                self.test(dataset = self.dataset, what_data_split='val', timestamp_index=val_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid, single_im_batch=True)
                       
            
    def test(self, dataset, what_data_split, timestamp_index, numb_of_timestamps, epoch, batch, inpainting_grid, single_im_batch):
        """
        Test autoencoder.model on images from the train, val or test dataset.
        Saves figure to file.
        To be used during training or after fully trained.
        'numb_of_timestamps'>1 when used after fully trained. NOT INCLUDED ATM. SHOULD PROBABLY BE NAMED AND USED AS 'numb_of_figures'.
        
        """
        #if timestamp_index > len(dataset.timestamp_list_train): 
        #    timestamp_index = np.random.randint(0,len(dataset.timestamp_list_train)) # needed since len(dataset) > len(dataset_reconstruct)
            
        if what_data_split == 'train':
            timestamps = dataset.timestamp_list_train[timestamp_index:timestamp_index+self.indexing_iterator]
        elif what_data_split == 'val':
            timestamps = dataset.timestamp_list_val[timestamp_index:timestamp_index+self.indexing_iterator]
        elif what_data_split == 'test':
            timestamps = dataset.timestamp_list_test[timestamp_index:timestamp_index+self.indexing_iterator]
        else:
            print('Invalid data argument, no reconstruction possible.')
        
        j = 0
        max_pred = 24
        while j < numb_of_timestamps:
            x,failed_im_load = dataset.load_batch(timestamps[j:j+self.indexing_iterator], failed_im_load=[])
            
            #if failed_im_load != []:
                #continue
            
            if not inpainting_grid==None:
                                
                if single_im_batch: # batch consist of one image
                    x_batch_masked, x_batch = dataset.mask_image(x[0], inpainting_grid)    # selects the first image in batch. 
                    x_batch_original_and_masked = np.concatenate((np.expand_dims(x_batch[0], axis=0), x_batch_masked), axis=0)
                    y_batch = self.model.predict_on_batch(x_batch_masked)
                    """
                    if len(x_batch_masked)>max_pred:
                        y_batch = np.copy(x_batch_masked)
                        y_batch *=0
                        for i in range(len(x_batch_masked)//max_pred):
                            y_batch[i*max_pred:(i+1)*max_pred] = self.model.predict_on_batch(x_batch_masked[i*max_pred:(i+1)*max_pred])
                    """
                    plot = create_reconstruction_plot_single_image(self, x_batch_original_and_masked, y_batch, inpainting_grid)                    
                    plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(j+1)+'-single.jpg')
                    print('Reconstruction single image inpainting saved')
                else: # same as below.
                    #x_batch_masked,_ = dataset.mask_batch(x, inpainting_grid)
                    x_batch_masked,_ = dataset.mask_batch_randomly(x, inpainting_grid)
                    y_batch = self.model.predict_on_batch(x_batch_masked)
                    plot = create_reconstruction_plot(self, x, y_batch, x_batch_masked)                    
                    plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(j+1)+'.jpg')
                    print('Reconstruction inpainting saved')                
            else:
                y_batch = self.model.predict_on_batch(x)
                plot = create_reconstruction_plot(self, x, y_batch)                    
                plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(j+1)+'.jpg')
                print('Reconstruction regular saved')            

            j += self.indexing_iterator

    
    def merge_inpaintings(self, y_batch, inpainting_grid):
        """
        Merges inpatinings in 'y_batch' to a single image. 
        MOVE TO FIGURES.
        """
        assert(len(y_batch)==np.prod(inpainting_grid))
        
        mask_shape = (self.dataset.IMAGE_SHAPE[0]//inpainting_grid[0], self.dataset.IMAGE_SHAPE[1]//inpainting_grid[1])

        inpainted = np.empty(self.dataset.IMAGE_SHAPE)
        
        x0, y0 = 0,0
        i = 0
        for x in range(inpainting_grid[1]):
            x = x*mask_shape[1]
            for y in range(inpainting_grid[0]):
                y = y*mask_shape[0]
                try:
                    inpainted[y:y+mask_shape[0],x:x+mask_shape[1]] = y_batch[i,y:y+mask_shape[0],x:x+mask_shape[1]]
                except:
                    break
                
                i += 1
                y0 = y0 + mask_shape[0]
            x0 = x0 + mask_shape[1]
        return inpainted

    def evaluate(self, inpainting_grid, visual):
        """
        Evaluate model on test data. Same procedure as for single_im_batch in test()
        could consider evaluating at resized image and not original. 
        """
        
        for threshold in range(50,160,1000): #should be the outermost loop
            metrics = {'box_tp': 0, 'box_fn': 0, 'recall':'NaN', 'pixel_tp':0, 'pixel_fp':0, 'pixel_precision':'NaN','cluster_tp':0, 'cluster_fp':0, 'cluster_precision':'NaN'}
            #threshold_array = threshold*np.ones((self.IMAGE_SHAPE[:2])+(,1),dtype=np.uint8).astype('float32') / 255.
            for filename in sorted(os.listdir(self.dataset.path_test))[49:50]: #self.dataset.path_test):
                if filename.endswith('.xml'):
                    image = imread(self.dataset.path_test+filename.replace('.xml','.jpg')).astype('float32') / 255.
                    
                    if image.shape != self.dataset.IMAGE_SHAPE:
                        image_lr = imresize(image,self.dataset.IMAGE_SHAPE,'bicubic')/ 255.
                        #reshaped = True
                    
                    x_batch_masked, _ = self.dataset.mask_image(image_lr, inpainting_grid)
                    y_batch = self.model.predict_on_batch(x_batch_masked)
                    #if reshaped:
                    #    y_batch = [imresize(im_pred,self.dataset.IMAGE_SHAPE_ORIGINAL,'bicubic') for im_pred in y_batch]
                    inpainted = self.merge_inpaintings(y_batch, inpainting_grid)
                    if inpainted.shape != image.shape:
                        inpainted = imresize(inpainted,image.shape,'bicubic')/ 255.
                    residual = np.mean(np.abs(np.subtract(image, inpainted)), axis=2) #mean of channels or RGB->grayscale conversion?
                    
                    #binary_map = np.greater(residual, threshold_array)
                    binary_map = residual > threshold/255. #*np.ones(residual.shape,dtype=np.uint8).astype('float32') / 255.
                    
                    box_tp, box_fn, recall = count_box_detections(binary_map, self.dataset.path_test+filename.replace('.jpg', '.xml'), self.dataset.IMAGE_SHAPE_ORIGINAL)# self.dataset.IMAGE_EXTENSION
                    pixel_tp, pixel_fp, pixel_precision, object_gt_map = count_pixel_detections(binary_map, self.dataset.path_test+filename.replace('.jpg', '.xml'))
                    cluster_tp, cluster_fp, cluster_precision = count_clustered_detections(binary_map, self.dataset.path_test+filename.replace('.jpg', '.xml'))
                    
                    metrics = update_metrics(metrics, box_tp, box_fn, pixel_tp, pixel_fp, cluster_tp, cluster_fp)
                    #print(object_gt_map.shape)
                    print(filename, threshold)
                    print(box_tp, box_fn,'recall=',recall)
                    print(cluster_tp, cluster_fp, 'cluster_precision=',cluster_precision)
                    print(pixel_tp, pixel_fp, 'pixel_precision=',pixel_precision)
                    
                    
                    if visual:
                        object_map = map_on_image(image, binary_map)
                        figure = show_detections(image, inpainted, residual, binary_map, object_map)
                        figure.savefig(self.path_results+'detections-threshold'+str(threshold)+'--'+filename.replace('.xml','.jpg'))
                        print('figsaved')
                        
                    with open(self.path_results+'metrics-threshold'+str(threshold)+'.txt', 'w') as fp:
                        json.dump(metrics,fp)

def update_metrics(metrics, box_tp, box_fn, pixel_tp, pixel_fp, cluster_tp, cluster_fp):
    metrics['box_tp'] += box_tp
    metrics['box_fn'] += box_fn
    try:
        metrics['recall'] = round(metrics['box_tp']/(metrics['box_tp']+metrics['box_fn']),4)
    except:
        metrics['recall'] = 'NaN'
    
    metrics['pixel_tp'] += pixel_tp
    metrics['pixel_fp'] += pixel_fp
    try:
        metrics['pixel_precision'] = round(metrics['pixel_tp']/(metrics['pixel_tp']+metrics['pixel_fp']),4)
    except:
        metrics['pixel_precision'] = 'NaN'
    
    metrics['cluster_tp'] += cluster_tp
    metrics['cluster_fp'] += cluster_fp
    try:
        metrics['cluster_precision'] = round(metrics['cluster_tp']/(metrics['cluster_tp']+metrics['cluster_fp']),4)
    except:
        metrics['cluster_precision'] = 'NaN'  
    return metrics
            
def count_pixel_detections(binary_map, gt_file):
    """counts pixel detections tp, tn, fp in binary predicted map with respect to ground truth files. use both background and object classes?"""
    y,x = binary_map.shape
    object_gt_map,_,background_gt_map,_ = read_gt_file(gt_file,(y,x))
    if binary_map.shape != object_gt_map.shape:
        object_gt_map = imresize(object_gt_map, binary_map.shape, interp='nearest')/255.
        background_gt_map = imresize(background_gt_map, binary_map.shape, interp='nearest')/255.
    gt_map = np.logical_or(object_gt_map,background_gt_map)
    
    assert(np.amax(binary_map)<=1)
    assert(np.amax(gt_map)<=1)
    
    tp, fp, fn, tn = 0,0,0,0
    for j in range(y):
        for i in range(x):
            if binary_map[j,i]==1:
                if gt_map[j,i] == 1:
                    tp +=1
                else:
                    fp +=1
            else:
                if gt_map[j,i] == 1:
                    fn +=1
                else:
                    tn +=1
    try:
        precision = round(tp/(tp+fp),4)
    except:
        precision = 'NaN'
    return tp, fp, precision, object_gt_map

def count_clustered_detections(binary_map, gt_file):
    """counts clustered detections tp, fp in binary predicted map with respect to ground truth files"""
    y,x = binary_map.shape
    object_gt_map,_,background_gt_map,_ = read_gt_file(gt_file,(y,x))
    
    if binary_map.shape != object_gt_map.shape:
        object_gt_map = imresize(object_gt_map, binary_map.shape, interp='nearest')/ 255.
        background_gt_map = imresize(background_gt_map, binary_map.shape, interp='nearest')/ 255.
        
    gt_map = np.logical_or(object_gt_map,background_gt_map)
    
    labeled_binary_map,_ = label(binary_map) #label each clustered detection
    
    object_slices = find_objects(labeled_binary_map)
    
    assert(np.amax(gt_map)<=1)

    tp, fp = 0,0
    for object_slice in object_slices:
        if gt_map[object_slice].any() == 1:
            tp +=1
        else:
            fp +=1

    try:
        precision = round(tp/(tp+fp),4)
    except:
        precision = 'NaN'
    return tp, fp, precision

def count_box_detections(binary_map, gt_file, IMAGE_SHAPE_ORIGINAL):
    """counts box detections. use object box class"""
    ### resoze binary map to large?
    #print(binary_map.shape)
    binary_map_copy = binary_map
    if binary_map_copy.shape != IMAGE_SHAPE_ORIGINAL:
        binary_map_copy = imresize(binary_map_copy, IMAGE_SHAPE_ORIGINAL, interp='nearest')/ 255.
    
    #print(binary_map.shape)
    #Image.fromarray(binary_map*255, 'L').show()
    #Image.fromarray(binary_map_copy, 'L').show()
    y,x = binary_map_copy.shape
    _, object_boxes,_,_ = read_gt_file(gt_file,(y,x))
    #binary_map, gt_map = binary_map/255, gt_map/255
    
    assert(np.amax(binary_map_copy)<=1)
    
    tp,fn= 0,0
    for box in object_boxes:
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        
        if binary_map_copy[ymin:ymax,xmin:xmax].any() == 1:
            tp +=1
        else:
            fn +=1
    try:
        recall = round(tp/(tp+fn),4)
    except:
        recall = 'NaN'
    return tp, fn, recall
    
from PIL import ImageDraw, Image

def read_gt_file(gt_file,shape):
    object_gt_map, object_boxes = read_bounding_box_category(gt_file, shape, 'object')
    background_gt_map, background_boxes = read_bounding_box_category(gt_file, shape, 'background')
    return object_gt_map, object_boxes, background_gt_map, background_boxes
    
def read_bounding_box_category(gt_file, shape, box_category):
    gt_im = Image.fromarray(np.zeros(shape, dtype=np.uint8)) 
    draw = ImageDraw.Draw(gt_im)
    tree = ET.parse(gt_file)
    root = tree.getroot()
    boxes = []
    for box_object in root.findall('object'):
        if box_object.find('name').text == box_category:
            for bbox in box_object.iter('bndbox'):
                box = []
                for child in bbox:
                    box.append(int(child.text))
        
                draw.rectangle(box,fill=255)
                boxes.append(box)
    gt_map = np.asarray(gt_im, dtype=np.uint8).astype('float32') / 255.
    
    return gt_map, boxes

def map_on_image(im, binary_map):
    image = np.copy(im)
    x,y,_ = image.shape
    mask_color = np.amax(image)
    for i in range(x):
        for j in range(y):
            if binary_map[i,j] == 1:
                #image[i,j,:] = [mask_color,0,0]
                image[i,j,0] = mask_color
                image[i,j,1] = 0
                image[i,j,2] = 0
    return image

def scale_range(array):
    """ scale range from [-1,1] to [0,1]"""
    return (array+1)/2
    
        
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

    
    path_test = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/test2/2017-10-22-11_11_09(1, 1).xml'
    #binary_map,_ = read_gt_file(path_test,(1920,2560))
    #binary_map = np.zeros((1920,2560))
    evaluate((3,4))
    #Image.fromarray(binary_map, mode='L').show()
    #tp, fp, fn, tn = count_pixel_detections(binary_map, path_test)
    #tp, fn = count_box_detections(binary_map, path_test)
    #print(tp,fn)
    
    
    
    """ 
    # initialize data
    ds = Dataset()
    ds.get_timestamp_list(sampling_period=60*6, randomize=False)
    print('Length of timestamp_list:',len(ds.timestamp_list))
    split_frac = (0.8,0.2,0.0)
    ds.split_list(split_frac)
    
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
    
    """
    

