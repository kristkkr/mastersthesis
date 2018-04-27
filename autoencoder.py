#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer

"""
import numpy as np

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, BatchNormalization, Lambda, LeakyReLU, Flatten, Reshape
from keras.models import Model, Sequential
from keras.callbacks import Callback#, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend import tf
from keras import backend as K

from figures import create_reconstruction_plot, plot_loss_history, insert_leading_zeros

class AutoencoderModel:

    IMAGE_SHAPE = (1920,2560,3)

    def create_fca(self):
        
        # conv layer parameters
        conv_kernel_size1 = 5
        conv_strides1 = 2
                        
        #filters = [8,16,32,64,128,256,512,1024] 
        #filters = [2**n for n in range(3,16)] 
        filters = [8, 8*3, 8*3**2, 8*3**3, 8*3**4, 8*3**5, 8*3**6]
        
        input_image = Input(shape=self.IMAGE_SHAPE) 
        
        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(input_image)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[2], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters[3], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)   
        x = Conv2D(filters=filters[4], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)           
                
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)   
        """
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size1, strides=conv_strides1, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)  
        ### BOTTLENECK ###            
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size1, strides=conv_strides1, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        """
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
        autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    
        self.model = autoencoder
    
    def create_context_encoder(self):
        """
        Model from the paper Context Encoders.
        """
        resized_shape = (384, 512)
        conv_kernel_size = 5 #(4,4)
        conv_strides = 2
        filters = [128, 256, 512, 1024, 2096, 4192] #64, 128, 
        
        input_image = Input(shape=self.IMAGE_SHAPE)
        x = Lambda(lambda image: tf.image.resize_images(image, resized_shape))(input_image)

        x = Conv2D(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)        

        x = Conv2D(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=filters[4], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)     
        """
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)      
        
        x = Conv2D(filters=filters[6], kernel_size=conv_kernel_size, strides=conv_strides, padding = 'same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)     
        """
        #x = Lambda(self.channel_wise_dense_layer_tensorflow, arguments={'name': "channelwisedense"})(x)
        #x = LeakyReLU()(x)
        #x = BatchNormalization()(x)   
        """
        x = Conv2DTranspose(filters=filters[5], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        
        x = Conv2DTranspose(filters=filters[4], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)   
        """
        x = Conv2DTranspose(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        
        
        x = Conv2DTranspose(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        

        x = Conv2DTranspose(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        
        
        x = Conv2DTranspose(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=3, kernel_size=conv_kernel_size, strides=conv_strides, activation = 'sigmoid', padding = 'same')(x)
        
        x = Lambda(lambda image: tf.image.resize_images(image, self.dataset.IMAGE_SHAPE[:2]))(x)
        
        autoencoder = Model(input_image, x)
        autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    
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
    def __init__(self, path_results, dataset, dataset_reconstruct): 
        #self.input_shape = dataset.IMAGE_SHAPE
        self.model = None #self.create_model()
        self.path_results = path_results
        self.dataset = dataset # datasets train and val for training and train+val loss
        self.dataset_reconstruct = dataset_reconstruct # dataset val for validation/reconstruction during training.
        
    def train(self, epochs, batch_size): #val_split
        """
        Train by the use of train_on_batch() and Dataset.load_batch()
        NOT MAINTAINED
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
                    
                self.save_to_directory(self, loss_history, failed_im_load, epoch, train_batch, train_val_ratio, model_freq=100*train_val_ratio, loss_freq=train_val_ratio, reconstruct_freq=5*train_val_ratio, n_move_avg=1)
                
    def train_inpainting(self, epochs, batch_size, inpainting_grid=None, single_im=False):
        """
        Train by the use of train_on_batch() and Dataset.load_batch()
        'single_im=True' if deep-image-prior. only training, no validation.
        fails if more than one of cams_lenses.
        """
        ds = self.dataset
        np.save(self.path_results+'data_timestamp_list_train', ds.timestamp_list_train)
        np.save(self.path_results+'data_timestamp_list_val', ds.timestamp_list_val)
                
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
        #print(self.indexing_iterator)
        
        for epoch in range(epochs):
            print('Epoch '+str(epoch+1)+'/'+str(epochs))
            val_batch = 0 # keeps track of what number of val_batch we're currently at
            train_timestamp_index = 0 # keeps track of index in timestamp_list_train
            val_timestamp_index = 0 # keeps track of index in timestamp_list_val
            
            # train
            for train_batch in range(train_batches):
                print('Training batch '+str(train_batch+1)+'/'+str(train_batches)+'. ', end='')
                x = []
                x, failed_im_load = ds.load_batch(ds.timestamp_list_train[train_timestamp_index:train_timestamp_index+self.indexing_iterator], failed_im_load)
                
                if x == []:
                    continue
                
                if not inpainting_grid==None:
                    #x_batch_masked, x_batch = ds.mask_image(x[0], inpainting_grid) # SHOULD BE x AND NOT x[0]
                    
                    x_batch_masked, x_batch = ds.mask_batch(x, inpainting_grid)
                    
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
                        #x_batch_masked, x_batch = ds.mask_image(x[0], inpainting_grid) # SHOULD BE x AND NOT x[0]
                        x_batch_masked, x_batch = ds.mask_batch(x, inpainting_grid)
                        loss = self.model.test_on_batch(x_batch_masked, x_batch)
                    else:
                        loss = self.model.test_on_batch(x, x)
                    
                    loss_history.on_val_batch_end(loss)                    
                    print('Validate loss: '+str(loss))    
                    
                    val_batch += 1
                    val_timestamp_index += self.indexing_iterator
                    
                self.save_to_directory(loss_history, failed_im_load, epoch, train_batch, train_timestamp_index-self.indexing_iterator, val_timestamp_index-self.indexing_iterator, train_val_ratio, model_freq=100*train_val_ratio, loss_freq=train_val_ratio,  reconstruct_freq_train=train_val_ratio*1000, reconstruct_freq_val=train_val_ratio, inpainting_grid=inpainting_grid, single_im=single_im)
                
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
                print('Timestamps and cam_lens of failed batches:\n{}'.format(failed_im_load), file=text)
    
        if (freq_counter+1)%model_freq == 0:
            try:
                self.model.save(self.path_results+'epoch'+epoch_str+'_batch'+batch_str+'.hdf5')
                print('Model saved')
            except:
                self.model.save_weights(self.path_results+'epoch'+epoch_str+'_batch'+batch_str+'.hdf5')
                print('Model weights saved')
            
        if (freq_counter+1)%reconstruct_freq_train == 0: #used when reconstucting training data
            self.test_inpainting(dataset = self.dataset, what_data_split='train', timestamp_index=train_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid)
        
        if (freq_counter+1)%reconstruct_freq_val == 0: #reconstruct_freq_val needs to be a multiplier of train_val_ratio
            self.test_inpainting(dataset = self.dataset, what_data_split='val', timestamp_index=val_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid)
            #self.test_inpainting(dataset = self.dataset, what_data_split='train', timestamp_index=train_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid)
            
            #self.test(dataset = self.dataset, what_data_split='train', timestamp_index=train_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch)
            #self.test_inpainting(dataset = self.dataset_reconstruct, what_data_split='val', timestamp_index=0, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid)
        #if (freq_counter+1)%(reconstruct_freq*10) == 0: 
            #self.test_inpainting(dataset = self.dataset_reconstruct, what_data_split='val', timestamp_index=val_timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid)    
                       
    def test(self, dataset, what_data_split, timestamp_index, numb_of_timestamps, epoch, batch):
        """
        Test autoencoder.model on images from the train, val or test dataset.
        Saves figure to file.
        To be used during training or after fully trained.
        PARTLY MAINTAINED
        """
        numb_of_timestamps = self.indexing_iterator#//len(dataset.cams_lenses)
        
        if what_data_split == 'train':
            timestamps = dataset.timestamp_list_train[timestamp_index:timestamp_index+numb_of_timestamps]
        elif what_data_split == 'val':
            timestamps = dataset.timestamp_list_val[timestamp_index:timestamp_index+numb_of_timestamps]
        elif what_data_split == 'test':
            timestamps = dataset.timestamp_list_test[timestamp_index:timestamp_index+numb_of_timestamps]
        else:
            print('Invalid data argument, no reconstruction possible.')
        
        #print(len(timestamps))
        #images_per_figure = dataset.images_per_timestamp
            
        i = 0
        while i < numb_of_timestamps:
            x_batch,_ = dataset.load_batch(timestamps[i:i+self.indexing_iterator], failed_im_load=[]) #images_per_figure//self.dataset.images_per_timestamp]
            y_batch = self.model.predict_on_batch(x_batch)
            #print(x_batch.shape)
            plot = create_reconstruction_plot(self, x_batch, y_batch)
            plot.savefig(self.path_results+'reconstruction-'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'.jpg')
            i += self.indexing_iterator #images_per_figure//self.dataset.images_per_timestamp
            print('Reconstruction saved')
            
    def test_inpainting(self, dataset, what_data_split, timestamp_index, numb_of_timestamps, epoch, batch, inpainting_grid):
        """
        Test autoencoder.model on images from the train, val or test dataset.
        Saves figure to file.
        To be used during training or after fully trained.
        'numb_of_timestamps'>1 when used after fully trained. NOT INCLUDED ATM. SHOULD PROBABLY BE NAMED AND USED AS 'numb_of_figures'.
        
        TODO: make compatible with not inpainting
        
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
        
        i = 0
        while i < numb_of_timestamps:
            x,failed_im_load = dataset.load_batch(timestamps[i:i+self.indexing_iterator], failed_im_load=[])
            #print(x.shape)    
            if not inpainting_grid==None:
                #for image in range(len(x)):
                """
                if np.all(x[0]==x[1]): # batch consist of one image
                    x_batch_masked, x_batch = dataset.mask_image(x[0], inpainting_grid)    
                    x_batch_original_and_masked = np.concatenate((np.expand_dims(x_batch[0], axis=0), x_batch_masked), axis=0)
                    y_batch = self.model.predict_on_batch(x_batch_original_and_masked)
                    plot = create_reconstruction_plot(self, x_batch_original_and_masked, y_batch, inpainting_grid)                    
                    plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'.jpg')
                    print('Reconstruction saved')
                else: # same as below.
                    """
                x_batch_masked,_ = dataset.mask_batch(x, inpainting_grid)
                y_batch = self.model.predict_on_batch(x_batch_masked)
                plot = create_reconstruction_plot(self, x, x_batch_masked, y_batch)                    
                plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'.jpg')
                print('Reconstruction saved')                
            else:
                y_batch = self.model.predict_on_batch(x)
                plot = create_reconstruction_plot(self, x, x, y_batch)                    
                plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'.jpg')
                print('Reconstruction saved')            

            i += self.indexing_iterator
    
    def merge_inpaintings(self, y_batch, inpainting_grid):
        """
        Merges inpatinings in 'y_batch' to a single image. ish tested
        """
        mask_shape = (self.dataset.IMAGE_SHAPE[0]//inpainting_grid[0], self.dataset.IMAGE_SHAPE[1]//inpainting_grid[1])

        inpainted = np.empty(self.dataset.IMAGE_SHAPE)
        
        x0, y0 = 0,0
        i = 1
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
    ds = Dataset('all')
    path_results = '/home/kristoffer/Documents/mastersthesis/results/ex2/'
    ae = Autoencoder(ds, path_results)
    ae.create_ContextEncoder_model()
    ae.model.summary()
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
    

