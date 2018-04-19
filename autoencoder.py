#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer

"""
import numpy as np

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, BatchNormalization, Lambda, LeakyReLU, Flatten, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.backend import tf as ktf

from figures import create_reconstruction_plot, save_to_directory


class Autoencoder:
    def __init__(self, dataset, path_results, dataset_reconstruct=None): 
        #self.input_shape = dataset.IMAGE_SHAPE
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
        
        input_image = Input(shape=self.dataset.IMAGE_SHAPE) # change to ds.IMAGE_SHAPE?
        
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
    
    def create_ContextEncoder_model(self):
        """
        Model from the paper Context Encoders.
        """
        conv_kernel_size = (4,4)
        conv_strides = (2)
        filters = [64, 64, 128, 256, 512, 4000]
        #bottleneck = 8192
        
        input_image = Input(shape=self.dataset.IMAGE_SHAPE)
        x = Lambda(lambda image: ktf.image.resize_images(image, (128, 128)))(input_image)

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
        
        #x = Flatten()(x)
        #x = Dense(units=bottleneck)(x)
        x = Conv2D(filters=filters[5], kernel_size=conv_kernel_size, activation=LeakyReLU())(x) #same as a dense layer
        x = BatchNormalization()(x)
        #x = Reshape(target_shape=(4,4,512))(x)
        
        x = Conv2DTranspose(filters=filters[4], kernel_size=conv_kernel_size, strides=4, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        
        
        x = Conv2DTranspose(filters=filters[3], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        

        x = Conv2DTranspose(filters=filters[2], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        

        x = Conv2DTranspose(filters=filters[1], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)        
        
        x = Conv2DTranspose(filters=filters[0], kernel_size=conv_kernel_size, strides=conv_strides, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=3, kernel_size=conv_kernel_size, strides=conv_strides, activation = 'sigmoid', padding = 'same')(x)
        

        x = Lambda(lambda image: ktf.image.resize_images(image, self.dataset.IMAGE_SHAPE[:2]))(x)
        
        autoencoder = Model(input_image, x)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
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
                
    def train_inpainting(self, epochs, batch_size, inpainting_grid, single_im=False):
        """
        Train by the use of train_on_batch() and Dataset.load_batch()
        'single_im=True' if deep-image-prior. only training, no validation.
        fails if more than one of cams_lenses.
        """
        ds = self.dataset
        np.save(self.path_results+'data_timestamp_list_train', ds.timestamp_list_train)
        np.save(self.path_results+'data_timestamp_list_val', ds.timestamp_list_val)
                
        batches_per_epoch = len(ds.timestamp_list)*ds.images_per_timestamp*np.product(inpainting_grid)//batch_size
        train_batches = len(ds.timestamp_list_train)*ds.images_per_timestamp*np.product(inpainting_grid)//batch_size
        val_batches = len(ds.timestamp_list_val)*ds.images_per_timestamp*np.product(inpainting_grid)//batch_size
        
        train_val_ratio = int(round(train_batches/val_batches)) #train_batches//val_batches #
                
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
                x = []
                x, failed_im_load = ds.load_batch(ds.timestamp_list_train[train_timestamp_index:train_timestamp_index+1], failed_im_load)
                if x == []:
                    continue
                
                ### TRAIN
                x_batch_masked, x_batch = ds.mask_image(x[0], inpainting_grid)

                loss = self.model.train_on_batch(x_batch_masked, x_batch)
                ###
                loss_history.on_train_batch_end(loss)
                print('Training loss: '+str(loss))
                
                train_timestamp_index += 1
                
                # validate
                if (not single_im) and (train_batch+1) % train_val_ratio == 0:           
                    print('Validate batch '+str(val_batch+1)+'/'+str(val_batches)+'. ', end='')
                    x = []
                    x, failed_im_load = ds.load_batch(ds.timestamp_list_val[val_timestamp_index:val_timestamp_index+1], failed_im_load)
                    if x == []:
                        continue
                    x_batch_masked, x_batch = ds.mask_image(x[0], inpainting_grid)

                    loss = self.model.test_on_batch(x_batch_masked, x_batch)
                    loss_history.on_val_batch_end(loss)                    
                    print('Validate loss: '+str(loss))    
                    
                    val_batch += 1
                    val_timestamp_index += 1
                    
                save_to_directory(self, loss_history, failed_im_load, epoch, train_batch, val_timestamp_index, train_val_ratio, model_freq=1*train_val_ratio, loss_freq=train_val_ratio, reconstruct_freq=1*train_val_ratio, n_move_avg=1, inpainting_grid=inpainting_grid, single_im=single_im)
                
                    

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
            
    def test_inpainting(self, what_data, timestamp_index, numb_of_timestamps, epoch, batch, inpainting_grid):
        """
        Test autoencoder.model on images from the train, val or test dataset.
        Saves figure to file.
        To be used during training or after fully trained.
        """
        
        if what_data == 'train':
            timestamps = self.dataset.timestamp_list_train[timestamp_index:timestamp_index+numb_of_timestamps]
        elif what_data == 'val':
            timestamps = self.dataset.timestamp_list_val[timestamp_index:timestamp_index+numb_of_timestamps]
        elif what_data == 'test':
            timestamps = self.dataset.timestamp_list_test[timestamp_index:timestamp_index+numb_of_timestamps]
        else:
            print('Invalid data argument, no reconstruction possible.')
        
        for i in range(numb_of_timestamps):
            x,failed_im_load = self.dataset.load_batch(timestamps[i:i+1], failed_im_load=[])
            
            for image in range(len(x)):
                    
                x_batch_masked, x_batch = self.dataset.mask_image(x[image], inpainting_grid)
            
                x_batch_original_and_masked = np.concatenate((np.expand_dims(x_batch[0], axis=0), x_batch_masked), axis=0)
                y_batch = self.model.predict_on_batch(x_batch_original_and_masked)
                plot = create_reconstruction_plot(self, x_batch_original_and_masked, y_batch, inpainting_grid)
                plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data+str(i+1)+'img'+str(image+1)+'.jpg')
                print('Reconstruction saved')
            
    
    def merge_inpaintings(self, y_batch, inpainting_grid):
        """
        Merges inpatinings in 'y_batch' to a single image. ish tested
        """
        #y_batch = self.model.predict_on_batch(x_batch_masked)
        #print(y_batch.shape)
        mask_shape = (self.dataset.IMAGE_SHAPE[0]//inpainting_grid[0], self.dataset.IMAGE_SHAPE[1]//inpainting_grid[1])
        #print(mask_shape)
        inpainted = np.empty(self.dataset.IMAGE_SHAPE)
        
        #ulc = (0,0) # upper left corner coordinates
        x0, y0 = 0,0
        i = 1
                        
        for x in range(inpainting_grid[0]):
            x = x*mask_shape[1]
            for y in range(inpainting_grid[1]):
                #rectangle_coordinates = [ulc, (ulc[0]+mask_shape[1],ulc[1]+mask_shape[0])]
                y = y*mask_shape[0]
                #print(y,x)
                #inpaint = y_batch[i][y0:y0+mask_shape[0]][x0:x0+mask_shape[1]]
                #print(inpaint.shape)
                #inpainted[rectangle_coordinates[0][0]:rectangle_coordinates[1][0], rectangle_coordinates[0][1]:rectangle_coordinates[1][1]] = y_batch[i][rectangle_coordinates[0][0]:rectangle_coordinates[1][0], rectangle_coordinates[0][1]:rectangle_coordinates[1][1]]
                
                inpainted[y:y+mask_shape[0],x:x+mask_shape[1]] = y_batch[i,y:y+mask_shape[0],x:x+mask_shape[1]]
                #print(inpaint.shape)                
                
                i += 1
                
                y0 = y0 + mask_shape[0]
                #ulc = (ulc[0],ulc[1]+mask_shape[0])
            x0 = x0 + mask_shape[1]
            #ulc = (ulc[0]+mask_shape[1],0)
        
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
    

