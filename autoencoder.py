#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:09 2018

@author: kristoffer

"""
import os

import numpy as np
from scipy.ndimage import imread
#from scipy.imageio import imread

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, BatchNormalization, Lambda, LeakyReLU, Flatten, Reshape
from keras.models import Model, Sequential
from keras.callbacks import Callback#, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend import tf
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import xml.etree.ElementTree as ET


from figures import create_reconstruction_plot, create_reconstruction_plot_single_image, plot_loss_history, insert_leading_zeros, show_detections

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
        optimizer = Adam()#lr=0.0001
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
                                       model_freq=100*train_val_ratio, 
                                       loss_freq=train_val_ratio,  
                                       reconstruct_freq_train=100000, 
                                       reconstruct_freq_val=5*train_val_ratio, 
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
        
        i = 0
        while i < numb_of_timestamps:
            x,failed_im_load = dataset.load_batch(timestamps[i:i+self.indexing_iterator], failed_im_load=[])
            
            #if failed_im_load != []:
                #continue
            
            if not inpainting_grid==None:
                                
                if single_im_batch: # batch consist of one image
                    x_batch_masked, x_batch = dataset.mask_image(x[0], inpainting_grid)    # selects the first image in batch. 
                    x_batch_original_and_masked = np.concatenate((np.expand_dims(x_batch[0], axis=0), x_batch_masked), axis=0)
                    y_batch = self.model.predict_on_batch(x_batch_original_and_masked)
                    plot = create_reconstruction_plot_single_image(self, x_batch_original_and_masked, y_batch, inpainting_grid)                    
                    plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'-single.jpg')
                    print('Reconstruction single image inpainting saved')
                else: # same as below.
                    #x_batch_masked,_ = dataset.mask_batch(x, inpainting_grid)
                    x_batch_masked,_ = dataset.mask_batch_randomly(x, inpainting_grid)
                    y_batch = self.model.predict_on_batch(x_batch_masked)
                    plot = create_reconstruction_plot(self, x, y_batch, x_batch_masked)                    
                    plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'.jpg')
                    print('Reconstruction inpainting saved')                
            else:
                y_batch = self.model.predict_on_batch(x)
                plot = create_reconstruction_plot(self, x, y_batch)                    
                plot.savefig(self.path_results+'reconstruction'+'-epoch'+str(epoch+1)+'-batch'+str(batch+1)+what_data_split+str(i+1)+'.jpg')
                print('Reconstruction regular saved')            

            i += self.indexing_iterator

    
    def merge_inpaintings(self, y_batch, inpainting_grid):
        """
        Merges inpatinings in 'y_batch' to a single image. 
        MOVE TO FIGURES.
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

    def evaluate(self, inpainting_grid):
        """
        Evaluate model on test data. Same procedure as for single_im_batch in test()
        """
        #timestamps = self.dataset.timestamp_list_val[:5]
        #numb_of_timestamps = len(timestamps)
        
        #for i in range(numb_of_timestamps):
            #x,failed_im_load = self.dataset.load_batch(timestamps[i:i+1], failed_im_load=[])
            
            #for cam_lens_im in range(len(x)):
        path_test = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/test2/'
        for filename in sorted(os.listdir(path_test))[:2]: #self.dataset.path_test):
            if filename.endswith('.xml'):
                image = imread(path_test+filename.replace('.xml','.jpg')).astype('float32') / 255.
                image_batch = np.expand_dims(image, 0)
                
                
                """
                x_batch_masked, _ = self.dataset.mask_image(image, inpainting_grid)
                y_batch = self.model.predict_on_batch(x_batch_masked)
                inpainted = self.merge_inpaintings(y_batch, inpainting_grid)
                residual = np.mean(np.abs(np.subtract(image, inpainted)), axis=2)
                """
                y_batch = self.model.predict_on_batch(image_batch)
                y = y_batch[0,:]
                
                #Image.fromarray(y).show()
                
                residual = np.mean(np.abs(np.subtract(image, y)), axis=2)
                
                #residual = imread(path_test+filename.replace('.xml','.jpg'), mode='L')
                #print(residual.shape)
                #print(np.amin(residual),np.amax(residual))
                #Image.fromarray(residual, mode='L').show()
                
                threshold = 50
                for threshold in range(50,200,30):
                    threshold_array = threshold*np.ones(residual.shape,dtype=np.uint8).astype('float32') / 255.
                    #print(threshold_array.shape)
                    #print(np.amin(threshold_array),np.amax(threshold_array))
                    binary_map = np.greater(residual, threshold_array)
                    #print(binary_map.shape)
                    #print(np.amin(binary_map),np.amax(binary_map))
                    #Image.fromarray(binary_map, mode='L').show()
                    ## evaluate binary_map against GT ##
                     
                    box_tp, box_fn, recall = count_box_detections(binary_map, path_test+filename.replace('.jpg', '.xml'))# self.dataset.IMAGE_EXTENSION
                    pixel_tp, pixel_fp, precision = count_pixel_detections(binary_map, path_test+filename.replace('.jpg', '.xml'))
                    
                    print(filename, threshold)
                    print(box_tp, box_fn,'recall=',recall)
                    print(pixel_tp, pixel_fp, 'precision=',precision)
                    ## visual 
                    object_map = map_on_image(image, binary_map)
                    #print(object_map.shape)
                    #Image.fromarray(object_map, mode='RGB').show()
                    
                    figure = show_detections(image, y, residual, object_map)
                    figure.savefig(self.path_results+'detections-threshold'+str(threshold)+'--'+filename.replace('.xml','.jpg'))
                    print('figsaved')
            
def count_pixel_detections(binary_map, gt_file):
    """counts pixel detections tp, tn, fp in binary predicted map with respect to ground truth files. use both background and object classes?"""
    y,x = binary_map.shape
    object_gt_map,_,background_gt_map,_ = read_gt_file(gt_file,(y,x))
    gt_map = np.logical_or(object_gt_map,background_gt_map)
    
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
        precision = tp/(tp+fp)
    except:
        precision = 'NaN'
    return tp, fp, precision

def count_box_detections(binary_map, gt_file):
    """counts box detections. use object box class"""
    y,x = binary_map.shape
    _, object_boxes,_,_ = read_gt_file(gt_file,(y,x))
    #binary_map, gt_map = binary_map/255, gt_map/255
    tp, fp, fn, tn = 0,0,0,0
    for box in object_boxes:
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        
        if binary_map[ymin:ymax,xmin:xmax].any() == 1:
            tp +=1
        else:
            fn +=1
    try:
        recall = tp/(tp+fn)    
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
                image[i,j,:] = [mask_color,0,0]
    return image
        
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
    

