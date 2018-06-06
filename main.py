#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:18:32 2018

@author: kristoffer
"""
import sys
from random import shuffle
import numpy as np
from scipy.ndimage import imread
from keras.models import load_model

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader

from datahandler import Dataset
from autoencoder import Autoencoder, AutoencoderModel

from figures import plot_loss_history



ds_train = Dataset(cams_lenses = [(1,1),(3,1)]) # only cameras to the right and left

path_data = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/speed>6/interval_5sec/removed_illumination/' 
ds_train.timestamp_list_train = np.load(path_data+'data_timestamp_list_train.npy') 
ds_train.timestamp_list_val = np.load(path_data+'data_timestamp_list_val.npy')
ds_train.timestamp_list_train2 = np.load(path_data+'data_timestamp_list_train2.npy') 
ds_train.timestamp_list_val2 = np.load(path_data+'data_timestamp_list_val2.npy') 



path_results = '/home/kristoffer/Documents/mastersthesis/results/ex48/continued2/' 

path_model_load = '/home/kristoffer/Documents/mastersthesis/results/ex48/continued/' 

# CREATE AUTEONCDOER MODEL
ae = Autoencoder(path_results, dataset = ds_train)
#ae.create_fca()
ae.create_ae_dilated()
#ae.create_ce()
ae.model.load_weights(path_model_load+'epoch00001_batch099200.hdf5')
#ae.model = load_model(path_model_load+'epoch00001_batch002400.hdf5')
#ae.model.optimizer.lr = 0.00001
ae.model.summary()


# TRAIN MODEL 
epochs = 1
batch_size = 16
inpainting_grid = (9,12) # a tuple (height,width) or 'None'

ae.train(epochs, batch_size, inpainting_grid, single_im=False)

# EVALUATE MODEL
#ae.dataset.path_test = '/home/kristoffer/Documents/mastersthesis/results/test/'
#ae.evaluate_metrics(inpainting_grid)
#ae.predict_inpaint_residual(inpainting_grid)
#ae.evaluate_visual()








