#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:18:32 2018

@author: kristoffer
"""
import sys

import numpy as np
from keras.models import load_model

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader

from datahandler import Dataset
from autoencoder import Autoencoder, AutoencoderModel

from figures import plot_loss_history
# initialize model


ds_train = Dataset(cams_lenses = [(1,1),(3,1)]) #  when (1,1) is included, the batch size during training varies. might affect convergence in a bad way.
#ds_train.read_timestamps_file('datasets/new2704/speed>6/interval_5sec/removed_hours/timestamps.npy')
#ds_train.split_list(split_frac = (0.8,0.1,0.1), shuffle_order=True)
#ds_train.path_test = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/test/'

path_results = '/home/kristoffer/Documents/mastersthesis/results/ex42/' 
path_data = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/speed>6/interval_5sec/removed_illumination/' 
path_model_load = '/home/kristoffer/Documents/mastersthesis/results/ex41/continued/' 

#np.save(path_results+'data_timestamp_list_train', ds_train.timestamp_list_train)
#np.save(path_results+'data_timestamp_list_val', ds_train.timestamp_list_val)
#np.save(path_results+'data_timestamp_list_test', ds_train.timestamp_list_test)
ds_train.timestamp_list_train = np.load(path_data+'data_timestamp_list_train.npy') 
ds_train.timestamp_list_val = np.load(path_data+'data_timestamp_list_val.npy') 
#ds_train.timestamp_list_test = np.load(path_data+'data_timestamp_list_test.npy') 

"""
trained, validated = 2400+31200, 300+3900
train_start = ds_train.timestamp_list_train[:trained*5]
val_start = ds_train.timestamp_list_val[:validated*5]
ds_train.timestamp_list_train = np.concatenate((ds_train.timestamp_list_train[trained*5:], train_start))
ds_train.timestamp_list_val = np.concatenate((ds_train.timestamp_list_val[validated*5:], val_start))
"""

"""
ds_val = Dataset(cams_lenses = [(0,1)]) 
dl = DataLoader(ds_train.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
ds_val.read_timestamps_file('datasets/ais/speed>0.5/interval_60sec/max_range1000/timestamps.npy')
ds_val.split_list(split_frac = (0.0,1,0.0), shuffle_order=True)
#ds_val.timestamp_list_val = [dl.get_time_from_basename('/nas0/2018-01-18/2018-01-18-14/Cam3/Lens2/2018-01-18-14_09_01')]
"""

 
ae = Autoencoder(path_results, dataset = ds_train)
ae.create_ce()
#ae.create_context_encoder()
#ae.model.load_weights(path_model_load+'epoch00001_batch031200.hdf5')
#ae.model = load_model(path_model_load+'epoch00001_batch031200.hdf5')
#ae.model.optimizer.lr = 0.00001

ae.model.summary()

# experiment
epochs = 1
batch_size = 10
inpainting_grid = (8,10) # a tuple (height,width) or 'None'



ae.train(epochs, batch_size, inpainting_grid, single_im=False)

#ae.dataset.path_test = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/test/'
#ae.evaluate(inpainting_grid, visual=0)
