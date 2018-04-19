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
from autoencoder import Autoencoder

#from figures import save_plot_loss_history
# initialize model
"""
ds = Dataset([(0,1)]) #[(1,1),(3,1)])
path_results = '/home/kristoffer/Documents/mastersthesis/results/ex15/validation/'
#save_plot_loss_history(path_results, train_val_ratio=9, n=100, single_im=False)

#ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy') 
#ds.timestamp_list_val = np.load(path_results+'data_timestamp_list_val.npy') 
ds.read_timestamps_file('datasets/ais/interval_60sec/max_range1000/timestamps.npy')
ds.split_list(split_frac = (1.0,0,0), shuffle_order=False)

ae = Autoencoder(ds, path_results)
ae.model = load_model('/home/kristoffer/Documents/mastersthesis/results/ex15/epoch00001_batch009000.hdf5')
ae.test_inpainting(what_data='train', timestamp_index=0, numb_of_timestamps=1, epoch=0, batch=8999, inpainting_grid=(2,2))

"""

ds = Dataset(cams_lenses = [(3,1)]) #(1,1)

ds.read_timestamps_file('datasets/speed/interval_60sec/timestamps.npy')
ds.split_list(split_frac = (0.9,0.1,0.0), shuffle_order=True)

#path_prev = '/home/kristoffer/Documents/mastersthesis/results/ex12/' 
path_results = '/home/kristoffer/Documents/mastersthesis/results/ex17/' 
#ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy') 
#ds.timestamp_list_val = np.load(path_results+'data_timestamp_list_val.npy') 


dl = DataLoader(ds.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
#ds.timestamp_list_train = [dl.get_time_from_basename('/nas0/2018-01-18/2018-01-18-14/Cam3/Lens2/2018-01-18-14_09_01')]
#ds.timestamp_list_val = ds.timestamp_list_train



ae = Autoencoder(ds, path_results)
ae.create_ContextEncoder_model()
#ae.model = load_model(path_prev+'epoch00300_batch000001.hdf5')
ae.model.summary()

# experiment
epochs = 10
batch_size = 9
inpaining_grid = (3,3)

ae.train_inpainting(epochs, batch_size, inpaining_grid, single_im=False)

