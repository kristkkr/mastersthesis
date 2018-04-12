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


# initialize model
"""
ds = Dataset('all')
path_results = '/home/kristoffer/Documents/mastersthesis/results/ex8/'

ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy') 
ds.timestamp_list_val = np.load(path_results+'data_timestamp_list_val.npy') 
train_val_ratio = round(len(ds.timestamp_list_train)/len(ds.timestamp_list_val))
save_plot_loss_history(path_results, train_val_ratio=train_val_ratio, n=1000)


ae = Autoencoder(ds, path_results)
ae.model = load_model(ae.path_results+'epoch01_batch000397_valloss0.0929.hdf5')
ae.reconstruct(data = 'val', numb_of_timestamps=1, images_per_figure=12)


"""

ds = Dataset(cams_lenses = [(1,1),(3,1)])

#ds.read_timestamps_file('datasets/ais/interval_60sec/max_range1000/timestamps.npy')
#ds.split_list(split_frac = (0.9,0.1,0.0), shuffle_order=True)

path_results = '/home/kristoffer/Documents/mastersthesis/results/ex11/' 
ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy') 
ds.timestamp_list_val = np.load(path_results+'data_timestamp_list_val.npy') 


#dl = DataLoader(ds.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
#ds.timestamp_list_train = [dl.get_time_from_basename('/nas0/2018-03-26/2018-03-26-12/Cam3/Lens2/2018-03-26-12_00_00')]
#ds.timestamp_list_val = ds.timestamp_list_train



ae = Autoencoder(ds, path_results)
ae.create_model()
#ae.model = load_model(path_results+'epoch02_batch000397_valloss0.1015.hdf5')
ae.model.summary()

# experiment
epochs = 1
batch_size = 6

ae.train(epochs, batch_size)

