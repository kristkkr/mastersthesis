#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:18:32 2018

@author: kristoffer
"""
import numpy as np
from keras.models import load_model

from datahandler import Dataset
from autoencoder import Autoencoder
from figures import save_plot_loss_history

# initialize model

ds = Dataset('all')
path_results = '/home/kristoffer/Documents/mastersthesis/results/ex5/'

ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy') 
ds.timestamp_list_val = np.load(path_results+'data_timestamp_list_val.npy') 
train_val_ratio = round(len(ds.timestamp_list_train)/len(ds.timestamp_list_val))
save_plot_loss_history(path_results, train_val_ratio=train_val_ratio, n=100)


#ae = Autoencoder(ds, path_results)
#ae.model = load_model(ae.path_results+'epoch01_batch004501_valloss0.0622.hdf5')
#ae.reconstruct(numb_of_timestamps=20, images_per_figure=12)


"""
#[(0,1), (0,2), (1,1), (2,1), (3,1)]
ds = Dataset(cams_lenses = [(1,1),(3,1)])
ds.read_timestamps_file('datasets/all/interval_60sec/timestamps')
ds.split_list(split_frac = (0.9,0.1,0.0), shuffle_order=True)

path_results = '/home/kristoffer/Documents/mastersthesis/results/ex5/'

ae = Autoencoder(ds, path_results)
#ae.create_autoencoder()
ae.model.summary()

# experiment
epochs = 2
batch_size = 8

ae.train(epochs, batch_size)

"""