#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:19:17 2018

@author: kristoffer
"""


from datahandler import Dataset

from figures import plot_loss_history
# initialize model

ds = Dataset([(0,1)]) #[(1,1),(3,1)])
path_results = '/home/kristoffer/Documents/mastersthesis/results/ex31/'
plot_loss_history(path_results, train_val_ratio=8, n=100, single_im=False)

