#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:19:17 2018

@author: kristoffer
"""



from figures import plot_loss_history
# initialize model


path_results = '/home/kristoffer/Documents/mastersthesis/results/ex48/'
plot_loss_history(path_results, train_val_ratio=8, n=1000, single_im=False, ylim = (0.0002,0.0008))

