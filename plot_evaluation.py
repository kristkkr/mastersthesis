#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:22:46 2018

@author: kristoffer
"""



from figures import plot_evaluation


path_results = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/test/test-day/eval/' 
n_im_day = 148
n_im_night = 61

plot_evaluation(path_results, n_images=n_im_day)