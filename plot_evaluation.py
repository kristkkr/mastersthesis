#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:22:46 2018

@author: kristoffer
"""



from figures import plot_evaluation_metrics


path_results = '/home/kristoffer/Documents/mastersthesis/results/test/test-daynight/eval/' 
n_im_day = 147
n_im_night = 56

plot_evaluation_metrics(path_results, n_images=n_im_night+n_im_day, name='daynight')

path_results = '/home/kristoffer/Documents/mastersthesis/results/test/test-day/eval/'
plot_evaluation_metrics(path_results, n_images=n_im_day, name='day')

path_results = '/home/kristoffer/Documents/mastersthesis/results/test/test-night/eval/' 
plot_evaluation_metrics(path_results, n_images=n_im_night, name='night')