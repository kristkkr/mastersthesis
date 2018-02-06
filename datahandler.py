#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:02:43 2018

@author: kristoffer
self.hour_list seems to be wrong

"""



import sys
import os
import itertools
import pickle

import numpy as np
from PIL import Image
from collections import Iterable

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader

class Dataset():
    def __init__(self, path):
        self.size = 0
        self.path = path
        self.date_list = []
        self.hour_list = [] #np.empty(0)
        self.timestamp_list = []
        
    def get_timestamp_list(self): #, t_start, t_end, min_speed=0):
        """
        Returns a list/dictionary containing the timestamps ### when the ownship speed is greater than 'min_speed'.
        The argument 't_start' is the start of the time t.
        The argument 't_end' is the end of the time t
        pseudo:            
        iterte through directories
        if speed>min_speed
            self.data.append
        """
        
        #self.date_list = sorted([date for date in os.listdir(self.path) if '201' in date]) # not used
        self.hour_list = sorted([hour for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour])
        #self.timestamp_list = sorted([timestamp for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour for timestamp in os.listdir(self.path+date+'/'+hour+'/Cam0/Lens0') if '.jpg' in timestamp])
        #print(self.hour_list)
        print(len(self.hour_list))
                
        #for date in self.date_list:
            #self.hour_list.append(sorted([name for name in os.listdir(self.path+date) if '201' in name]))
            ##self.hour_list = [string for sublist in self.hour_list for string in sublist]
            ##self.hour_list = list(itertools.chain.from_iterable(self.hour_list))
            #np.append(self.hour_list, sorted([name for name in os.listdir(self.path+date) if '201' in name]))
            #hour_list = sorted([hour for hour in os.listdir(self.path+date) if '2017-10-22' in hour])
        for hour in self.hour_list:
            try:
                time_list = sorted([name[:19] for name in os.listdir(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0') if '.jpg' in name]) # name[:19] to remove extension
                #print(time_list)
                self.timestamp_list.append(time_list)
            except:
                print('Does not exist: '+hour)
        self.timestamp_list = self.flatten_list(self.timestamp_list)
        

    def flatten_list(self, l):
        return list(itertools.chain(*l))
            
        
    def vel_to_speed(self, vel):
        """
        Returns the speed of the ownship.
        vel is the velocity of the ownship in NED given in m/s
        """
        raise NotImplementedError
    
    
path = '/nas0/'

ds = Dataset(path)
ds.get_timestamp_list()
#print(ds.timestamp_list)
print(len(ds.timestamp_list))
with open('timestamps','wb') as fp:
    pickle.dump(ds.timestamp_list,fp)
    
"""
with open('timestamps','rb') as fp:
    l = pickle.load(fp)
    print(l)
"""

"""
### HOW TO USE DATALOADER TO GET IMAGE ###
file_basename = '/nas0/2017-10-22/2017-10-22-11/Cam2/Lens1/2017-10-22-11_00_00'
dl = DataLoader(path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
sensortype, sensor = dl.get_sensor_from_basename(file_basename)
folder = dl.get_sensor_folder(sensortype, sensor)
t = dl.get_time_from_basename(file_basename)
im = dl.load_image(t,sensortype,sensor)
Image.fromarray(im).show()
### --------------- ###

"""


