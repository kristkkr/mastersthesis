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
from random import shuffle
#from collections import Iterable

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader

class Dataset():
    
    def __init__(self):
        self.path = '/nas0/'
        self.image_shape = (1920,2560,3)
        self.size = 0
        #self.date_list = []
        self.hour_list = [] 
        self.timestamp_list = []
        
    def get_timestamp_list(self, randomize = False): #, t_start, t_end, min_speed=0):
        """
        Returns a list containing the timestamps ### when the ownship speed is greater than 'min_speed'.
        The argument 't_start' is the start of the time t.
        The argument 't_end' is the end of the time t
        """
        
        #self.date_list = sorted([date for date in os.listdir(self.path) if '201' in date]) # not used
        self.hour_list = sorted([hour for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '2017-10-22-11' in hour])
        #self.timestamp_list = sorted([timestamp for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour for timestamp in os.listdir(self.path+date+'/'+hour+'/Cam0/Lens0') if '.jpg' in timestamp])
                        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        for hour in self.hour_list:
            try:
                time_list = sorted([name[:19] for name in os.listdir(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0') if '.jpg' in name]) # name[:19] to remove extension
                for i in range(len(time_list)):
                    time_list[i] = dl.get_time_from_basename(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0/'+time_list[i])
                self.timestamp_list.append(time_list)
                
            except:
                print('Does not exist: '+hour)
                
        self.timestamp_list = self.flatten_list(self.timestamp_list)
        
        if randomize: 
            shuffle(self.timestamp_list)
                

    def flatten_list(self, l):
        """
        Flattens a list 'l'
        """
        return list(itertools.chain(*l))
    
    def load_batch(self, timestamps):
        """
        Load a batch of images for the timestamps in 'timestamps'
        The argument 'timestamps' is a list of timestamps defining the batch, most likely one/two timestamp(s).
        Returns a numpy array of shape (len(timestamps),1920,2560,3) of type np.float32 with range [0,1].
        """
        
        batch = np.empty((12*len(timestamps),)+(self.image_shape), np.uint8) #use self.image_shape
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        
        i = 0
        for timestamp in timestamps:
            for cam in range(4):
                for lens in range(3):
                        
                    batch[i,:] = dl.load_image(timestamp, dl.TYPE_CAMERA,(cam,lens))
                    i+=1
                    
        return batch.astype('float32') / 255.
    
    def generate_batches(self):
        """
        Returns a Python generator that generates batches of data indefinetily. To be used in Keras fit_generator().
        """
        for timestamp in self.timestamp_list:
            x_batch = self.load_batch([timestamp])
            yield (x_batch, x_batch)
        
            
        
    def vel_to_speed(self, vel):
        """
        Returns the speed of the ownship.
        vel is the velocity of the ownship in NED given in m/s
        """
        raise NotImplementedError
    
    
if __name__ == "__main__":
    
    ds = Dataset()
    
    ds.get_timestamp_list()
    #print(ds.timestamp_list[:1])
    #print(len(ds.timestamp_list))
    #batch = ds.load_batch(ds.timestamp_list[:2])
    #Image.fromarray(batch[0,:]).show()
    
    """
    #write to file
    with open('timestamps','wb') as fp:
        pickle.dump(ds.timestamp_list,fp)
        print("Timestamps written to file")
    
    # read from file
    with open('timestamps','rb') as fp:
        l = pickle.load(fp)
        batch = ds.load_batch(l[:1])
        print("Timestamps read from file")
    """
    
    """
    
    ### HOW TO USE DATALOADER TO GET IMAGE ###
    file_basename = '/nas0/2017-10-22/2017-10-22-11/Cam2/Lens1/2017-10-22-11_00_00'
    dl = DataLoader(path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
    sensortype, sensor = dl.get_sensor_from_basename(file_basename)
    folder = dl.get_sensor_folder(sensortype, sensor)
    t = dl.get_time_from_basename(file_basename)
    #im = dl.load_image(t,sensortype,sensor)
    #Image.fromarray(im).show()
    ### --------------- ###
        
    """ 
        
