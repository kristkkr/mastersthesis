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
#from PIL import Image
from random import shuffle
#from collections import Iterable

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader

class Dataset():
    IMAGES_PER_TIMESTAMP = 12
    NUMB_OF_CAMERAS = 4
    NUMB_OF_LENSES = 3
    IMAGE_SHAPE = (1920,2560,3)
    
    def __init__(self):
        self.path = '/nas0/'
        self.image_shape = (1920,2560,3)
        #self.size = 0
        #self.date_list = []
        self.hour_list = [] 
        self.timestamp_list = []
        self.get_timestamp_list_train = []
        self.get_timestamp_list_val = []
        self.get_timestamp_list_test = []
        
    def get_timestamp_list(self, sampling_period = 1, randomize = False): #, t_start, t_end, min_speed=0):
        """
        Returns the instance variable timestamp_list. (when the ownship speed is greater than 'min_speed'.)
        The argument 't_start' is the start of the time t.
        The argument 't_end' is the end of the time t
        """
        
        #self.date_list = sorted([date for date in os.listdir(self.path) if '201' in date]) # not used
        self.hour_list = sorted([hour for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '2017-10-22-11' in hour])
        #self.timestamp_list = sorted([timestamp for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour for timestamp in os.listdir(self.path+date+'/'+hour+'/Cam0/Lens0') if '.jpg' in timestamp])
        print('Lenght of hour_list: '+str(len(self.hour_list)))
                
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        for hour in self.hour_list:
            try:
                time_list = sorted([name[:19] for name in os.listdir(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0') if '.jpg' in name]) # time_list contains filename strings
                
                for i in range(len(time_list)):
                    time_list[i] = dl.get_time_from_basename(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0/'+time_list[i])  # time_list now contains datetime instances
              
                self.timestamp_list.append(time_list)
                print('Hour added: '+hour)
                
            except:
                print('Hour not added: '+hour)
                
        self.timestamp_list = self.flatten_list(self.timestamp_list)
        
        if sampling_period > 1:
            self.timestamp_list = self.sample_list(self.timestamp_list, sampling_period)

        if randomize: 
            shuffle(self.timestamp_list)
            
    def split_data(self,train_frac, val_frac):
        
        raise NotImplementedError
                
    def sample_list(self, l, sampling_period):
        return l[0:len(l):sampling_period]
    
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
        
        batch = np.empty((self.IMAGES_PER_TIMESTAMP*len(timestamps),)+(self.IMAGE_SHAPE), np.uint8) #use self.image_shape
        
        #print(timestamps)
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        
        i = 0
        for timestamp in timestamps:
            for cam in range(self.NUMB_OF_CAMERAS):
                for lens in range(self.NUMB_OF_LENSES):
                        
                    batch[i,:] = dl.load_image(timestamp, dl.TYPE_CAMERA,(cam,lens))
                    i+=1
                          
        return batch.astype('float32') / 255.
    
    def generate_batches(self,batch_size): 
        """
        Returns a Python generator that generates batches of data indefinetily. To be used in Keras fit_generator().
        Works well, but waste the first batch and feed 2nd batch to fit_generator().
        """
        numb_of_timestamps_in_batch = batch_size//self.IMAGES_PER_TIMESTAMP
        
        while 1:
            t = 0
            for t in range(len(self.timestamp_list)):
                               
                x_batch = self.load_batch(self.timestamp_list[t:t+numb_of_timestamps_in_batch])
    
                t += numb_of_timestamps_in_batch
                yield (x_batch, x_batch)
            
    def write_timestamps_file(self, filename):
        with open(filename,'wb') as fp:
            pickle.dump(self.timestamp_list,fp)
            print("Timestamps written to file")
        
    def read_timestamps_file(self, filename):
        with open(filename,'rb') as fp:
            self.timestamp_list = pickle.load(fp)
            print("Timestamps read from file. Length: "+str(len(self.timestamp_list)))
 
        
    def vel_to_speed(self, vel):
        """
        Returns the speed of the ownship.
        vel is the velocity of the ownship in NED given in m/s
        """
        raise NotImplementedError
    
    
if __name__ == "__main__":
    
    ds = Dataset()
    ds.get_timestamp_list(sampling_period=60*12)
    
    #print(len(ds.timestamp_list))
    #ds.timestamp_list = ds.sample_list(ds.timestamp_list, sampling_period=60)
    print(len(ds.timestamp_list))


    ds.write_timestamps_file('timestamps2017-10-22-11-sampled')
    
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
        
