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
import datetime
import json

import numpy as np
from PIL import Image, ImageDraw
from random import shuffle
#from collections import Iterable

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader

class Dataset():
    NUMB_OF_CAMERAS = 4
    NUMB_OF_LENSES = 3
    IMAGE_SHAPE = (1920,2560,3)
    IMAGE_EXTENSION = '.jpg'
    
    def __init__(self, cams_lenses):
        self.path = '/nas0/'
        self.sampling_interval = 1
        self.date_list = []
        self.hour_list = [] 
        self.timestamp_list = []
        self.timestamp_list_shuffled = []
        self.timestamp_list_train = []
        self.timestamp_list_val = []
        self.timestamp_list_test = []
        if cams_lenses == 'all':
            self.cams_lenses = [(i,j) for i in range(self.NUMB_OF_CAMERAS) for j in range(self.NUMB_OF_LENSES)]
            print('All cameras and lenses included in dataset')
        else: 
            self.cams_lenses = cams_lenses
            print('Selection of cameras and lenses:', self.cams_lenses)
        self.images_per_timestamp = len(self.cams_lenses)        
        
    def get_timestamp_list(self, t_start, t_end, sampling_interval):
        """
        Returns the instance variable timestamp_list. 
        Bad method in most ways, but needs only to be run once every time the entire dataset is updated.
        """
        
        self.date_list = sorted([date for date in os.listdir(self.path) if '201' in date]) # not used
        self.hour_list = sorted([hour for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour])
        #self.timestamp_list = sorted([timestamp for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour for timestamp in os.listdir(self.path+date+'/'+hour+'/Cam0/Lens0') if '.jpg' in timestamp])
        print('Lenght of hour_list: '+str(len(self.hour_list)))
                
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        for hour in self.hour_list:
            try:
                timestamps_in_hour = sorted([name[:19] for name in os.listdir(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0') if '.jpg' in name]) # time_list contains filename strings
                
                for i in range(len(timestamps_in_hour)):
                    timestamps_in_hour[i] = dl.get_time_from_basename(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0/'+timestamps_in_hour[i])  # get_time_from_basetime returns datetime instances
              
                self.timestamp_list.append(timestamps_in_hour)
                print('Hour added: '+hour)
                
            except:
                print('Hour not added: '+hour)
                
        self.timestamp_list = self.flatten_list(self.timestamp_list)
                    
        if sampling_interval > 1:
            self.timestamp_list = self.sample_list(self.timestamp_list, sampling_interval)
            print('sampling')
            
    def read_metadata(self, timestamp):
        """
        Returns a ditctionary containing metadata about images taken at 'timestamp'.
        """
        date_string = str(timestamp.year)+'-'+str(timestamp.month).zfill(2)+'-'+str(timestamp.day).zfill(2)
        path = self.path+date_string+'/'+date_string+'-'+str(timestamp.hour).zfill(2)+'/Cam0/Lens0/'+date_string+'-'+str(timestamp.hour).zfill(2)+'_'+str(timestamp.minute).zfill(2)+'_'+str(timestamp.second).zfill(2)+'.json'
        
        return json.load(open(path))
       
            
    def split_list(self, split_frac, shuffle_order = False):
        """
        Splits the 'timestamp_list' into three lists.
        The argument 'split_frac' is a tuple on the form (train_frac, val_frac, test_frac).
        """
        assert(sum(split_frac) == 1.0)
        
        if shuffle_order:
            shuffle(self.timestamp_list)
            
        size = len(self.timestamp_list)
                
        self.timestamp_list_train = self.timestamp_list[0:int(size*split_frac[0])]
        self.timestamp_list_val = self.timestamp_list[int(size*split_frac[0]):int(size*sum(split_frac[0:2]))]
        self.timestamp_list_test = self.timestamp_list[int(size*sum(split_frac[0:2])):size]
    
    def select_subset(self, t_start=None, t_end=None, min_speed=None, targets_ais=None):
        """ 
        The arguments 't_start' and 't_end' is the start and end of the time t.
        'min_speed'
        'targets_ais' does not work yet.
        
        """        
        tl = ds.timestamp_list
        del ds.timestamp_list
        #print('Timestamp_list now empty')
        
        ds.timestamp_list = []
                        
        for timestamp in tl:
            if t_start != None or t_end != None:
                if t_start <= timestamp and timestamp < t_end:
                    self.timestamp_list.append(timestamp)
            elif min_speed != None:
                metadata = self.read_metadata(timestamp)
                vel = metadata["own_vessel"]["velocity"]
                speed = np.sqrt(sum([vel_i **2 for vel_i in vel]))
                if speed > min_speed:
                    self.timestamp_list.append(timestamp)
            elif targets_ais != None:
                metadata = self.read_metadata(timestamp)
                targets = metadata["targets_ais"]
                #print(len(targets))
                if len(targets) <= targets_ais:
                    self.timestamp_list.append(timestamp)
                    print(targets)
                    print(timestamp)
                    return
                    
        print('Subset of timestamp_list selected. Length: '+str(len(self.timestamp_list)))
                
        
    def sample_list(self, l, sampling_interval):
        return l[0:len(l):sampling_interval]
    
    
    def flatten_list(self, l):
        return list(itertools.chain(*l))
    
    def get_data_dict(self, timestamp_list, cams_lenses):
        """
        Returns a dictionary containing timestamp_list as keys and images as values.
        """
        pass
    
    def load_batch(self, timestamps): # batch_size added, could be removed again?
        """
        Load a batch of images.
        The argument 'timestamps' is a list of timestamps to be included in the batch.
        Returns a numpy array of shape (batch_size,1920,2560,3) of type np.float32 with range [0,1].    print(ds.timestamp_list)
        
        """
        try:
            batch_size = len(timestamps)*self.images_per_timestamp
        except:
            batch_size = len(timestamps)*self.images_per_timestamp
            
        batch = np.empty((batch_size,)+(self.IMAGE_SHAPE), np.uint8) 
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        
        i = 0
        for timestamp in timestamps:
            for cam_lens in self.cams_lenses: 
                
                batch[i,:] = dl.load_image(timestamp, dl.TYPE_CAMERA,cam_lens)
                i+=1
                            
        return batch.astype('float32') / 255.
    
    
    def generate_batches(self, timestamp_list, batch_size): 
        """
        Returns a Python generator that generates batches of data indefinetily. To be used in Keras fit_generator().
        Works well, but waste the first batch and feed 2nd batch to fit_generator(). STILL THE CASE? 
        
        """
        numb_of_timestamps_in_batch = batch_size//self.images_per_timestamp
        #print(self.images_per_timestamp)
        while 1:
            t = 0
            for t in range(0,len(timestamp_list), numb_of_timestamps_in_batch):
                x_batch = self.load_batch(timestamp_list[t:t+numb_of_timestamps_in_batch]) # batch_size added, could be removed again?
    
                yield (x_batch, x_batch)
    
    def mask_image(self, image, grid):
        #masked_image = np.empty((grid[0]*grid[1], self.IMAGE_SHAPE))
        
        polygons = [(i*self.IMAGE_SHAPE[0]//grid[0] ,j*self.IMAGE_SHAPE[1]//grid[1]) for i in range(grid[0]+1) for j in range(grid[1]+1)]
        #polygons2 = [[(i,j),(i+1,j),(i,j+1),(i+1,j+1)] for i in range(grid[0]) for j in range(grid[1])]
        
        #g = np.meshgrid()
        
        #l = []
        #for i in range(grid[0]):
        #    for j in range(grid[1]):
            
        #im = Image.fromarray(image)
        
        #for i in range(grid[0]*grid[1]):
         #   masked_im = ImageDraw.Draw(im).polygon()
    
        print(polygons2)
        
        pass
        
        
        
    def write_timestamps_file(self, filename):
        with open(filename,'wb') as fp:
            pickle.dump(self.timestamp_list,fp)
        with open(filename+'_about.txt', 'w') as text:
            print('Metadata\nNumber of timestamps in timestamp_list: {}'.format(len(self.timestamp_list)), file=text)
        print('Timestamps and meta written to file. Length: '+str(len(self.timestamp_list)))
            
        
    def read_timestamps_file(self, filename):
        with open(filename,'rb') as fp:
            self.timestamp_list = pickle.load(fp)
            print('Timestamps read from file. Length: '+str(len(self.timestamp_list)))
 
class Figures():
    pass

if __name__ == "__main__":
    
    
    ### TESTS ###
    cams_lenses = [(1,1), (3,1)]
    ds = Dataset('all')
    print(ds.cams_lenses)
    #ds.read_timestamps_file('datasets/all/interval_60sec/timestamps')
    #ds.images_per_timestamp = len(ds.cams_lenses)
    #batch = ds.load_batch(ds.timestamp_list[:6])
    #print(batch.shape)
    #ds.read_metadata(ds.timestamp_list[0])
    
    
    """
    ### CREATE NEW DATASET ###
    ds = Dataset()
    ds.read_timestamps_file('datasets/all/interval_30min/timestamps')
    t_start = datetime.datetime(2017,10,23)
    t_end = datetime.datetime(2017,10,24)
    ds.select_subset(t_start, t_end)
    #ds.write_timestamps_file('datasets/dates/2017-10-23:24/interval_30min/timestamps')
    
    #ds.select_subset(targets_ais=0)
    
    
    
    ### CREATE NEW DATASET 2 ###

    ds = Dataset([(1,1)])
    ds.sampling_interval = 10
    ds.read_timestamps_file('datasets/all/interval_1sec/timestamps')
    ds.timestamp_list = ds.sample_list(ds.timestamp_list, ds.sampling_interval)
    ds.write_timestamps_file('datasets/all/interval_10sec/timestamps')
    """
    """
    #print(ds.timestamp_list)
    
    print(ds.timestamp_list_train)
    print(len(ds.timestamp_list_val))
    print(ds.timestamp_list_val)
    print(len(ds.timestamp_list_test))
    print(ds.timestamp_list_test)
    """
    path_results = '/home/kristoffer/Documents/mastersthesis/results/ex3/'
    ds = Dataset('all')
    ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy')
    #print(ds.timestamp_list_train[4*5206-1:4*5206+2])
    ds.mask_image(ds.timestamp_list_train[0], (2,2))
    
    #batch = ds.generate_batches(ds.timestamp_list, 12)
    
    #arr=next(batch)[0][7,:]     
    #arr=next(batch)[0][7,:]     
    #Image.fromarray(np.uint8(next(batch)[0][7,:]*255),'RGB').show()
    #Image.fromarray(np.uint8(next(batch)[0][7,:]*255),'RGB').show()
    #arr=next(batch)[0][1,:]     
    
    #Image.fromarray(np.uint8(arr*255),'RGB').show()
    



    

    
    
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
