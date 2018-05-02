#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:02:43 2018

@author: kristoffer
"""

import sys
import os
import itertools
#import datetime

import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw
from random import shuffle
from scipy.misc import imsave
import matplotlib.pyplot as plt

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
        self.path_timestamps = None
        self.timestamp_list = []
        #self.timestamp_list_shuffled = []
        self.timestamp_list_train = []
        self.timestamp_list_val = []
        self.timestamp_list_test = []
        self.init_cams_lenses(cams_lenses)
        self.images_per_timestamp = len(self.cams_lenses)    
        
    def init_cams_lenses(self, cams_lenses):
        if cams_lenses == 'all':
            self.cams_lenses = [(i,j) for i in range(self.NUMB_OF_CAMERAS) for j in range(self.NUMB_OF_LENSES)]
            print('All cameras and lenses included in dataset')
        else: 
            self.cams_lenses = cams_lenses
            print('Selection of cameras and lenses:', self.cams_lenses)
        
    def get_all_timestamps_list(self):
        """
        Returns the instance variable timestamp_list. 
        Bad method in most ways, but needs only to be run once every time the entire dataset is updated.
        """
        
        hour_list = sorted([hour for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour])
        #self.timestamp_list = sorted([timestamp for date in os.listdir(self.path) if '201' in date for hour in os.listdir(self.path+date) if '201' in hour for timestamp in os.listdir(self.path+date+'/'+hour+'/Cam0/Lens0') if '.jpg' in timestamp])
        print('Lenght of hour_list: '+str(len(hour_list)))
                
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        for hour in hour_list:
            try:
                timestamps_in_hour = sorted([name[:19] for name in os.listdir(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0') if '.jpg' in name]) # time_list contains filename strings
                
                for i in range(len(timestamps_in_hour)):
                    timestamps_in_hour[i] = dl.get_time_from_basename(self.path+hour[:10]+'/'+hour+'/Cam0/Lens0/'+timestamps_in_hour[i])  # get_time_from_basetime returns datetime instances
              
                self.timestamp_list.append(timestamps_in_hour)
                print('Hour added: '+hour)
                
            except:
                print('Hour not added: '+hour)
                
        self.timestamp_list = self.flatten_list(self.timestamp_list)

            
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
    
    def select_subset(self, t_start=None, t_end=None, min_speed=None, targets_ais_min=None, max_range=None):
        """ 
        The arguments 't_start' and 't_end' is the start and end of the time t.
        'min_speed'
        'targets_ais_min' is the minimum number of ais-targets required for including the timestamp.
        'max_range' is the corresponding distance to ais-targets. 
        
        """        
        tl = ds.timestamp_list
        del ds.timestamp_list
        #print('Timestamp_list now empty')
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        ds.timestamp_list = []
        
        t=0             
        for timestamp in tl:
            if t_start != None or t_end != None:
                if t_start <= timestamp and timestamp < t_end:
                    self.timestamp_list.append(timestamp)
            elif min_speed != None:
                if self.get_speed(dl,timestamp) > min_speed:
                    self.timestamp_list.append(timestamp)
            elif targets_ais_min != None:
                targets = dl.get_ais_targets(timestamp, own_pos=self.get_pos(dl, timestamp), max_range=max_range) #also returns ownship
                
                #print(targets)
                if len(targets) >= targets_ais_min:
                    self.timestamp_list.append(timestamp)

            t += 1     
            #if len(self.timestamp_list)%10 == 0:
            print(t,'/',len(tl))
            print(len(self.timestamp_list))
        
                    
        print('Subset of timestamp_list selected. Length: '+str(len(self.timestamp_list)))
                
        
    def sample_list(self, l, sampling_interval):
        return l[0:len(l):sampling_interval]
    
    
    def flatten_list(self, l):
        return list(itertools.chain(*l))
    
    
    def load_batch(self, timestamps, failed_im_load): #, mask_image=false, grid_size=None) # batch_size added, could be removed again?
        """
        Load a batch of images.
        The argument 'timestamps' is a list of timestamps to be included in the batch.
        Returns a numpy array of shape (batch_size,1920,2560,3) of type np.float32 with range [0,1].
        batch size depends on if images are correctly loaded.
        """
                           
        batch = np.empty((0,)+(self.IMAGE_SHAPE), np.uint8) 
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        
        i = 0
        for timestamp in timestamps:
            for cam_lens in self.cams_lenses: 
                try:
                    im = dl.load_image(timestamp, dl.TYPE_CAMERA, cam_lens)
                    #if not im == []: 
                    #if mask_image:
                    #print(im)
                    batch.resize(((i+1,)+(self.IMAGE_SHAPE))) # not double tested if correct. if wrong, we train at only zeros without noiticing
                    batch[i,:] = im
                
                    i+=1
                except:
                    failed_im_load.append((timestamp, cam_lens))
        
        return batch.astype('float32') / 255., failed_im_load
    
    
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
                x_batch = self.load_batch(timestamp_list[t:t+numb_of_timestamps_in_batch]) 
    
                yield (x_batch, x_batch)
    
    def mask_image(self, image, grid): #rows, columns):
        """
        Returns two numpy arrays of size rows*columns containing the masked and original and versions of the argument image
        """
        #rows, columns = grid
        
        original_image_batch = np.empty((np.prod(grid),)+ self.IMAGE_SHAPE)
        for i in range(len(original_image_batch)):
            original_image_batch[i] = image
        
        return self.mask_batch(original_image_batch, grid)
    

    def mask_batch(self, batch, grid):
        """
        Mask a 'batch' with 'grid'. In some cases, len(batch)< prod(grid) since cam_lens(1,1) is missing. this leads to unbalanced batches of masks, but should not be a too large problem.
        """
        rows, columns = grid 
        
        masked_batch = np.copy(batch)
        mask_shape = (self.IMAGE_SHAPE[0]//rows, self.IMAGE_SHAPE[1]//columns)
        ulc = (0,0) # upper left corner coordinates
        
        i = 0
                        
        for x in range(columns):
            for y in range(rows):
                try:
                    rectangle_coordinates = [ulc, (ulc[0]+mask_shape[1],ulc[1]+mask_shape[0])]
                    im = Image.fromarray(np.uint8(batch[i]*255),'RGB') # remove scaleing
                    draw = ImageDraw.Draw(im)
                    draw.rectangle(rectangle_coordinates,fill=0)
                    masked_batch[i] = np.asarray(im, dtype=np.uint8)
                except:
                    break
                i += 1
                                
                ulc = (ulc[0],ulc[1]+mask_shape[0])
            ulc = (ulc[0]+mask_shape[1],0)
        
        return masked_batch.astype('float32') / 255., batch
        
        
        
    def write_timestamps_file(self, filename):
        #with open(filename,'wb') as fp:
        #    pickle.dump(self.timestamp_list,fp)
        np.save(filename, self.timestamp_list)
        with open(filename+'_about.txt', 'w') as text:
            print('Metadata\nNumber of timestamps in timestamp_list: {}'.format(len(self.timestamp_list)), file=text)
        print('Timestamps and meta written to file. Length: '+str(len(self.timestamp_list)))
            
        
    def read_timestamps_file(self, filename):
        #with open(filename,'rb') as fp:
        #    self.timestamp_list = pickle.load(fp)
        self.timestamp_list = np.load(filename)
        print('Timestamps read from file. Length: '+str(len(self.timestamp_list)))
    
    def explore_speeds(self):
        """
        Returns histogram of attributes of dataset
        """
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        speeds = []
        for timestamp in self.timestamp_list:
            speeds.append(self.get_speed(dl, timestamp))
            if len(speeds)%10 == 0:
                print(len(speeds))
                
        sns.distplot(speeds, rug=True)
        plt.savefig(self.path_timestamps+'speeds_hist.pdf', format='pdf')
        np.save(self.path_timestamps+'speeds.npy', speeds)
    
    def explore_illumination(self):
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        mean_illumination, hour_of_day = [], []
        for timestamp in self.timestamp_list:
            hour_of_day.append(timestamp.hour)
            mean_illumination.append(255-np.mean(dl.load_image(timestamp, dl.TYPE_CAMERA, (3,1))))
        #print(hour_of_day, mean_illumination)
        
        #dataframe = pd.DataFrame({'hour': hour_of_day, 'illumination': mean_illumination})
        #print(dataframe)
        x1 = pd.Series(hour_of_day)
        x2 = pd.Series(mean_illumination)
        #sns.jointplot(x="hour_of_day", y="mean_illumination", data=dataframe, kind='kde')
        sns.jointplot(x1, x2, kind='kde')
        plt.savefig(self.path_timestamps+'illumination.pdf', format='pdf')
        #np.save(self.path_timestamps+'speeds.npy', speeds)
        
    def get_speed(self, dl, timestamp):
        vel = dl.get_seapath_data(timestamp)['velocity']
        return np.linalg.norm(vel)
    
    def get_pos(self, dl, timestamp):
        pos = dl.get_seapath_data(timestamp)['position']
        return pos
    
    def copy_images_to_dir(self, numb_of_images, dir_save):
        """ 
        Move a subset of test data to a directory. 
        """
        numb_of_timestamps = numb_of_images// self.images_per_timestamp
        timestamps = self.timestamp_list_test[:numb_of_timestamps]
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        FILE_NAME_FORMAT_SEC = '{:%Y-%m-%d-%H_%M_%S}'
        
        image = []
        
        for timestamp in timestamps:
            for cam_lens in self.cams_lenses: 
                image = dl.load_image(timestamp, dl.TYPE_CAMERA, cam_lens)
                try:
                    image.shape
                except:
                    print('image cam_lens not loaded')
                    continue
                name = FILE_NAME_FORMAT_SEC.format(timestamp)+str(cam_lens)+'.jpg'
                imsave(dir_save+name, image)
                
if __name__ == "__main__":
    
    """
    ### TESTS ###
    #cams_lenses = [(1,1), (3,1)]
    ds = Dataset('all')
    
    ds.read_timestamps_file('datasets/all/interval_60min/timestamps.npy')
    ds.images_per_timestamp = len(ds.cams_lenses)
    batch = ds.load_batch([ds.timestamp_list[0]])
    image = batch[0,:]
    masked_images = ds.mask_image(image,3,3)
    #Image.fromarray(np.uint8(masked_images[4]*255),'RGB').show()
    """
   
    """
    ### CREATE NEW DATASET ###
    ds = Dataset('all')
    #ds.path_timestamps = 'datasets/new2704/all/interval_5sec/'
    ds.read_timestamps_file('results/ex30/data_timestamp_list_test.npy')
    #ds.timestamp_list = ds.timestamp_list[:1]
    #ds.select_subset(min_speed=6)
    ds.select_subset(targets_ais_min=2, max_range=1000)
    #ds.timestamp_list = ds.sample_list(ds.timestamp_list,60*30)
    ds.write_timestamps_file('datasets/new2704/ais/interval_5sec/timestamps_list_test')
    """
    
    ### MOVE TEST DATA TO DIRECTORY ###
    ds = Dataset([(1,1),(3,1)])
    path_data = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/' 
    ds.timestamp_list_test = np.load(path_data+'timestamps_list_test.npy')
    ds.copy_images_to_dir(20, path_data+'test/')
    
    """
    ### EXPLORE ILLUMINATION ###
    ds = Dataset('all')
    ds.path_timestamps = 'datasets/new2704/all/interval_30min/'
    ds.read_timestamps_file(ds.path_timestamps+'timestamps.npy')
    ds.explore_illumination()
    """
    
    
    """
    ### LOAD DATASET ###
    ds = Dataset('all')
    ds.get_all_timestamps_list()
    ds.write_timestamps_file('datasets/new2704/all/interval_1sec/timestamps')
    """
    
    #print(ds.timestamp_list)
    
    """
    ### load changing batch size testing ###
    path_results = '/home/kristoffer/Documents/mastersthesis/results/ex6/'
    ds = Dataset(cams_lenses = [(1,1),(3,1)])
    ds.timestamp_list_train = np.load(path_results+'data_timestamp_list_train.npy')
    
    index = 183*4
    indices = range(index,index+4)
    #print(len(ds.timestamp_list_train))
    print(ds.timestamp_list_train[indices])
    
    batch = ds.load_batch(ds.timestamp_list_train[indices])
    print(batch.shape)
    
    """
