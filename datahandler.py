#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:02:43 2018

@author: kristoffer
"""

import sys
import os
import itertools
import datetime

import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw
from random import shuffle, randint
from scipy.misc import imsave, imresize
import matplotlib.pyplot as plt

#from collections import Iterable

sys.path.insert(0,'/home/kristoffer/Documents/sensorfusion/polarlys')
from dataloader import DataLoader
from figures import show_images

class Dataset():
    NUMB_OF_CAMERAS = 4
    NUMB_OF_LENSES = 3
    IMAGE_SHAPE = (144,192,3) 
    IMAGE_SHAPE_ORIGINAL = (1920,2560,3)
    IMAGE_EXTENSION = '.jpg'
    
    def __init__(self, cams_lenses):
        self.path = '/nas0/'
        self.path_timestamps = None
        self.path_test = None
        self.timestamp_list = []
        self.timestamp_list_train = []
        self.timestamp_list_val = []
        self.timestamp_list_test = []
        self.init_cams_lenses(cams_lenses)
        self.images_per_timestamp = len(self.cams_lenses)    
        self.mask_color = (100,100,100)
        
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
    
    def select_subset(self, t_start=None, t_end=None, min_speed=None, max_speed=None, targets_ais_min=None, max_range=None):
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
            elif max_speed != None:
                if self.get_speed(dl,timestamp) < max_speed:
                    self.timestamp_list.append(timestamp)                    
            elif targets_ais_min != None:
                targets = dl.get_ais_targets(timestamp, own_pos=self.get_pos(dl, timestamp), max_range=max_range) #also returns ownship
                
                #print(targets)
                if len(targets) >= targets_ais_min:
                    self.timestamp_list.append(timestamp)

            t += 1     
            if len(self.timestamp_list)%10 == 0:
                print(len(self.timestamp_list),'/',len(tl))
            
        
                    
        print('Subset of timestamp_list selected. Length: '+str(len(self.timestamp_list)))
    
    def remove_hour_from_timestamplist(self, remove_hour, removal_freq):
            
        self.timestamp_list = list(self.timestamp_list)
        i=0
        removed_hours, actually_removed_hours = [0]*24, [0]*24 #'removed_hours' is a bad name, they are not all removed.
        while i < len(self.timestamp_list):
            if self.timestamp_list[i].hour in remove_hour:
                removed_hours[self.timestamp_list[i].hour] +=1
                if removed_hours[self.timestamp_list[i].hour] % removal_freq == 0:
                    actually_removed_hours[self.timestamp_list[i].hour] +=1
                    del self.timestamp_list[i]
                    i = i-1
            i = i+1    
        print('Number of timestemps for each hour removed:',actually_removed_hours)
        print('Subset of timestamp_list selected. Length: '+str(len(self.timestamp_list)))


    def remove_timestamp_illumination(self, remove_illumination_range, removal_freq):
        self.timestamp_list = list(self.timestamp_list)
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        i=0
        freq_counter=0
        
        while i < len(self.timestamp_list):
            try:
                illumination = int(np.mean(dl.load_image(self.timestamp_list[i], dl.TYPE_CAMERA, (3,1))))
            except:
                i = i+1 
                continue
            if illumination in remove_illumination_range:
                freq_counter +=1
                if freq_counter % removal_freq == 0:
                    del self.timestamp_list[i]
                    i = i-1
            i = i+1  
            if i%100 ==0:
                print(i)
        print('Subset of timestamp_list selected. Length: '+str(len(self.timestamp_list)))        
    
    def append_timestamps(self, second):
        
        self.timestamp_list = [t + datetime.timedelta(seconds=second) for t in self.timestamp_list]
        
        return
        
        
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
                    batch.resize(((i+1,)+(self.IMAGE_SHAPE))) 
                    if im.shape != self.IMAGE_SHAPE:
                        im = imresize(im,self.IMAGE_SHAPE,'bicubic')
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
                    draw.rectangle(rectangle_coordinates,fill=self.mask_color)
                    masked_batch[i] = np.asarray(im, dtype=np.uint8)
                except:
                    break
                i += 1
                
                ulc = (ulc[0],ulc[1]+mask_shape[0])
            ulc = (ulc[0]+mask_shape[1],0)
        
        return masked_batch.astype('float32') / 255., batch
        
    def mask_batch_randomly(self, batch, grid):
        """
        Mask a 'batch' with 'grid'. Allows prod(grid)>batch size - not all masks are included. 
        """
        masked_batch = np.copy(batch)
        rows, columns = grid 
        mask_shape = (self.IMAGE_SHAPE[0]//rows, self.IMAGE_SHAPE[1]//columns)
        
        try:
            self.ulc
        except:
            self.ulc = self.compute_ulc(grid)
            #print(self.ulc)
            
        shuffle(self.ulc)
        ulc = self.ulc
        #print(ulc)
        
        for i in range(len(batch)):
            #print(ulc[i], (ulc[i][0]+mask_shape[1],ulc[i][1]+mask_shape[0]))
            rectangle_coordinates = [ulc[i], (ulc[i][0]+mask_shape[1],ulc[i][1]+mask_shape[0])]
            im = Image.fromarray(np.uint8(batch[i]*255),'RGB') # remove scaleing
            draw = ImageDraw.Draw(im)
            draw.rectangle(rectangle_coordinates,fill=self.mask_color)
            masked_batch[i] = np.asarray(im, dtype=np.uint8)
            
        return masked_batch.astype('float32') / 255., batch    

    def compute_ulc(self,grid):
        """ Computes all ulc (upper left corners) for a grid in a terrible way."""
        rows, columns = grid 
        mask_shape = (self.IMAGE_SHAPE[0]//rows, self.IMAGE_SHAPE[1]//columns)
        
        ulc_single = (0,0)
        ulc = []
        i=0
        for x in range(columns):
            for y in range(rows):
                ulc.append(ulc_single)                                            
                ulc_single = (ulc_single[0],ulc_single[1]+mask_shape[0])
                
            ulc_single = (ulc_single[0]+mask_shape[1],0)
            i +=1
        return ulc
        
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
        try:
            speeds = np.load(self.path_timestamps+'speeds.npy')
            print('Speeds loaded')
        except:
            dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
            speeds = []
            for timestamp in self.timestamp_list:
                speeds.append(self.get_speed(dl, timestamp))
                if len(speeds)%10 == 0:
                    print(len(speeds))
        
        
        #fig, ax = plt.subplots()
        #ax.set(yscale="log")
        sns.distplot(speeds, kde=False)
        plt.ylabel('Number of examples')
        plt.xlabel('Speed [m/s]')
        plt.savefig(self.path_timestamps+'speeds_hist.pdf', format='pdf')
        np.save(self.path_timestamps+'speeds.npy', speeds)
    
    def explore_illumination(self):
        
        dfname = 'illumination'
                
        hourname, illuminationname = 'Hour of day', 'Mean illumination intensity'
        try:
            df = pd.read_pickle(self.path_timestamps+dfname)    
            print('Dataframe', dfname, 'loaded')
        except:    
            dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
            mean_illumination, hour_of_day = [], []
            counter = 0
            for timestamp in self.timestamp_list:
                hour_of_day.append(timestamp.hour)
                mean_illumination.append(np.mean(dl.load_image(timestamp, dl.TYPE_CAMERA, (3,1))))
                counter +=1
                if counter%100==0:
                    print(counter)

            df = pd.DataFrame({hourname: hour_of_day, illuminationname: mean_illumination})
            df.to_pickle(self.path_timestamps+dfname)
        
        figkind='hex'
        figname = 'illumination'+figkind
        sns.jointplot(x=hourname, y=illuminationname, data=df, kind=figkind, stat_func=None, ratio = 3,  marginal_kws=dict(bins=24), xlim=(0,24), ylim=(0,255))
        plt.savefig(self.path_timestamps+figname+'2.pdf', format='pdf')
        
        figname = 'hist-illumination'
        plt.figure()
        sns.distplot(df[illuminationname], kde=False, rug=False)
        plt.xlim([0,255])
        plt.ylabel('Number of examples')
        plt.savefig(self.path_timestamps+figname+'.pdf', format='pdf')
                
        figname = 'hist-hours'
        plt.figure()
        sns.distplot(df[hourname], bins=24, kde=False, rug=False)
        plt.xlim([0,24])
        plt.ylabel('Number of examples')
        plt.savefig(self.path_timestamps+figname+'.pdf', format='pdf')        
        
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
        if numb_of_images == 'all':
            timestamps = self.timestamp_list_test
        else:
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
        
    def show_raw_data(self, rows, cols):
        
        n_images = rows*cols
        batch = np.empty((n_images,)+(1920,2560,3), np.uint8) 
        
        dl = DataLoader(self.path, sensor_config='/home/kristoffer/Documents/sensorfusion/polarlys/dataloader.json')
        path_data = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/all/interval_5sec/'
        timestamps = np.load(path_data+'timestamps.npy')
        shuffle(timestamps)
        timestamps = timestamps[:n_images*2]
        i = 0
        for timestamp in timestamps:
            #for cam_lens in self.cams_lenses:
            cam_lens = self.cams_lenses[randint(0,1)]
            try:
                im = dl.load_image(timestamp, dl.TYPE_CAMERA, cam_lens)
                batch[i] = im
                i+=1                    
            except:
                continue

        fig = show_images(batch, rows, cols)
        fig.savefig('raw_data4.jpg')
        
                
if __name__ == "__main__":
    
    ds = Dataset([(1,1), (3,1)])
    ds.show_raw_data(5,6)
    
    """
    ### TESTS ###
    #cams_lenses = [(1,1), (3,1)]
    ds = Dataset('all')
    
    ds.read_timestamps_file('datasets/all/interval_60min/timestamps.npy')
    ds.images_per_timestamp = len(ds.cams_lenses)
    batch,_ = ds.load_batch([ds.timestamp_list[0]], [])
    #print(batch.shape)
    #masked,_ = ds.mask_batch(batch, grid=(3,3))
    masked_images,_ = ds.mask_batch_randomly(batch[:2], grid=(2,1))
    #image = batch[0,:]
    #masked_images = ds.mask_image(image,3,3)
    Image.fromarray(np.uint8(masked_images[0]*255),'RGB').show()
    """
    
    """
    ### CREATE NEW DATASET ###
    #ds = Dataset('all')
    
    #ds.path_timestamps = 'datasets/new2704/speed>6/interval_5sec/removed_illumination/'
       
    #ds.read_timestamps_file(ds.path_timestamps+'data_timestamp_list_val.npy')
    #ds.timestamp_list = ds.timestamp_list[:1]
    #ds.select_subset(min_speed=6)
    #ds.select_subset(targets_ais_min=2, max_range=1000)
    #ds.timestamp_list = ds.sample_list(ds.timestamp_list,2)
    #ds.remove_hour_from_timestamplist([1,2,3,4,20,21,22,23], 2)
    #ds.remove_timestamp_illumination(range(50),2)
    #ds.append_timestamps(2)
    
    #ds.write_timestamps_file('datasets/new2704/speed>6/interval_5sec/removed_illumination/data_timestamp_list_val2')
    """
    
    """
    ### MOVE TEST DATA TO DIRECTORY ###
    ds = Dataset([(1,1),(3,1)])
    path_data = '/home/kristoffer/Documents/mastersthesis/datasets/new2704/speed<6/' 
    ds.timestamp_list_test = np.load(path_data+'timestamps.npy')
    ds.copy_images_to_dir('all', '/home/kristoffer/Documents/mastersthesis/datasets/new2704/ais/interval_5sec/speed<6/')
    """
    
    """
    ### EXPLORE ILLUMINATION ###
    ds = Dataset('all')
    ds.path_timestamps = 'datasets/new2704/all/interval_30min/'
    ds.read_timestamps_file(ds.path_timestamps+'timestamps.npy')
    ds.explore_illumination()
    """
    
    
    """
    ### LOAD ENTIRE DATASET ###
    ds = Dataset('all')
    ds.get_all_timestamps_list()
    ds.write_timestamps_file('datasets/new2704/all/interval_1sec/timestamps')
    """
    

    