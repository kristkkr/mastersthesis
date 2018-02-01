#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:02:43 2018

@author: kristoffer
"""

import numpy as np

class Dataset():
    def __init__(self, path):
        self.size = 0
        self.path = path
        self.data = {}
        
    def get_timestamp_dict(self, t_start, t_end, min_speed=0):
        """
        Returns a dictionary containing the timestamps when the ownship speed is greater than 'min_speed'.
        The argument 't_start' is the start of the time t.
        The argument 't_end' is the end of the time t
        
        pseudo:
            
        iterte through directories
        if speed>min_speed
            self.data.append
        """
        
        raise NotImplementedError
        
    def vel_to_speed(self, vel):
        """
        Returns the speed of the ownship.
        vel is the velocity of the ownship in NED given in m/s
        """
        raise NotImplementedError
    
    
path = '/nas0/'

ds = Dataset(path)

