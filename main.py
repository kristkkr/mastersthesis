#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:18:32 2018

@author: kristoffer
"""

from datahandler import Dataset
from autoencoder import Autoencoder
        
"""
if __name__ == "__main__":
                            
    # initialize model
    ae = Autoencoder()
    ae.create_autoencoder()
    ae.model.summary()
    
    # initialize data
    ds = Dataset()
    
    ds.get_timestamp_list(randomize=True)
    ds.size = len(ds.timestamp_list)
    
    # hyperparameters
    epochs = 1
    batch_size = 24
    val_split = 0.1
    ae.train(ds, epochs, batch_size, val_split)
    
"""
