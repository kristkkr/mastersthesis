#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:08:28 2017

@author: kristkkr

"""
#import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
#from skimage import filters, morphology
from keras.preprocessing.image import ImageDataGenerator


def plot_loss_history(path_results, train_val_ratio, single_im, n):
    """
    Saves a plot of loss history for in 'path_'results with a 'n' moving average.
    """
        
    train_loss = np.load(path_results+'loss_history_train.npy')
    train_loss_avg = moving_average(train_loss, n=n)
    
    if not single_im:
        val_loss = np.load(path_results+'loss_history_val.npy')
        val_loss_interp = np.interp(range(len(train_loss)), range(train_val_ratio-1,len(train_loss),train_val_ratio), val_loss)
        val_loss_avg = moving_average(val_loss_interp, n=n)
    
    plt.figure()
    plt.gca()
    plt.clf()
    plt.plot(range(1,len(train_loss_avg)+1),train_loss_avg, label='Training', linewidth=0.5)
    if not single_im: plt.plot(range(1,len(val_loss_avg)+1),val_loss_avg, label='Validation', linewidth=0.5)
    
    plt.legend()
    
    plt.gca().yaxis.grid(True)
    plt.title('Loss during training')
    if not single_im:
        plt.xlabel('Time [batch]')
    else:
        plt.xlabel('Time [epoch]')
    plt.ylabel('Loss')
    plt.savefig(path_results+'loss_history_avg_n'+str(n)+'.eps', format='eps')
    plt.close()
        
        
def create_reconstruction_plot(autoencoder, original_imgs, masked_imgs, reconstructed_imgs):
    
    fig_rows = 4
    fig_columns = len(masked_imgs)
    residual = np.empty((masked_imgs.shape[:3]))
    
    gs = gridspec.GridSpec(fig_rows, fig_columns)
    gs.update(wspace=0.02, hspace=0.02)
    scale=80
    plot = plt.figure(figsize=(2560*fig_columns//scale,1920*fig_rows//scale))
    for i in range(fig_columns):

        # display original and masked
        ax = plt.subplot(gs[i])#fig_rows, fig_columns, i + 1)
        try:
            plt.imshow(original_imgs[i])
        except:
            continue
        ax.axis('off')  
        
        ax = plt.subplot(gs[i+fig_columns]) 
        plt.imshow(masked_imgs[i])
        ax.axis('off')        
        
        # display reconstructions
        ax = plt.subplot(gs[i+2*fig_columns]) 
        plt.imshow(reconstructed_imgs[i])
        ax.axis('off')        
        
        # reconstruction residual in grayscale
        residual = np.mean(np.abs(np.subtract(reconstructed_imgs[i], original_imgs[i])), axis=2)
        ax = plt.subplot(gs[i+3*fig_columns]) 
        plt.imshow(residual, cmap='gray')
        ax.axis('off')
        
    #plt.subplots_adjust(wspace=0.02,hspace=0.02)
    
    plt.close(plot)
    return plot

def create_reconstruction_plot_single_image(autoencoder, original_imgs, masked_imgs, reconstructed_imgs, inpainting_grid):
    """ not finished """
    
    fig_rows = 4
    if not inpainting_grid==None: 
        fig_rows +=2
    
    fig_columns = len(masked_imgs)
    residual = np.empty((masked_imgs.shape[:3]))
    
    gs = gridspec.GridSpec(fig_rows, fig_columns)
    gs.update(wspace=0.02, hspace=0.02)
    scale=80
    plot = plt.figure(figsize=(2560*fig_columns//scale,1920*fig_rows//scale))
    for i in range(fig_columns):

        # display original and masked
        ax = plt.subplot(gs[i])#fig_rows, fig_columns, i + 1)
        try:
            plt.imshow(original_imgs[i])
        except:
            continue
        ax.axis('off')  
        
        ax = plt.subplot(gs[i+fig_columns]) 
        plt.imshow(masked_imgs[i])
        ax.axis('off')        
        
        
        # display reconstructions
        ax = plt.subplot(gs[i+2*fig_columns]) 
        plt.imshow(reconstructed_imgs[i])
        ax.axis('off')        
        
        # reconstruction residual in grayscale
        if not inpainting_grid==None:
            residual[i] = np.mean(np.abs(np.subtract(reconstructed_imgs[i], masked_imgs[0])), axis=2)
        else:
            residual[i] = np.mean(np.abs(np.subtract(reconstructed_imgs[i], original_imgs[i])), axis=2)
        ax = plt.subplot(gs[i+3*fig_columns]) 
        plt.imshow(residual[i], cmap='gray')
        ax.axis('off')
        
        if not inpainting_grid==None:
            residual_reconstruction = np.abs(np.subtract(residual[0], residual[i]))
            ax = plt.subplot(gs[i+4*fig_columns])
            plt.imshow(residual_reconstruction, cmap='gray')
            ax.axis('off')
            
            if i == 0:
                inpainted = autoencoder.merge_inpaintings(reconstructed_imgs, inpainting_grid)
                ax = plt.subplot(gs[i+5*fig_columns])
                plt.imshow(inpainted, cmap='gray')            
            
    #plt.subplots_adjust(wspace=0.02,hspace=0.02)
    
    plt.close(plot)
    return plot

def show_raw_data(rows,cols,path_save):

    datagen = ImageDataGenerator(rescale=1./255)
    path_windows = 'C:\\Users\\kristkkr\\OneDrive - NTNU\\Prosjektoppgave\\Datasett\\Hurtigruten\\'
    
    n = rows*cols
    scale = 300
    fig = plt.figure(figsize=(1280*cols//scale,720*rows//scale)) #(figsize=(width,height))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace=0.02, hspace=0.02)

    for image in datagen.flow_from_directory(path_windows, shuffle=True, target_size = (720,1280), class_mode=None, batch_size=n):
        for i in range(n):
            ax = plt.subplot(gs1[i]) #rows,cols,i+1)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(image[i])
        break
    fig.savefig(path_save+'raw_data.png')
    return  

def add_mask(image,mask):
    x,y,_ = image.shape
    for i in range(x):
        for j in range(y):
            if mask[i,j] == 1:
                image[i,j,:] = [1,0,0]
    return image

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n        

def insert_leading_zeros(numb, n):
    # converts numb to string and add n leading zeros
    numb_str = str(numb)
    while len(numb_str)<n:
        numb_str = '0'+numb_str
    return numb_str

def remove_subplot_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax