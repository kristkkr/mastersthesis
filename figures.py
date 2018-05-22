#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:08:28 2017

@author: kristkkr

"""
#import pickle
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import pandas as pd

from PIL import Image


from keras.preprocessing.image import ImageDataGenerator

def plot_evaluation(path_results, n_images):
    
    recalls, precisions, false_positives = [],[],[]
    
    for metric_filename in os.listdir(path_results):
        if metric_filename.endswith('.txt'):
            with open(path_results+metric_filename, 'r') as fp:
                metrics = json.load(fp)
                
            recalls.append(metrics['recall'])
            precisions.append(metrics['precision'])
            false_positives.append(metrics['cluster_fp']//n_images)
    
    df = pd.DataFrame({'Recall': recalls, 'Precision': precisions, 'Mean false positives':false_positives})
    
    sns.regplot('Precision', 'Recall', df, fit_reg=False)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.savefig(path_results+'roc.pdf', format='pdf')

    plt.figure()
    sns.regplot('Mean false positives', 'Recall', df, fit_reg=False)
    _,xmax = plt.xlim()
    plt.xlim([0,xmax])
    plt.ylim([0,1])
    plt.savefig(path_results+'recall-fp.pdf', format='pdf')    
    
            

def plot_loss_history(path_results, train_val_ratio, single_im, n, ylim=0.01):
    """
    Saves a plot of loss history for in 'path_results' including all subdirectories with a 'n' moving average.
    Note: when models are stopped and restarted, the loss_history include more training than what is in the last saved model.
    """

    train_loss, val_loss = [],[]
    for subdir in os.walk(path_results):
        #print(subdir)
        subdir = subdir[0]
        if os.path.isdir(subdir):
            try:
                train_loss.extend(list(np.load(subdir+'/loss_history_train.npy')))
                if not single_im:
                    val_loss.extend(list(np.load(subdir+'/loss_history_val.npy')))
            except:
                continue            
    
    train_loss_avg = moving_average(np.asarray(train_loss), n=n)    
    if not single_im:
        val_loss_interp = np.interp(range(len(train_loss)), range(train_val_ratio-1,len(train_loss),train_val_ratio), np.asarray(val_loss))
        val_loss_avg = moving_average(val_loss_interp, n=n)  
    
    
    plt.figure()
    plt.gca()
    plt.clf()
    plt.plot(range(1,len(train_loss_avg)+1),train_loss_avg, label='Training', linewidth=0.5)
    if not single_im: plt.plot(range(1,len(val_loss_avg)+1),val_loss_avg, label='Validation', linewidth=0.5)
    
    plt.legend()
    
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.title('Loss during training')
    if not single_im:
        plt.xlabel('Time [batch]')
    else:
        plt.xlabel('Time [epoch]')
    plt.ylabel('Loss')
    plt.ylim(0,ylim)
    plt.savefig(path_results+'loss_history_avg_n'+str(n)+'.eps', format='eps')
    plt.close()
        
        
def create_reconstruction_plot(autoencoder, original_imgs, reconstructed_imgs, *args):
    
    fig_rows = 3
    fig_columns = min(len(original_imgs),4)    
    
    if args:
        masked_imgs = args[0]
        fig_rows += 1
    
    residual = np.empty((original_imgs.shape[:3]))
    
    gs = gridspec.GridSpec(fig_rows, fig_columns)
    gs.update(wspace=0.02, hspace=0.02)
    font_size=12
    rotation = 90
    scale=80
    plot = plt.figure(figsize=(autoencoder.dataset.IMAGE_SHAPE[1]*fig_columns//scale,autoencoder.dataset.IMAGE_SHAPE[0]*fig_rows//scale))
    for i in range(fig_columns):
        row=0
        # display original and masked
        ax = plt.subplot(gs[i])#fig_rows, fig_columns, i + 1)
        try:
            plt.imshow(original_imgs[i])
        except:
            continue
        ax.axis('off')  
        if i==0: 
            ax.set_ylabel('Original', rotation=rotation, fontsize=font_size)
        row += 1
        
        if args:
            ax = plt.subplot(gs[i+row*fig_columns]) 
            plt.imshow(masked_imgs[i])
            ax.axis('off')        
            if i==0: 
                ax.set_ylabel('Masked', rotation=rotation, fontsize=font_size)
            row += 1
        
        # display reconstructions
        ax = plt.subplot(gs[i+row*fig_columns]) 
        plt.imshow(reconstructed_imgs[i])
        ax.axis('off')        
        if i==0: 
            ax.set_ylabel('Reconstructed', rotation=rotation, fontsize=font_size)
        row += 1
        
        # reconstruction residual in grayscale
        residual = np.mean(np.abs(np.subtract(reconstructed_imgs[i], original_imgs[i])), axis=2)
        ax = plt.subplot(gs[i+row*fig_columns]) 
        plt.imshow(residual, cmap='gray')
        ax.axis('off')
        if i==0: 
            ax.set_ylabel('Residual', rotation=rotation, fontsize=font_size)
                
    #plt.subplots_adjust(wspace=0.02,hspace=0.02)
    
    plt.close(plot)
    return plot

def create_reconstruction_plot_single_image(autoencoder, original_and_masked_imgs, reconstructed_imgs, inpainting_grid):
    
    fig_rows = 3
    
    fig_columns = min(len(original_and_masked_imgs),3)
    #print(original_and_masked_imgs.shape, reconstructed_imgs.shape)
    
    residual = np.empty((original_and_masked_imgs.shape[:3]))
    inpainted = autoencoder.merge_inpaintings(reconstructed_imgs, inpainting_grid) #reconstructed_imgs[1:]
    
    """
    ### DELETE ###
    original_masked12 = original_and_masked_imgs[:2]
    del original_and_masked_imgs
    original_and_masked_imgs = []
    original_and_masked_imgs[:2] = original_masked12
    
    reconstructed0 = reconstructed_imgs[:2]
    del reconstructed_imgs
    reconstructed_imgs = []
    reconstructed_imgs[:2] = reconstructed0
    """
    gs = gridspec.GridSpec(fig_rows, fig_columns)
    gs.update(wspace=0.02, hspace=0.02)
    font_size=12
    rotation = 90
    scale=80
    plot = plt.figure(figsize=(autoencoder.dataset.IMAGE_SHAPE[1]*fig_columns//scale,autoencoder.dataset.IMAGE_SHAPE[0]*fig_rows//scale))
    
    for i in range(fig_columns):
        row=0
        # display original and masked
        ax = plt.subplot(gs[i+row*fig_columns])#fig_rows, fig_columns, i + 1)
        try:
            plt.imshow(original_and_masked_imgs[i])
        except:
            continue
        ax.axis('off')  
        if i==0: 
            ax.set_ylabel('Original', rotation=rotation, fontsize=font_size)
        row += 1
        
        # display reconstructions
        ax = plt.subplot(gs[i+row*fig_columns]) 
        
        ax.axis('off')        
        if i==0:
            plt.imshow(inpainted)
            ax.set_ylabel('Reconstructed', rotation=rotation, fontsize=font_size)
        else:
            plt.imshow(reconstructed_imgs[i-1])
        row += 1
        
        # reconstruction residual in grayscale
        residual[i] = np.mean(np.abs(np.subtract(reconstructed_imgs[i], original_and_masked_imgs[0])), axis=2)
        ax = plt.subplot(gs[i+row*fig_columns]) 
        
        ax.axis('off')
        if i==0:
            res_inpainted = np.mean(np.abs(np.subtract(inpainted, original_and_masked_imgs[0])), axis=2)
            plt.imshow(res_inpainted, cmap='gray')
            ax.set_ylabel('Residual', rotation=rotation, fontsize=font_size) 
        else:
            plt.imshow(residual[i-1], cmap='gray')
        row += 1
        """
        residual_reconstruction = np.abs(np.subtract(residual[0], residual[i]))
        ax = plt.subplot(gs[i+row*fig_columns])
        plt.imshow(residual_reconstruction, cmap='gray')
        ax.axis('off')
        if i==0: 
            ax.set_ylabel('residual[0]\n -residual[i]', rotation=rotation, fontsize=font_size)
        row += 1
        """
        """
        if i == 0:
            
            ax = plt.subplot(gs[i+row*fig_columns])
            plt.imshow(inpainted, cmap='gray')            
            ax.set_ylabel('Inpainted', rotation=rotation, fontsize=font_size)
            row += 1
            
            inpainted_residual = np.mean(np.abs(np.subtract(inpainted, original_and_masked_imgs[0])), axis=2)
            ax = plt.subplot(gs[i+row*fig_columns])
            plt.imshow(inpainted_residual, cmap='gray')            
            ax.set_ylabel('Inpainted \n residual', rotation=rotation, fontsize=font_size)
        """
    #plt.subplots_adjust(wspace=0.02,hspace=0.02)
    
    plt.close(plot)
    return plot

def show_detections(*args):
        
    rows, columns = len(args), 1 
    
    gs = gridspec.GridSpec(rows, columns)
    gs.update(wspace=0.02, hspace=0.02)
    scale=300
    plot = plt.figure(figsize=(2560*columns//scale,1920*rows//scale))
    
    for i in range(len(args)):
        ax = plt.subplot(gs[i])
        
        if len(args[i].shape)==2:
            plt.imshow(args[i], cmap='gray')
        else:
            plt.imshow(args[i])
        ax.axis('off')
    
    return plot

def show_images(images,rows,cols):

    scale = 400
    fig = plt.figure(figsize=(2560*cols//scale,1920*rows//scale)) #(figsize=(width,height))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace=0.02, hspace=0.02)

    for i in range(min(len(images),rows*cols)):
        ax = plt.subplot(gs1[i]) #rows,cols,i+1)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(images[i])

    return fig

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