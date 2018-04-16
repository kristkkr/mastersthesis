#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:08:28 2017

@author: kristkkr

"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage import filters, morphology
from keras.preprocessing.image import ImageDataGenerator


def save_to_directory(autoencoder, loss_history, failed_im_load, epoch, batch,timestamp_index, train_val_ratio, model_freq, loss_freq, reconstruct_freq, n_move_avg, inpainting_grid=None, single_im=False):
    
    # THIS FUNCTION DOES NOT BELONG HERE. autoencoder?
    
    epoch_str = insert_leading_zeros(epoch+1,5)
    batch_str = insert_leading_zeros(batch+1,6)
    
    if single_im:
        freq_counter = epoch
    else:
        freq_counter = batch
    
    if (freq_counter+1)%loss_freq == 0:
        np.save(autoencoder.path_results+'loss_history_train', loss_history.train_loss)
        np.save(autoencoder.path_results+'loss_history_val', loss_history.val_loss)
        save_plot_loss_history(autoencoder.path_results, train_val_ratio, n_move_avg, single_im)
        np.save(autoencoder.path_results+'failed_imageload_during_training', failed_im_load)
        with open(autoencoder.path_results+'failed_imageload_during_training.txt', 'w') as text: 
            print('Timestamps and cam_lens of failed batches:\n{}'.format(failed_im_load), file=text)

    if (freq_counter+1)%model_freq == 0: 
        autoencoder.model.save(autoencoder.path_results+'epoch'+epoch_str+'_batch'+batch_str+'.hdf5')
        print('Model saved')
    if (freq_counter+1)%reconstruct_freq == 0: 
        autoencoder.test_inpainting(what_data='val', timestamp_index=timestamp_index, numb_of_timestamps=1, epoch = epoch, batch = batch, inpainting_grid=inpainting_grid)

def save_plot_loss_history(path_results, train_val_ratio, n, single_im):
    # THIS FUNCTION DOES NOT BELONG HERE. datahandler -> Figures?
    # n: in moving average
        
    train_loss = np.load(path_results+'loss_history_train.npy')
    train_loss_avg = moving_average(train_loss, n=n)
    
    if not single_im:
        val_loss = np.load(path_results+'loss_history_val.npy')
        val_loss_interp = np.interp(range(len(train_loss)), range(train_val_ratio-1,len(train_loss),train_val_ratio), val_loss)
        val_loss_avg = moving_average(val_loss_interp, n=n)
    
    
    fig = plt.figure()
    ax = plt.gca()
    plt.clf()
    plt.plot(range(1,len(train_loss_avg)+1),train_loss_avg, label='Training', linewidth=0.5)
    if not single_im: plt.plot(range(1,len(val_loss_avg)+1),val_loss_avg, label='Validation', linewidth=0.5)
    
    plt.legend()
    
    #handles, labels = ax.get_legend_handles_labels()
    #ax1.legend()

    plt.gca().yaxis.grid(True)
    plt.title('Loss during training')
    plt.xlabel('Time [batch]')
    plt.ylabel('Loss [MAE]')
    plt.savefig(path_results+'loss_history_avg_n'+str(n)+'.eps', format='eps')
    plt.close()
        
        
def create_reconstruction_plot(original_imgs, masked_imgs, reconstructed_imgs):
    fig_rows = 4
    
    batch_size = len(original_imgs)
    gs = gridspec.GridSpec(fig_rows, batch_size)
    gs.update(wspace=0.02, hspace=0.02)
    
    scale=80
    plot = plt.figure(figsize=(2560*batch_size//scale,1920*fig_rows//scale))#figsize=(30, 20)) #(figsize=(width,height))
    for i in range(batch_size):

        # display original
        ax = plt.subplot(gs[i])#fig_rows, batch_size, i + 1)
        try:
            plt.imshow(original_imgs[i])
        except:
            continue
        ax.axis('off')  
        
        # display masked
        ax = plt.subplot(gs[i+batch_size]) 
        plt.imshow(masked_imgs[i])
        ax.axis('off')        
        
        # display reconstruction
        ax = plt.subplot(gs[i+2*batch_size]) #fig_rows, batch_size, i + 1 + batch_size)
        plt.imshow(reconstructed_imgs[i])
        ax.axis('off')

        # reconstruction error in grayscale
        err0 = np.abs(reconstructed_imgs[i,:,:,0] - original_imgs[i,:,:,0])
        err1 = np.abs(reconstructed_imgs[i,:,:,1] - original_imgs[i,:,:,1])
        err2 = np.abs(reconstructed_imgs[i,:,:,2] - original_imgs[i,:,:,2])
        
        abs_err_gray = (err0+err1+err2)/3
        
        ax = plt.subplot(gs[i+3*batch_size]) #fig_rows, batch_size, i + 1 + 2*batch_size)
        plt.imshow(abs_err_gray, cmap='gray')
        ax.axis('off')

    #plt.subplots_adjust(wspace=0.02,hspace=0.02)
    
    plt.close(plot)
    return plot

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
   

def remove_subplot_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
            
    return ax