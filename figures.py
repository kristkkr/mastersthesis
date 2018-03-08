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

"""
def test(autoencoder, path_test, path_results, batch_size, n_batches, shuffle=False, image_shape=(720,1280,3)):
    target_size = image_shape[:2]

    test_datagen = ImageDataGenerator(rescale=1./255)
    batch = 0
    for test_batch in test_datagen.flow_from_directory(path_test, target_size=target_size, class_mode=None, batch_size=batch_size, shuffle=shuffle):
        original_imgs = test_batch
        reconstructed_imgs = autoencoder.predict_on_batch(original_imgs).reshape((batch_size,)+image_shape)
        fig = create_reconstruction_plot(original_imgs, reconstructed_imgs, batch_size)
        batch_str = insert_leading_zeros(batch,4)
        fig.savefig(path_results+'test_batch'+batch_str+'.png', transparent=False, bbox_inches='tight')
        batch +=1
        if batch >= n_batches:
            return 
"""
def save_to_directory(autoencoder, loss_history, epoch, batch, train_val_ratio, model_freq, loss_freq, n_move_avg):
    # THIS FUNCTION DOES NOT BELONG IN FIGURES
    
    epoch_str = insert_leading_zeros(epoch,2)
    batch_str = insert_leading_zeros(batch,4)

    if batch%loss_freq == 0:
        np.save(autoencoder.path_results+'loss_history_train', loss_history.train_loss)
        np.save(autoencoder.path_results+'loss_history_val', loss_history.val_loss)
        try:
            save_plot_loss_history(autoencoder.path_results, loss_history, train_val_ratio, n_move_avg)
        except: 
            print('Loss history plot not saved')
    #if batch%reconstruction_freq == 0:
        #save_reconstructions_during_training(autoencoder.model, path_results, val_batch, image_shape, val_batch_size, epoch_str, batch_str, loss_history)
    if batch%model_freq == 0:
        autoencoder.model.save(autoencoder.path_results+'epoch'+epoch_str+'_batch'+batch_str+'_valloss'+str(round(loss_history.val_loss[-1],4))+'.hdf5')
        print('Model saved')


def save_plot_loss_history(path_results, loss_history, train_val_ratio, n):
    # n: in moving average
        
    #train_loss = np.load(path_results+'loss_history_train.npy')
    #val_loss = np.load(path_results+'loss_history_val.npy')
    
    # moving average
    #train_loss_avg = moving_average(loss_history.train_loss, n=n)
    #val_loss_avg = moving_average(loss_history.val_loss, n=n)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.clf()
    plt.plot(loss_history.train_loss[:-(train_val_ratio-1)], label='Training', marker='o')
    plt.plot(range(0,len(loss_history.train_loss),train_val_ratio), loss_history.val_loss, label='Validation', marker='o')
    #plt.xticks([x*1830 for x in range(21)],range(21))
    plt.legend()
    
    #handles, labels = ax.get_legend_handles_labels()
    #ax1.legend()

    plt.gca().yaxis.grid(True)
    plt.title('Loss during training')
    plt.xlabel('Time [batch]')
    plt.ylabel('Loss [MAE]')
    plt.savefig(path_results+'loss_history_avg_n'+str(n)+'.eps', format='eps')

        
def save_reconstructions_during_training(autoencoder, path_results, val_batch, image_shape, batch_size, epoch_str, batch_str, loss_history):

    original_imgs = val_batch

    reconstructed_imgs = autoencoder.predict_on_batch(original_imgs)
    reconstructed_imgs = reconstructed_imgs.reshape((batch_size,)+image_shape)

    plot = create_reconstruction_plot(original_imgs, reconstructed_imgs, batch_size)
       
    plot.savefig(path_results+'epoch'+epoch_str+'_batch'+batch_str+'_valloss'+str(round(loss_history.val_loss[-1],4))+'.png')
    print('Plot saved')
    plt.close(plot)

        
def create_reconstruction_plot(original_imgs, reconstructed_imgs, batch_size):
    fig_rows = 6


    gs = gridspec.GridSpec(fig_rows, batch_size)
    gs.update(wspace=0.02, hspace=0.02)
    font_size=12
    rotation = 90
    
    threshold = 55/255.
    opening_size = 1

    gauss_sigma = 1.5
    gauss_threshold = 45/255.
    gauss_opening_size = 1

    scale = 300
    plot = plt.figure(figsize=(1280*batch_size//scale,720*fig_rows//scale))#figsize=(30, 20)) #(figsize=(width,height))

    for i in range(batch_size):

        # display original
        ax = plt.subplot(gs[i])
        plt.imshow(original_imgs[i])
        #ax = remove_subplot_spines(ax)
        if i==0: 
            ax.set_ylabel('Original', rotation=rotation, fontsize=font_size)
        #else:   
        plt.yticks([],[])        
        plt.xticks([],[])

        # display reconstruction
        ax = plt.subplot(gs[i+batch_size]) #ax = plt.subplot(fig_rows, batch_size, i + 1 + batch_size)
        plt.imshow(reconstructed_imgs[i])
        #ax = remove_subplot_spines(ax)
        if i==0: 
            ax.set_ylabel('Reconstructed', rotation=rotation, fontsize=font_size)
        #else:   
        plt.yticks([],[])
        plt.xticks([],[]) 

        # reconstruction error in grayscale
        err0 = np.abs(reconstructed_imgs[i,:,:,0] - original_imgs[i,:,:,0])
        err1 = np.abs(reconstructed_imgs[i,:,:,1] - original_imgs[i,:,:,1])
        err2 = np.abs(reconstructed_imgs[i,:,:,2] - original_imgs[i,:,:,2])
        
        abs_err_gray = (err0+err1+err2)/3
        
        ax = plt.subplot(gs[i+2*batch_size]) #ax = plt.subplot(fig_rows, batch_size, i + 1 + 2*batch_size)
        plt.imshow(abs_err_gray, cmap='gray')
        #ax = remove_subplot_spines(ax)
        if i==0: 
            ax.set_ylabel('Absolute error', rotation=rotation, fontsize=font_size)
        #else:   
        plt.yticks([],[]) 
        plt.xticks([],[])
        

        """
        #### THRESHOLDING + OPENING START
        binary = abs_err_gray > threshold
        
        ax = plt.subplot(gs[i+3*batch_size]) #ax = plt.subplot(fig_rows, batch_size, i + 1 + 3*batch_size)
        plt.imshow(binary, cmap='gray') 
        if i==0:
            ax.set_ylabel('binary diff', rotation=rotation, fontsize=font_size)
        
        opened = morphology.opening(binary, morphology.star(a=opening_size)) 
        ax = plt.subplot(gs[i+4*batch_size]) #ax = plt.subplot(fig_rows, batch_size, i + 1 + 4*batch_size)
        plt.imshow(opened, cmap='gray')
        if i==0: 
            ax.set_ylabel('Opened', rotation=rotation, fontsize=font_size)
        else:   
            plt.yticks([],[]) 
        plt.xticks([],[])
        
        #### THRESHOLDING + OPENING END
        
        """
        
        #### GAUSSIAN+ THRESHOLDING + OPENING START
        abs_err_gauss = filters.gaussian(abs_err_gray, gauss_sigma)

        #k = 5
        #kernel_median = np.ones((k,k))
        #abs_err_median = filters.median(abs_err_gray, selem=kernel_median) #,selem=3) default selem is square 3x3

        binary_gauss = abs_err_gauss > gauss_threshold

        #ax = plt.subplot(gs[i+3*batch_size])
        #plt.imshow(binary_gauss, cmap='gray')
        #if i==0: 
        #    ax.set_ylabel('binary_gauss', rotation=rotation, fontsize=font_size)
        #else:   
        #    plt.yticks([],[]) 
        #plt.xticks([],[])
        
        
        opened_gauss = morphology.opening(binary_gauss, morphology.star(a=gauss_opening_size)) 
        
        #ax = plt.subplot(gs[i+4*batch_size]) #ax = plt.subplot(fig_rows, batch_size, i + 1 + 4*batch_size)
        #plt.imshow(opened_gauss, cmap='gray')
        
        #### GAUSSIAN + THRESHOLDING + OPENING END
        
        masked = add_mask(original_imgs[i],opened_gauss)
        ax = plt.subplot(gs[i+3*batch_size])
        plt.imshow(masked)
        #ax = remove_subplot_spines(ax)
        if i==0: 
            ax.set_ylabel('Object map', rotation=rotation, fontsize=font_size) #\non original'
        #else:
        plt.yticks([],[]) 
        plt.xticks([],[])

        
    #plt.subplots_adjust(wspace=0,hspace=0)
    plt.close(plot)
    return plot

def create_simple_reconstruction_plot(original_imgs, reconstructed_imgs, batch_size):
    fig_rows = 3

    gs = gridspec.GridSpec(fig_rows, batch_size)
    gs.update(wspace=0.02, hspace=0.02)
    
    scale=300
    plot = plt.figure(figsize=(1280*batch_size//scale,720*fig_rows//scale))#figsize=(30, 20)) #(figsize=(width,height))
    for i in range(batch_size):

        # display original
        ax = plt.subplot(gs[i])#fig_rows, batch_size, i + 1)
        plt.imshow(original_imgs[i])
        ax.axis('off')  
        
        # display reconstruction
        ax = plt.subplot(gs[i+batch_size]) #fig_rows, batch_size, i + 1 + batch_size)
        plt.imshow(reconstructed_imgs[i])
        ax.axis('off')

        # reconstruction error in grayscale
        err0 = np.abs(reconstructed_imgs[i,:,:,0] - original_imgs[i,:,:,0])
        err1 = np.abs(reconstructed_imgs[i,:,:,1] - original_imgs[i,:,:,1])
        err2 = np.abs(reconstructed_imgs[i,:,:,2] - original_imgs[i,:,:,2])
        
        abs_err_gray = (err0+err1+err2)/3
        
        ax = plt.subplot(gs[i+2*batch_size]) #fig_rows, batch_size, i + 1 + 2*batch_size)
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
    numb_str = str(numb+1)
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