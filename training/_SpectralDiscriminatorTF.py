# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:18:48 2020

@author: Steff
"""

import tensorflow as tf
import numpy as np

###############################################################################
class SpectralLoss:
    
    ############################################################
    def __init__(self, rows=64, cols=64, eps=1E-8):
        self.eps = eps
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(rows / 2)
        # number of cols after onesided fft
        cols_onesided = int(cols / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices((rows,cols_onesided)) - np.array([[[shift_rows]],[[0]]])
        r = np.sqrt(r[0,:,:]**2+r[1,:,:]**2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r, axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = np.tile(
            r,
            reps = (r_max+1, 1, 1)
        )
        radius_to_slice = np.arange(r_max+1).reshape((-1,1,1))
        # generate mask for each radius
        mask = np.where(
            r == radius_to_slice,
            1,
            0
        ).astype(np.float)
        mask = np.expand_dims(mask, axis=0) # add batch dimension
        
        # how man entries for each radius?
        mask_n = 1 / np.sum(mask, axis=(2,3))
        
        self.mask = tf.constant(mask, dtype=tf.float32)
        self.mask_n = tf.constant(mask_n, dtype=tf.float32)
        
        self.r_max = r_max
        self.vector_length = r_max+1
        
    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################
            
    ############################################################
    def fft(self, data):
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data =  0.299 * data[:,0,:,:] + \
                    0.587 * data[:,1,:,:] + \
                    0.114 * data[:,2,:,:]
        fft = tf.signal.rfft2d(data)
        fft_abs = tf.abs(fft)
        fft_abs = fft_abs + self.eps
        fft_abs = 20 * tf.math.log(fft_abs)
        
        return fft_abs
    
    ############################################################
    def spectral_vector(self, data):
        """Assumes first dimension to be batch size."""
        fft = self.fft(data)
        fft = tf.expand_dims(
            fft,
            axis = 1
        )
        fft = tf.tile(
            fft,
            multiples = (1, self.vector_length, 1, 1)
        )
        # apply mask and compute profile vector
        profile = tf.reduce_sum(fft * self.mask, axis=(2,3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - tf.reduce_min(profile, axis=1, keepdims=True)
        profile = profile / tf.reduce_max(profile, axis=1, keepdims=True)
        
        return profile