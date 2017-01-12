#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains perturbations generation algorithm.

Author: Alexandre Péré

"""

import numpy

def compute_fs_grad_num(network, X, y):
    """
    This function allows to compute the numerical gradient of the output Loss with respect to input pixels.
    
    WARNING: This function is extremely slow and is for test/debug use. Checkout the Symbolic version for production use.
    
    Parameters:
        + network: the network considered
        + X: the input with shape [nb_samples, pixels]
        + y: the label with shape [nb_samples, nb_classes]
       
    Returns:
        + Gradient
    """
    # We set useful variables
    delta = 0.001
    nb_samples = X.shape[0]
    nb_pixels = X.shape[1]
    output = numpy.zeros([nb_samples, nb_pixels])
    
    # We loop through samples
    for sampleiter in range(0,nb_samples):
        print('Processing sample n°: %i'%sampleiter)
        # We loop through pixels
        for pixiter in range(0,nb_pixels):
            X_p = X[sampleiter:sampleiter+1].copy()
            X_m = X[sampleiter:sampleiter+1].copy()
            X_p[0,pixiter] += delta
            X_m[0,pixiter] += -delta
            in_dict = {network._net_input:X_p, network._net_label:y[sampleiter:sampleiter+1]}
            L_p = network.evaluate_tensor('Loss/Loss:0', initial_dict='test', update_dict=in_dict)
            in_dict = {network._net_input:X_m, network._net_label:y[sampleiter:sampleiter+1]}
            L_m = network.evaluate_tensor('Loss/Loss:0', initial_dict='test', update_dict=in_dict)
            grad = (L_p-L_m)/(2*delta)
            output[sampleiter,pixiter] = grad
    
    return output

def compute_fs_grad_sym(network, X, y):
    """
    This function allows to compute the symbolic gradient of the output Loss with respect to input pixels.
    
    Parameters:
        + network: the network considered
        + X: the input with shape [nb_samples, pixels]
        + y: the label with shape [nb_samples, nb_classes]
       
    Returns:
        + Gradient
    """
    # We set useful variables
    delta = 0.001
    nb_samples = X.shape[0]
    nb_pixels = X.shape[1]
    output = numpy.zeros([nb_samples, nb_pixels])
    
    # We loop through samples
    for sampleiter in range(0,nb_samples):
        in_dict = {network._net_input:X[sampleiter:sampleiter+1], network._net_label:y[sampleiter:sampleiter+1]}
        out_arr = network.evaluate_tensor('Pert/UnitImGrad/Input/Reshape_grad/Reshape:0', initial_dict='test', update_dict=in_dict)
        output[sampleiter] = out_arr
 
    return output


def fast_sign_perturbation(network, X, y, eps):
    """
    This function implements Fast Sign Perturbation as proposed in https://arxiv.org/pdf/1412.6572v3.pdf .
    
    Parameters:
        + network: the network considered
        + X: the input with shape [1, width, height, depth]
        + y: the label with shape [1, nb_classes]
        + eps: the unitary perturbation magnitude
        
    Returns:
        + Fast Sign Perturbed Sample.
    """
    
    gradient = compute_loss_gradient_wrt_input(network, X, y)
    gradient_sign = numpy.sign(gradient)
    output = X + eps*gradient_sign
    
    return output
    
        
    
    
    