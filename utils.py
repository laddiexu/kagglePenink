import numpy as np
import tensorflow as tf
from keras import backend as K


def logistic(z):
    res = 1 / (1 + K.exp(-65535 * z))
    return res

def creatMask(tx,ty,tl,input_shape):
    x_axis_mask = np.zeros((3,input_shape,input_shape))
    y_axis_mask = np.zeros((3,input_shape,input_shape))
    attention_mask = np.zeros((3,input_shape,input_shape))
    
    x_axis_mask[:,:,range(input_shape)] = range(input_shape)
    y_axis_mask[0] = x_axis_mask[0].T
    y_axis_mask[1] = y_axis_mask[1].T
    y_axis_mask[2] = y_axis_mask[2].T
    
    x_axis_mask = K.variable(value = x_axis_mask,dtype ='float32',name = 'x_axis_mask')
    y_axis_mask = K.variable(value = y_axis_mask,dtype ='float32',name = 'y_axis_mask')
    t_x_tl = tx - tl 
    t_x_br = tx + tl
    t_y_tl = ty - tl
    t_y_br = ty + tl

    x_axis_res = logistic(x_axis_mask - t_x_tl) - logistic(x_axis_mask - t_x_br)
    y_axis_res = logistic(y_axis_mask - t_y_tl) - logistic(y_axis_mask - t_y_br)
    attention_mask = tf.multiply(x_axis_res,y_axis_res)
    attention_mask = K.variable(value = attention_mask,dtype ='float32',name = 'attention_mask')
    return attention_mask

def upSampleing(x_att,lambd):
    input_shape = x_amp.shape()
    x_axis_mask = np.arange(0,input_shape[1])
    y_axis_mask = np.arange(0,input_shape[2])
    
    x0_axis_mask  = np.floor(x_axis_mask / lambd).astype(int)
    x1_axis_mask = x0_axis_mask + 1
    y0_axis_mask = np.floor(y_axis_mask / lambd).astype(int)
    y1_axis_mask = y0_axis_mask + 1
    
    Ia = x_att[x0_axis_mask,y0_axis_mask]
    Ib = x_att[x0_axis_mask,y1_axis_mask]
    Ic = x_att[x1_axis_mask,y0_axis_mask]
    Id = x_att[x1_axis_mask,y1_axis_mask]

    wa = np.abs(1 - )

    