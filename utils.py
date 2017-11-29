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
    y_axis_mask[1] = x_axis_mask[1].T
    y_axis_mask[2] = x_axis_mask[2].T
    
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


def upSampleing(x_att,lambd,tx,ty,tl):
    
    input_shape = attention_mask.shape
    attention_mask = K.eval(attention_mask)

    x_axis_mask = np.zeros((input_shape))
    y_axis_mask = np.zeros((input_shape))

    x_axis_mask[:,:,range(int(input_shape[1]))] = range(int(input_shape[1]))
    y_axis_mask[0] = x_axis_mask[0].T
    y_axis_mask[1] = x_axis_mask[1].T
    y_axis_mask[2] = x_axis_mask[2].T

    x0_axis_mask = (x_axis_mask // lambd).astype(int) + tx - tl + 1
    x1_axis_mask = x0_axis_mask + 1
    y0_axis_mask = (y_axis_mask // lambd).astype(int) + ty - tl + 1
    y1_axis_mask = y0_axis_mask + 1

    I1 = np.zeros((input_shape))
    I2 = np.zeros((input_shape))
    I3 = np.zeros((input_shape))
    I4 = np.zeros((input_shape))

    for i in range(3):
        for j in range(int(input_shape[1])):
            for k in range(int(input_shape[2])):
                att_coor_x0_y0 = (i,x0_axis_mask[i,j,k],y0_axis_mask[i,j,k])
                att_coor_x0_y1 = (i,x0_axis_mask[i,j,k],y1_axis_mask[i,j,k])
                att_coor_x1_y0 = (i,x1_axis_mask[i,j,k],y0_axis_mask[i,j,k])
                att_coor_x1_y1 = (i,x1_axis_mask[i,j,k],y1_axis_mask[i,j,k])
                I1[i,j,k] = attention_mask[att_coor_x0_y0]
                I2[i,j,k] = attention_mask[att_coor_x0_y1]
                I3[i,j,k] = attention_mask[att_coor_x1_y0]
                I4[i,j,k] = attention_mask[att_coor_x1_y1]

    W_x0_y0 = np.abs(1 - frac(x_axis_mask / lambd)) * np.abs(1 - frac(y_axis_mask / lambd))
    W_x0_y1 = np.abs(1 - frac(x_axis_mask / lambd)) * np.abs(1 - 1 - frac(y_axis_mask / lambd))
    W_x1_y0 = np.abs(1 - 1 - frac(x_axis_mask / lambd)) * np.abs(1 - frac(y_axis_mask / lambd))
    W_x1_y1 = np.abs(1 - 1 - frac(x_axis_mask / lambd)) * np.abs(1 - 1 - frac(y_axis_mask / lambd))

    return x_amp = W_x0_y0*I1 + W_x0_y1*I2 + W_x1_y0 * I3 + W_x1_y1*I4
    