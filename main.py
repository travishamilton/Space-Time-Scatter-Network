import numpy as np
import tensorflow as tf
import os

from files import *
from weights import *

import pickle

data_type = tf.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well

# ----------------------- Simulation Constants ------------------#
n_c = 12    # number of field components per node
n_w = 1     # number of weights per node

#------------------------ Read in Data --------------------------#
with tf.name_scope('read_data'):
    file_address = "C:/Users/travi/Documents/Northwestern/STSN/forward_model/field_data/"
    in_field , out_field , layers , mask_start , mask_end , n_x , n_y , n_z = GET_FIELD_DATA(file_address)

    sample_n, feat_n = in_field.shape	# sampN: number of training samples, featN: features per sample

#------------------------ Create Weights ------------------------#
with tf.name_scope('create_weights'):
    weight_tens = WEIGHT_CREATION(mask_start, mask_end, layers, data_type, n_x, n_y, n_z , n_w)

#--------------------------- Placeholder Instantiation --------------------------#
with tf.name_scope('instantiate_placeholders'):
    in_field_tens = tf.placeholder(dtype = data_type, shape = [n_x,n_y,n_z])
    out_field_tens = tf.placeholder(dtype = data_type, shape = [n_x,n_y,n_z])

#--------------------------- Cost Function Definition --------------------------#
# compute least squares cost for each sample and then average out their costs
print("Building Cost Function (Least Squares) ... ... ...")

with tf.name_scope('cost_function'):
	pre_out_field_tens = transmit(in_field_tens, weight_tens, layers) # prediction function

	