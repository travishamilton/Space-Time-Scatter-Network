import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plots import PLOT_RESULTS_1D , PLOT_TIME_SOURCE , PLOT_VIDEO_1D , PLOT_BOUNDARY_1D
from parameters import REFRACTIVE_INDEX , ALPHA
from files import FILE_ID
from fields import SOURCE
from weights import WEIGHT_CREATION
from layers import PROPAGATE

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
initial_weight = 1/1.5**2 # initial weight value in masked region

# ------------ Simulation Parameters ----------------------- #
n_x = 1
n_y = 800
n_z = 1 

n_c = 12
n_t = 400

time_changes = 0
scatter_type = 'mask'
#index of the start of the mask and end of the mask
mask = np.array([[0,39,0],[0,61,0]])
mask_start = mask[0,:]
mask_end = mask[1,:]

alpha = ALPHA(n_x,n_y,n_z)
n = REFRACTIVE_INDEX(n_x,n_y,n_z,scatter_type,mask_start,mask_end,initial_weight)

file_id = FILE_ID(n_t,n_x,n_y,n_z,scatter_type,mask,time_changes)

# ------------- Source Parameters --------------------------- #
location = (0,25,0)
polarization = 2
wavelength = 80
injection_axis = 1
injection_direction = 0
fwhm = 3*wavelength
fwhm_mode = 15
n_m = 0
center_mode = 0
mode_axis = 0
source_type = 'Line'

in_field,time_source = SOURCE(polarization,n_c,n_t,wavelength,fwhm,n,alpha,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m ,center_mode,mode_axis)

PLOT_TIME_SOURCE(time_source,n[location[0],location[1],location[2],0],alpha[location[0],location[1],location[2],:],fig_num=1)

#------------------------ Create Weights ------------------------#
with tf.name_scope('create_weights'):
    weights_tens , weights_train_tens = WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z , n_w, initial_weight)

#--------------------------- Tensor Creation --------------------------#
with tf.name_scope('instantiate_placeholders'):
    in_field_tens = tf.convert_to_tensor(in_field,dtype = data_type)

#--------------------------- Graph Construction --------------------------#
# compute least squares cost for each sample and then average out their costs
print("Building Graph ... ... ...")

with tf.name_scope('graph'):
    
    out_field_tens,boundary = PROPAGATE(in_field_tens,alpha,n_c,weights_tens,n_t,n_w)

print("Done!\n")

#--------------------------- Merge Summaries ---------------------------#
merged = tf.summary.merge_all()

#--------------------------- Run Forward Model --------------------------#
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run( tf.global_variables_initializer() )

    print("--------- Starting Run of Forward Model ---------\n")

    # run the graph to determine the output field
    out_field = sess.run(out_field_tens)

    # plot results
    PLOT_VIDEO_1D(out_field,n_c,n,alpha,fig_num = 2)

    PLOT_RESULTS_1D(out_field[:,:,:,:,n_t-1],n,alpha,fig_nums = [3,4,5,6])

    PLOT_BOUNDARY_1D(boundary,fig_num=7)

plt.show()
