import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plots import PLOT_RESULTS_1D , PLOT_TIME_SOURCE , PLOT_VIDEO_1D , PLOT_BOUNDARY_1D , PLOT_RESULTS , PLOT_VIDEO
from parameters import REFRACTIVE_INDEX , ALPHA
from fields import SOURCE
from weights import WEIGHT_CREATION
from layers import PROPAGATE , PROPAGATE_1D , MASK

import pickle

data_type = tf.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well


def FORWARD_1D(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type):

    # ----------------- Simulation Parameters ---------------------- #
    alpha = ALPHA(n_x,n_y,n_z)
    mask_start = mask[0,:]
    mask_end = mask[1,:]
    n = REFRACTIVE_INDEX(n_x,n_y,n_z,scatter_type,mask_start,mask_end,initial_weight)

    # ----------------- Source ------------------------------------- #
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
        
        out_field_tens,boundary = PROPAGATE_1D(in_field_tens,alpha,n_c,weights_tens,n_t,n_w)

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

def FORWARD(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type):

    # ----------------- Simulation Parameters ---------------------- #
    alpha = ALPHA(n_x,n_y,n_z)
    mask_start = mask[0,:]
    mask_end = mask[1,:]
    n = REFRACTIVE_INDEX(n_x,n_y,n_z,scatter_type,mask_start,mask_end,initial_weight)

    # ----------------- Source ------------------------------------- #
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
        
        out_field_tens = PROPAGATE(in_field_tens,alpha,n_c,weights_tens,n_t,n_w)

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
        PLOT_VIDEO(out_field,n_c,n,alpha,fig_num = 2)

        PLOT_RESULTS(out_field[:,:,:,:,n_t-1],n,alpha,fig_nums = [3,4,5,6])

    plt.show()
    
def INVERSE_1D(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type):

    #------------------------ Read in Data --------------------------#
    with tf.name_scope('read_data'):
        file_address_fields = "C:/Users/travi/Documents/Northwestern/STSN/field_data/"
        file_address_mesh = "C:/Users/travi/Documents/Northwestern/STSN/mesh_data/"
        
        in_field , out_field , layers , mask_start , mask_end , n_x , n_y , n_z , mesh , file_id , ref_index = GET_DATA(file_address_fields , file_address_mesh)

    #------------------------ Create Weights ------------------------#
    with tf.name_scope('create_weights'):
        weights_tens , weights_train_tens = WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z , n_w, initial_weight)

    #--------------------------- Placeholder Instantiation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        in_field_tens = tf.placeholder(dtype = data_type, shape = [n_x,n_y,n_z,n_c,layers])
        out_field_tens = tf.placeholder(dtype = data_type, shape = [n_x,n_y,n_z,n_c])

    #--------------------------- Cost Function Definition --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Cost Function (Least Squares) ... ... ...")

    with tf.name_scope('cost_function'):
        
        pre_out_field_tens = PROPAGATE(in_field_tens,mesh,n_c,weights_tens,layers,n_w) # prediction function

        mask_pre_out_field_tens = MASK(mask_start,mask_end,pre_out_field_tens[:,:,:,:,layers-1],n_x,n_y,n_z,n_c,np.float32)

        mask_out_field_tens = MASK(mask_start,mask_end,out_field_tens,n_x,n_y,n_z,n_c,np.float32)
        
        least_squares = tf.norm(mask_pre_out_field_tens-mask_out_field_tens, ord=2,name='least_squre')**2 	#

    print("Done!\n")

    #--------------------------- Define Optimizer --------------------------#
    print("Building Optimizer ... ... ...")
    lr = 0.01
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(least_squares, var_list = [weights_train_tens])
    with tf.name_scope('clip'):
        clip_op = tf.assign(weights_train_tens, tf.clip_by_value(weights_train_tens, 0.25, 1.0))
    print("Done!\n")

    #--------------------------- Merge Summaries ---------------------------#
    merged = tf.summary.merge_all()

    #--------------------------- Training --------------------------#
    epochs = 6000
    loss_tolerance = 1e-10
    table = []

    # saves objects for every iteration
    fileFolder = "results/" + file_id

    # if the results folder does not exist for the current model, create it
    if not os.path.exists(fileFolder):
            os.makedirs(fileFolder)


    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run( tf.global_variables_initializer() )

        print("Tensor in field:")		# show info. for in field
        print(in_field)
        print("")

        print("Tensor out field: ")		# show info. for out field
        print(out_field)
        print("")

        print("--------- Starting Training ---------\n")
        for i in range(1, epochs+1):

            # run X and Y dynamically into the network per iteration
            _,loss_value = sess.run([train_op, least_squares], feed_dict = {in_field_tens: in_field, out_field_tens: out_field})

            # perform clipping 
            with tf.name_scope('clip'):
                sess.run(clip_op)

            print('Epoch: ',i)
            print('Loss: ',loss_value)

            w = sess.run(weights_train_tens)

            with open(fileFolder + '/' + str(i) + '_loss.pkl', 'wb') as f:

                # Pickle loss value
                pickle.dump(loss_value, f, pickle.HIGHEST_PROTOCOL)

            with open(fileFolder + '/' + str(i) + '_trained_weights.pkl', 'wb') as f:

                # Pickle loss value
                pickle.dump(w, f, pickle.HIGHEST_PROTOCOL)

            # break from training if loss tolerance is reached
            if loss_value <= loss_tolerance:
                endCondition = '_belowLossTolerance_epoch' + str(i)
                print(endCondition)
                break

        plt.imshow(np.sqrt(1/w[:,:,0]))
        plt.colorbar()
        plt.show()
