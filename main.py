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
    
def MULTIPLE_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par):
    #performs a forward multiple Lorentz resonace simulation
    # n_x: number of spatial parameters in the x or 0th axis direction - tf.constant (int), shape(1,)
    # n_y: number of spatial parameters in the y or 1st axis direction - tf.constant (int), shape(1,)
    # n_z: number of spatial parameters in the z or 2nd axis direction - tf.constant (int), shape(1,)
    # n_t: number of time steps the simulation takes - tf.constant (int), shape(1,)
    # del_l: the lenght of the mesh step in all three directions (m) - tf.constant (int), shape(1,)
    # n_r: number of resonances - tf.constant (int), shape(1,)
    # source_par: contains the parameters relevent for making the source - list, shape(11,)
    # mat_par: contains the parameters relevent for the material - list, shape(5,)

    #determine time step based on the criteria del_l/del_t = 2*c0
    del_t = del_l/(2*c0)

    # ----------------- Simulation Parameters ---------------------- #
    inf_x,w_0,damp,del_x = MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,mat_par[0],mat_par[1],mat_par[2],mat_par[3],mat_par[4])
    PLOT_DISPERSION_PARAMETERS_1D(inf_x[0,:,0],w_0[0,:,0,:],damp[0,:,0,:],del_x[0,:,0,:],fig_num = [1,2])

    # ----------------- Source ------------------------------------- #
    v_f,time_source,e_time_source = SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,source_par[0],source_par[1],source_par[2],source_par[3],source_par[4],source_par[5],source_par[6],source_par[7],source_par[8],source_par[9],source_par[10])
    PLOT_TIME_SOURCE(v_f,time_source,e_time_source,del_l,fig_num=[4,5,6])

    #--------------------------- Tensor Creation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        v_f = tf.convert_to_tensor(v_f,dtype = data_type)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    with tf.name_scope('graph'):
        
        v_i , f_time, f_final = MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,del_t,n_c,n_t,n_f)

    print("Done!\n")

    #--------------------------- Merge Summaries ---------------------------#
    merged = tf.summary.merge_all()

    #--------------------------- Run Forward Model --------------------------#
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run( tf.global_variables_initializer() )

        print("--------- Starting Run of Forward Model ---------\n")

        # run the graph to determine the output field
        f_time = sess.run(f_time)
        f_final = sess.run(f_final)

        # plot results
        PLOT_VIDEO_1D(f_time[:,:,:,source_par[0],:],del_l,fig_num = 7)

        PLOT_RESULTS_2(f_final[:,:,:,2],del_l,fig_num = 8)

    plt.show()
