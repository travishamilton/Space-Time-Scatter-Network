import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plots import*
from parameters import REFRACTIVE_INDEX , REFRACTIVE_INDEX_DISPERSION , MULTIPLE_DISPERSION_PARAMETERS
from fields import SOURCE , POINT_SOURCE , TIME_SOURCE_LUMERICAL
from weights import WEIGHT_CREATION
from layers import PROPAGATE , MULTIPLE_PROPAGATE
import pickle

### ----------------General Notes ------------------------------- ###
# This work is based off of the paper "Generalized Material Models 
# in TLM—Part I: Materials with Frequency-Dependent Properties" (1999)
# by John  Paul,  Christos  Christopoulos,  and  David  W.  P.  Thomas.
# Following the paper's notation, field values correspond to the 
# electric and magnetic field components (x,y,z) present at a node, 
# while voltage and current values correspond to voltage and current 
# values present on the transmission line network.

data_type = tf.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well

### ---------------Global Constants ---------------------------- ###
n_f = 6                       #number of field components at node
n_c = 12                      #number of voltage components at node
c0 = 2.99792458e8             #speed of light in vaccum (m/s)

def FORWARD(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type):

    del_l = 0.0075/wavelength
    del_t = del_l/(2*c0)
    wavelength = del_l*wavelength
    fwhm = fwhm*del_l

    # ----------------- Simulation Parameters ---------------------- #
    mask_start = mask[0,:]
    mask_end = mask[1,:]
    n,inf_x,w_0,damp,del_x = REFRACTIVE_INDEX_DISPERSION(n_x,n_y,n_z,n_f,scatter_type,mask_start,mask_end,initial_weight)

    # ----------------- Source ------------------------------------- #
    v_f,time_source,e_time_source = SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis)

    PLOT_TIME_SOURCE(v_f,time_source,e_time_source,del_l,fig_num=[1,2,3])

    #------------------------ Create Weights ------------------------#
    # with tf.name_scope('create_weights'):
    #     weights_tens , weights_train_tens = WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z , n_w, initial_weight)

    #--------------------------- Tensor Creation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        v_f = tf.convert_to_tensor(v_f*1.0e16,dtype = data_type)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    with tf.name_scope('graph'):
        
        v_i , f_time, f_final = PROPAGATE(v_f,inf_x,w_0,damp,del_x,del_t,n_c,n_t)

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
        PLOT_VIDEO_1D(f_time[:,:,:,polarization,:],del_l,fig_num = 4)

        PLOT_RESULTS_2(f_final[:,:,:,2],del_l,fig_num = 7)

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

def COMPARE(n_c,n_x,n_y,n_z,n_t,del_l,del_t,x_inf,w_0,damp,del_x,location,time_source,polarization,f_lumerical_time):
    #compares the Lumerical results with the forward model STSN results

    # ----------------- Source ------------------------------------- #
    v_f_time = TIME_SOURCE_LUMERICAL(polarization,n_f,del_t,n_t,del_l,time_source)
    v_f = POINT_SOURCE(location,v_f_time,n_x,n_y,n_z)
    e_time_source = time_source

    PLOT_TIME_SOURCE_LUMERICAL(v_f,fig_num=1)

    #--------------------------- Tensor Creation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        v_f = tf.convert_to_tensor(v_f*1.0e16,dtype = data_type)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    with tf.name_scope('graph'):
        
        v_i , f_time, f_final = PROPAGATE(v_f,x_inf,w_0,damp,del_x,del_t,n_c,n_t)

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
        PLOT_COMPARISON_VIDEO_1D(f_time[:,:,:,polarization,:],f_lumerical_time,del_l,fig_num = 4)

        PLOT_RESULTS_2(f_final[:,:,:,2],del_l,fig_num = 7)

    plt.show()
