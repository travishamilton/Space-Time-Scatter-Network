import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plots import*
from parameters import*
from fields import SOURCE , TIME_SOURCE_LUMERICAL , LINE_SOURCE_E , MULTIPLE_SOURCE
from layers import *

### ----------------General Notes ------------------------------- ###
# This work is based off of the paper "Generalized Material Models 
# in TLM—Part I: Materials with Frequency-Dependent Properties" (1999)
# by John  Paul,  Christos  Christopoulos,  and  David  W.  P.  Thomas.
# Following the paper's notation, field values correspond to the 
# electric and magnetic field components (x,y,z) present at a node, 
# while voltage and current values correspond to voltage and current 
# values present on the transmission line network.

data_type = np.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

### ---------------Global Constants ---------------------------- ###
n_f = 6                       #number of field components at node
n_c = 12                      #number of voltage components at node
c0 = 2.99792458e8             #speed of light in vaccum (m/s)

def LARGE_FIELD_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par):
    #performs a forward non-linear multiple Lorentz resonace simulation
    # n_x: number of spatial parameters in the x or 0th axis direction - tf.constant (int), shape(1,)
    # n_y: number of spatial parameters in the y or 1st axis direction - tf.constant (int), shape(1,)
    # n_z: number of spatial parameters in the z or 2nd axis direction - tf.constant (int), shape(1,)
    # n_t: number of time steps the simulation takes - tf.constant (int), shape(1,)
    # del_l: the lenght of the mesh step in all three directions (m) - tf.constant (int), shape(1,)
    # n_r: number of resonances - tf.constant (int), shape(1,)
    # source_par: contains the parameters relevent for making the source - list, shape(11,)
    # mat_par: contains the parameters relevent for the material - list, shape(9,)

    #determine time step based on the criteria del_l/del_t = 2*c0
    del_t = del_l/(2*c0)

    # ----------------- Simulation Parameters ---------------------- #
    inf_x,w_0,damp,del_x,x_nl = NL_MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,mat_par[0],mat_par[1],mat_par[2],mat_par[3],mat_par[4],mat_par[5],mat_par[6],mat_par[7],mat_par[8])
    PLOT_DISPERSION_PARAMETERS_1D(inf_x[0,:,0],w_0[0,:,0,:],damp[0,:,0,:],del_x[0,:,0,:],fig_num = [1,2])

    # ----------------- Source ------------------------------------- #
    v_f,time_source,current_density = SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,source_par[0],source_par[1],source_par[2],source_par[3],source_par[4],source_par[5],source_par[6],source_par[7],source_par[8],source_par[9],source_par[10])
    PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num=[3,4,5],location = source_par[3],del_t = del_t)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    v_f = v_f*10**7

    v_i , f_time, f_final = NL_MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f)

    print("Done!\n")

    #------------------ Spectrum -------------------------------------------#
    PLOT_RESULTS_2(f_final[0,:,0,2],del_l,fig_num = 100)

    PLOT_VIDEO_1D(f_time[:,:,:,source_par[0],:],del_l,fig_num = 28)

    results = [f_time,del_t,n_t]

    with open("forward_data.pkl","wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    plt.show()

def FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par):
    #performs a forward non-linear multiple Lorentz resonace simulation
    # n_x: number of spatial parameters in the x or 0th axis direction - tf.constant (int), shape(1,)
    # n_y: number of spatial parameters in the y or 1st axis direction - tf.constant (int), shape(1,)
    # n_z: number of spatial parameters in the z or 2nd axis direction - tf.constant (int), shape(1,)
    # n_t: number of time steps the simulation takes - tf.constant (int), shape(1,)
    # del_l: the lenght of the mesh step in all three directions (m) - tf.constant (int), shape(1,)
    # n_r: number of resonances - tf.constant (int), shape(1,)
    # source_par: contains the parameters relevent for making the source - list, shape(11,)
    # mat_par: contains the parameters relevent for the material - list, shape(9,)

    #determine time step based on the criteria del_l/del_t = 2*c0
    del_t = del_l/(2*c0)

    # ----------------- Simulation Parameters ---------------------- #
    inf_x,w_0,damp,del_x,x_nl = NL_MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,mat_par[0],mat_par[1],mat_par[2],mat_par[3],mat_par[4],mat_par[5],mat_par[6],mat_par[7],mat_par[8])
    PLOT_DISPERSION_PARAMETERS_1D(inf_x[0,:,0],w_0[0,:,0,:],damp[0,:,0,:],del_x[0,:,0,:],fig_num = [1,2])

    # ----------------- Source ------------------------------------- #
    v_f,time_source,current_density = SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,source_par[0],source_par[1],source_par[2],source_par[3],source_par[4],source_par[5],source_par[6],source_par[7],source_par[8],source_par[9],source_par[10])
    PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num=[3,4,5],location = source_par[3],del_t = del_t)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    v_i , f_time, f_final = NL_MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f)

    print("Done!\n")

    #------------------ Spectrum -------------------------------------------#
    PLOT_VIDEO_1D(f_time[:,:,:,source_par[0],:],del_l,fig_num = 28)

    results = [f_time,del_t,n_t]

    with open("forward_data.pkl","wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    plt.show()

def FORWARD_2D_LINEAR_NONDISPERSIVE(n_x,n_y,n_z,n_t,del_l,source_par,mat_par):
    #performs a forward non-linear multiple Lorentz resonace simulation
    # n_x: number of spatial parameters in the x or 0th axis direction - tf.constant (int), shape(1,)
    # n_y: number of spatial parameters in the y or 1st axis direction - tf.constant (int), shape(1,)
    # n_z: number of spatial parameters in the z or 2nd axis direction - tf.constant (int), shape(1,)
    # n_t: number of time steps the simulation takes - tf.constant (int), shape(1,)
    # del_l: the lenght of the mesh step in all three directions (m) - tf.constant (int), shape(1,)
    # n_r: number of resonances - tf.constant (int), shape(1,)
    # source_par: contains the parameters relevent for making the source - list, shape(11,)
    # mat_par: contains the parameters relevent for the material - list, shape(9,)

    #determine time step based on the criteria del_l/del_t = 2*c0
    del_t = del_l/(2*c0)

    # ----------------- Simulation Parameters ---------------------- #
    inf_x = LINEAR_NONDISPERSIVE_PARAMETERS(n_x,n_y,n_z,mat_par[0],mat_par[1],mat_par[2])
    PLOT_LINEAR_NONDISPERSIVE_PARAMETERS_2D_Z(inf_x,del_l,fig_num = 1)

    # ----------------- Source ------------------------------------- #
    v_f,time_source,current_density = SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,source_par[0],source_par[1],source_par[2],source_par[3],source_par[4],source_par[5],source_par[6],source_par[7],source_par[8],source_par[9],source_par[10])
    PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num=[3,4,5],location = source_par[3],del_t = del_t)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    v_i , f_time, f_final = LINEAR_NONDISPERSIVE_PROPAGATE(v_f,inf_x,del_t,n_c,n_t,n_f)

    PLOT_VIDEO_2D_Z(f_time,del_l,del_t,6)

    print("Done!\n")

    plt.show()

def TIME_DEP_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par):
    #performs a forward non-linear multiple Lorentz resonace simulation
    # n_x: number of spatial parameters in the x or 0th axis direction - tf.constant (int), shape(1,)
    # n_y: number of spatial parameters in the y or 1st axis direction - tf.constant (int), shape(1,)
    # n_z: number of spatial parameters in the z or 2nd axis direction - tf.constant (int), shape(1,)
    # n_t: number of time steps the simulation takes - tf.constant (int), shape(1,)
    # del_l: the lenght of the mesh step in all three directions (m) - tf.constant (int), shape(1,)
    # n_r: number of resonances - tf.constant (int), shape(1,)
    # source_par: contains the parameters relevent for making the source - list, shape(11,)
    # mat_par: contains the parameters relevent for the material - list, shape(9,)

    #determine time step based on the criteria del_l/del_t = 2*c0
    del_t = del_l/(2*c0)

    # ----------------- Simulation Parameters ---------------------- #
    inf_x,w_0,damp,del_x,x_nl,n_r = TIME_DEP_PARAMETERS(n_x,n_y,n_z,n_t,mat_par[0],mat_par[1],mat_par[2],mat_par[3],mat_par[4],mat_par[5],mat_par[6],mat_par[7],mat_par[8])
    #PLOT_DISPERSION_PARAMETERS_1D(inf_x[0,:,0,0],w_0[0,:,0,:,0],damp[0,:,0,:,0],del_x[0,:,0,:,0],fig_num = [1,2])
    last_fig = PLOT_TIME_DEP_DISPERSION_PARAMETERS_1D(inf_x[0,:,0,:],w_0[0,:,0,:,:],damp[0,:,0,:,:],del_x[0,:,0,:,:],fig_num = 1,del_l = del_l,del_t = del_t)

    # ----------------- Source ------------------------------------- #
    v_f,time_source,current_density = SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,source_par[0],source_par[1],source_par[2],source_par[3],source_par[4],source_par[5],source_par[6],source_par[7],source_par[8],source_par[9],source_par[10])
    PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num=[last_fig+1,last_fig+2,last_fig+3],location = source_par[3],del_t = del_t)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    v_i , f_time, f_final = TIME_DEP_NL_MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f)

    print("Done!\n")

    #------------------ Plot -------------------------------------------#
    PLOT_TIME_SPACE(f_time[0,:,0,:,:],last_fig+4,del_l,del_t)

    PLOT_VIDEO_1D(f_time[:,:,:,source_par[0],:],del_l,fig_num = last_fig+5)

    plt.show()