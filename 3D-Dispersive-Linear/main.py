import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plots import*
from parameters import  NL_MULTIPLE_DISPERSION_PARAMETERS
from fields import SOURCE , TIME_SOURCE_LUMERICAL , LINE_SOURCE_E , MULTIPLE_SOURCE
from weights import WEIGHT_CREATION , WEIGHT_INDEXING , WEIGHT_CREATION_TEST
from layers import NL_MULTIPLE_PROPAGATE , NL_MULTIPLE_PROPAGATE_TRAIN , SPECTRUM_Z , NONLINEAR_OVERLAP_INTEGRAL , OVERLAP_INTEGRAL
import pickle

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

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well

### ---------------Global Constants ---------------------------- ###
n_f = 6                       #number of field components at node
n_c = 12                      #number of voltage components at node
c0 = 2.99792458e8             #speed of light in vaccum (m/s)

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
    PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num=[4,5,6],location = source_par[3],del_t = del_t)

    #--------------------------- Tensor Creation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        v_f = tf.convert_to_tensor(v_f,dtype = data_type)

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    with tf.name_scope('graph'):

        v_i , f_time, f_final = NL_MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f)

    print("Done!\n")

    #------------------ Spectrum -------------------------------------------#
    freq_1 = c0/1.5e-6
    sp_1 , sp_2 = SPECTRUM_Z(tf.complex(f_time,f_time*0),del_t,n_t,freq_1,freq_1*2)

    #--------------------------- Merge Summaries ---------------------------#
    merged = tf.summary.merge_all()

    #--------------------------- Run Forward Model --------------------------#
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run( tf.global_variables_initializer() )

        print("--------- Starting Run of Forward Model ---------\n")

        # run the graph to determine the output field
        f_time = sess.run(f_time)
        f_final = sess.run(f_final)
        sp_2 = sess.run(sp_2)
        sp_1 = sess.run(sp_1)

        # plot results
        PLOT_RESULTS_2(f_final[:,:,:,2],del_l,fig_num = 7)

        plt.figure(8)
        plt.plot(np.abs(sp_1),label = 'res freq')
        plt.plot(np.abs(sp_2),label = 'double res freq')
        plt.xlabel('position')
        plt.ylabel('mag')
        plt.title('spectrum')
        plt.legend()
        plt.show()

        plt.figure(9)
        plt.plot(np.abs(np.conj(sp_1**2)*sp_2),label = 'real')
        plt.plot(np.angle(np.conj(sp_1**2)*sp_2),label = 'angle')
        plt.xlabel('position')
        plt.ylabel('mag')
        plt.title('spectrum')
        plt.legend()
        plt.show()

        PLOT_VIDEO_1D(f_time[:,:,:,source_par[0],:],del_l,fig_num = 9)

def INVERSE(n_x,n_y,n_z,n_t,del_l,source_par,mat_par,train_par):
    # performs a inverse linear multiple Lorentz resonace simulation
    # n_x: number of spatial parameters in the x or 0th axis direction - tf.constant (int), shape(1,)
    # n_y: number of spatial parameters in the y or 1st axis direction - tf.constant (int), shape(1,)
    # n_z: number of spatial parameters in the z or 2nd axis direction - tf.constant (int), shape(1,)
    # n_t: number of time steps the simulation takes - tf.constant (int), shape(1,)
    # del_l: the lenght of the mesh step in all three directions (m) - tf.constant (int), shape(1,)
    # n_r: number of resonances - tf.constant (int), shape(1,)
    # source_par: contains the parameters relevent for making the source - list, shape(11,)
    # mat_par: contains the parameters relevent for the material - list, shape(9,)
    # train_par: contains the parameters relevent for training - list, shape()

    #determine time step based on the criteria del_l/del_t = 2*c0
    del_t = del_l/(2*c0)

    # ----------------- Simulation Parameters ---------------------- #
    inf_x,w_0,damp,del_x,x_nl = NL_MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,mat_par[0],mat_par[1],mat_par[2],mat_par[3],mat_par[4],mat_par[5],mat_par[6],mat_par[7],mat_par[8])
    PLOT_DISPERSION_PARAMETERS_1D(inf_x[0,:,0],w_0[0,:,0,:],damp[0,:,0,:],del_x[0,:,0,:],fig_num = [1,2])

    #------------------------ Create Weights ------------------------#
    with tf.name_scope('create_weights'):
        weights_tens , weights_train_tens = WEIGHT_CREATION(mat_par[6],mat_par[7], n_x, n_y, n_z)
        #weights_tens = WEIGHT_CREATION_TEST(n_x, n_y, n_z)

    # ----------------- Source ------------------------------------- #
    v_f,time_source,current_density = MULTIPLE_SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,source_par[0],source_par[1],source_par[2],source_par[3],source_par[4],source_par[5],source_par[6],source_par[7],source_par[8],source_par[9],source_par[10])
    location = source_par[3]
    PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num=[4,5,6],location = location[:,0],del_t = del_t)

    #--------------------------- Tensor Creation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        v_f_tens = tf.placeholder(dtype = tf.float32, shape = [n_x,n_y,n_z,n_f,n_t],name = 'free_source_fields_placeholder')

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")

    with tf.name_scope('graph'):

        v_i , f_time, f_final = NL_MULTIPLE_PROPAGATE_TRAIN(v_f_tens,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f,weights_tens)

        f_time_cavity = tf.slice(f_time,np.concatenate((mat_par[6],np.array([0,0]))),np.concatenate((mat_par[7],np.array([n_f,n_t])))-np.concatenate((mat_par[6],np.array([1,1])))+1)

        if train_par[6] >= 0.5/del_t:
            ValueError('double frequency range is too high')

        sp_1 , sp_2 , del_freq = SPECTRUM_Z(tf.complex(f_time_cavity[:,:,:,:,int(source_par[2]/del_t):n_t],f_time_cavity[:,:,:,:,int(source_par[2]/del_t):n_t]*0),del_t,n_t,train_par[3],train_par[4],train_par[5],train_par[6])

        print('del_freq:',del_freq*10**-12)

        overlap_integral = NONLINEAR_OVERLAP_INTEGRAL(sp_1,sp_2,del_l,del_freq,tf.complex(weights_train_tens[0,:,0],weights_train_tens[0,:,0]*0))
        #overlap_integral = OVERLAP_INTEGRAL(sp_1,sp_2,del_l,del_freq)

        loss_func = - overlap_integral

    print("Done!\n")

    # #--------------------------- Define Optimizer --------------------------#
    print("Building Optimizer ... ... ...")
    lr = train_par[0]
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_func, var_list = [weights_train_tens])
    print("Done!\n")

    #--------------------------- Merge Summaries ---------------------------#
    merged = tf.summary.merge_all()

    #--------------------------- Training --------------------------#
    epochs = train_par[1]

    # if the results folder does not exist for the current model, create it
    if not os.path.exists(train_par[2]):
            os.makedirs(train_par[2])

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        sess.run( tf.global_variables_initializer())

        print("--------- Starting Training ---------\n")
        for i in range(1, epochs+1):

            # run v_f dynamically into the network per iteration
            _,loss_value,spectrum_1,spectrum_2,weights,field = sess.run([train_op, loss_func,sp_1,sp_2,weights_tens,f_time],  feed_dict = {v_f_tens : v_f} )

            print('Epoch: ',i)
            print('Loss: ',loss_value)

            results = [loss_value,weights,spectrum_1,spectrum_2,field]

            with open(train_par[2]+"/epoch_"+(str(i))+".pkl","wb") as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    plt.figure(100)
    plt.imshow(np.abs(spectrum_1))
    plt.title('spectrum 1 mag')
    plt.colorbar

    plt.figure(101)
    plt.imshow(np.angle(spectrum_1))
    plt.title('spectrum 1 phase')
    plt.colorbar

    plt.figure(102)
    plt.imshow(np.abs(spectrum_2))
    plt.title('spectrum 2 mag')
    plt.colorbar

    plt.figure(103)
    plt.imshow(np.angle(spectrum_2))
    plt.title('spectrum 2 phase')
    plt.colorbar

    plt.figure(104)
    plt.plot(1/(1+np.exp(-100*np.squeeze(weights))))
    plt.title('sigmoid of weights')

    plt.figure(105)
    plt.plot(np.squeeze(weights))
    plt.title('weights')    

    #plt.show()

def MULTIPLE_COMPARE(n_c,n_x,n_y,n_z,n_t,del_l,del_t,del_x,location,time_source,polarization,f_lumerical_time,mat_par):
    #compares the Lumerical results with the forward model STSN results

    # ----------------- Source ------------------------------------- #
    v_f_time = TIME_SOURCE_LUMERICAL(polarization,n_f,del_t,n_t,del_l,time_source)
    v_f = LINE_SOURCE_E(location,1,v_f_time,n_x,n_y,n_z)

    e_time_source = time_source

    PLOT_TIME_SOURCE_LUMERICAL(v_f,fig_num=1,location = location)

    #--------------------------- Tensor Creation --------------------------#
    with tf.name_scope('instantiate_placeholders'):
        v_f = tf.convert_to_tensor(v_f*1.0e16,dtype = data_type)
    
    # ----------------- Simulation Parameters ---------------------- #
    inf_x,w_0,damp,del_x = MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,mat_par[0],mat_par[1],mat_par[2],mat_par[3],mat_par[4])
    PLOT_DISPERSION_PARAMETERS_1D(inf_x[0,:,0],w_0[0,:,0,:],damp[0,:,0,:],del_x[0,:,0,:],fig_num = [2,3])

    #--------------------------- Graph Construction --------------------------#
    # compute least squares cost for each sample and then average out their costs
    print("Building Graph ... ... ...")
        
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
        PLOT_COMPARISON_VIDEO_1D(f_time[:,:,:,polarization,:],f_lumerical_time,del_l,fig_num = 4)

        PLOT_RESULTS_2(f_final[:,:,:,2],del_l,fig_num = 5)

        plt.show()