import numpy as np
import pickle

from forward_fields import TENSORIZE
from forward_plots import PLOT_TIME_SOURCE

def FILE_ID(n_t,n_i,n_j,n_k,scatter_type,mask,time_changes):

    file_id = "timeSteps_" + str(n_t) + "_spaceSteps_" + str(n_i) + "_" + str(n_j) + "_" + str(n_k) + "_scatterType_" + scatter_type + "_maskStart_" + str(mask[0,0]) + "_" + str(mask[0,1]) + "_" + str(mask[0,2]) + "_maskEnd_" + str(mask[1,0]) + "_" + str(mask[1,1]) + "_" + str(mask[1,2]) + "_timeChanges_" + str(time_changes)

    return file_id

def SAVE_FEILD_DATA(field_in,field_out,file_id,address):
    # saves data as pickle data
    # scatter_field_vector_time: contains the scatter fields in vector form - np.array shape(n_i*n_j*n_k*n_c,n_t)
    
    _,_,_,n_c,n_t = np.shape(field_in)
    source_time = np.zeros((n_t,n_c))

    with open(address + file_id + '_in.pkl', 'wb') as f:

        # Pickle input
        pickle.dump(field_in, f, pickle.HIGHEST_PROTOCOL)

    with open(address + file_id + '_out.pkl', 'wb') as f:

        # Pickle output
        pickle.dump(field_out, f, pickle.HIGHEST_PROTOCOL)

def SAVE_MESH_DATA(alpha,n,file_id,address):
    # saves mesh data as pickle data
    # scatter_field_vector_time: contains the scatter fields in vector form - np.array shape(n_i*n_j*n_k*n_c,n_t)

    with open(address + file_id + '_alpha.pkl', 'wb') as f:

        # Pickle alpha terms
        pickle.dump(alpha, f, pickle.HIGHEST_PROTOCOL)

    with open(address + file_id + '_ref_ind.pkl', 'wb') as f:

        # Pickle n terms
        pickle.dump(n, f, pickle.HIGHEST_PROTOCOL)