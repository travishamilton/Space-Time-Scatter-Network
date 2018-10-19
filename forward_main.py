import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pickle

os.system('cls' if os.name == 'nt' else 'clear')

from forward_layers import PROPAGATE
from forward_parameters import ALPHA , REFRACTIVE_INDEX
from forward_fields import POINT_SOURCE , SCATTER_2_ELECTRIC_NODES , TENSORIZE , FIELD_ENERGY , LINE_SOURCE , MODE_SOURCE
from forward_plots import PLOT_BOUNDARIES , PLOT_TIME_SOURCE , PLOT_VIDEO , PLOT_RESULTS
from forward_files import SAVE_FEILD_DATA , FILE_ID , SAVE_MESH_DATA

# ------------ Simulation Parameters ----------------------- #
n_i = 50
n_j = 250
n_k = 1 

n_c = 12
n_t = 500

time_changes = 0
scatter_type = 'none'
mask = np.array([[39,39,0],[61,61,0]])

alpha = ALPHA(n_i,n_j,n_k)
n = REFRACTIVE_INDEX(n_i,n_j,n_k,scatter_type)

# --------- Temp Test -----------------------------------------#
# time_steps = '500'
# space_steps_x = '100'
# space_steps_y = '100'
# space_steps_z = '1'
# scatter_type = 'cylinder'
# mask_start_x = '39'
# mask_start_y = '39'
# mask_start_z = '0'
# mask_end_x = '61'
# mask_end_y = '61'
# mask_end_z = '0'
# time_changes = '0'

# epoch = '6000'

# file_address = "C:/Users/travi/Documents/Northwestern/STSN/results/"
# file_id = "timeSteps_" + time_steps + "_spaceSteps_" + space_steps_x + "_" + space_steps_y + "_" + space_steps_z + "_scatterType_" + scatter_type + "_maskStart_" + mask_start_x + "_" + mask_start_y + "_" + mask_start_z + "_maskEnd_" + mask_end_x + "_" + mask_end_y + "_" + mask_end_z + "_timeChanges_" + time_changes

# with open(file_address + file_id + "/" + epoch + '_trained_weights.pkl', 'rb') as f:
#     weights = pickle.load(f)

# n_mask = np.sqrt(1/weights)
# n = np.ones((n_i,n_j,n_k,1))
# n[mask[0,0]:mask[1,0]+1,mask[0,1]:mask[1,1]+1,mask[0,2]:mask[1,2]+1,0:1] = n_mask

# -------------------------------------------------------------#

file_id = FILE_ID(n_t,n_i,n_j,n_k,scatter_type,mask,time_changes)

# ------------- Source Parameters --------------------------- #
center = (25,50,0)
polarization = 2
wavelength = 80
injection_axis = 1
injection_direction = 0
full_width_half_maximum = 3*wavelength
full_width_half_maximum_mode = 15

source_scatter_field_vector,source_time = LINE_SOURCE(center,injection_axis,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_direction)

injection_direction = 1
source_scatter_field_vector_tmp,source_time_tmp = LINE_SOURCE(center,injection_axis,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_direction)

source_scatter_field_vector = 0.55*source_scatter_field_vector_tmp + source_scatter_field_vector

# ------------- Propagate ------------------------------------- #
scatter_field_vector , scatter_field_vector_time , transfer_matrix = PROPAGATE(alpha,n,n_c,source_scatter_field_vector,n_t)

E0,E1,E2 = SCATTER_2_ELECTRIC_NODES(scatter_field_vector,n_c,n,alpha)

# --------------- Plot Results -------------------------------- #
PLOT_TIME_SOURCE(source_time,n[center],alpha[center],fig_num = 1)

PLOT_RESULTS(E0,E1,E2,n,fig_nums = (2,3,4,5))

PLOT_VIDEO(scatter_field_vector_time,n_c,n,alpha)

plt.show()

# ------------- Calculate Field Power out of Masked Region ------ #
FIELD_ENERGY(E0,E1,E2,n[:,:,:,0],mask)

# -------------- Save Data in Pickle Files ----------------------- #
file_id = FILE_ID(n_t,n_i,n_j,n_k,scatter_type,mask,time_changes)

field_out = TENSORIZE(scatter_field_vector_time[:,n_t-1],n_i,n_j,n_k,n_c)
field_in = np.zeros((n_i,n_j,n_k,n_c,n_t))

for t in range(n_t):
    field_in[:,:,:,:,t] = TENSORIZE(source_scatter_field_vector[:,t],n_i,n_j,n_k,n_c)

file_address = "C:/Users/travi/Documents/Northwestern/STSN/field_data/"

SAVE_FEILD_DATA(field_in,field_out,file_id,file_address)

file_address = "C:/Users/travi/Documents/Northwestern/STSN/mesh_data/"

SAVE_MESH_DATA(alpha,n,file_id,file_address)

plt.show()