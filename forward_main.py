import numpy as np
import os
import matplotlib.pyplot as plt
import time

os.system('cls' if os.name == 'nt' else 'clear')

from forward_layers import PROPAGATE
from forward_parameters import ALPHA , REFRACTIVE_INDEX
from forward_fields import POINT_SOURCE , SCATTER_2_ELECTRIC_NODES , TENSORIZE
from forward_plots import PLOT_BOUNDARIES , PLOT_TIME_SOURCE , PLOT_VIDEO
from forward_files import SAVE_FEILD_DATA , FILE_ID , SAVE_MESH_DATA

# ------------ Simulation Parameters ----------------------- #
n_i = 70
n_j = 70
n_k = 1

n_c = 12
n_t = 100

time_changes = 0
scatter_type = 'none'
mask = np.array([[25,25,0],[45,45,0]])

alpha = ALPHA(n_i,n_j,n_k)
n = REFRACTIVE_INDEX(n_i,n_j,n_k,scatter_type)

file_id = FILE_ID(n_t,n_i,n_j,n_k,scatter_type,mask,time_changes)

# ------------- Source Parameters --------------------------- #
location = (5,1,0)
polarization = 2
wavelength = 30
full_width_half_maximum = 3*wavelength

source_scatter_field_vector,source_time = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum)

# ------------- Propagate ------------------------------------- #
scatter_field_vector , scatter_field_vector_time , transfer_matrix = PROPAGATE(alpha,n,n_c,source_scatter_field_vector,n_t)

E0,E1,E2 = SCATTER_2_ELECTRIC_NODES(scatter_field_vector,n_c,n,alpha)

# --------------- Plot Results -------------------------------- #
PLOT_TIME_SOURCE(source_time,n[location],alpha[location],fig_num = 1)

plt.figure(2)
plt.imshow(E2[:,:,0])
plt.title('E2')
plt.colorbar()

plt.figure(3)
plt.imshow(E1[:,:,0])
plt.title('E1')
plt.colorbar()

plt.figure(4)
plt.imshow(E0[:,:,0])
plt.title('E0')
plt.colorbar()

plt.figure(7)
plt.imshow(n[:,:,0,0])
plt.title('Refractive Index')
plt.colorbar()

PLOT_VIDEO(scatter_field_vector_time,n_c,n,alpha)

plt.show()

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