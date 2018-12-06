import numpy as np
from main import FORWARD_1D , FORWARD

# ----------------------- Simulation Constants ------------------#
n_c = 12    # number of field components per node
n_w = 1     # number of weights per node

##################################################################
########################## 1D Example ############################
##################################################################

# ------------ Simulation Parameters ----------------------- #
n_x = 1
n_y = 800
n_z = 1 

n_t = 400

time_changes = 0
scatter_type = 'mask'
#index of the start of the mask and end of the mask
mask = np.array([[0,39,0],[0,61,0]])

initial_weight = 1/1.5**2 # initial weight value in masked region

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

FORWARD_1D(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type)


##################################################################
########################## 2D Example ############################
##################################################################

# ------------ Simulation Parameters ----------------------- #
n_x = 70
n_y = 70
n_z = 1 

n_t = 100

time_changes = 0
scatter_type = 'mask'
#index of the start of the mask and end of the mask
mask = np.array([[39,39,0],[61,61,0]])

initial_weight = 1/1.5**2 # initial weight value in masked region

# ------------- Source Parameters --------------------------- #
location = (25,25,0)
polarization = 2
wavelength = 20
injection_axis = 1
injection_direction = 0
fwhm = 3*wavelength
fwhm_mode = 15
n_m = 0
center_mode = 0
mode_axis = 0
source_type = 'Line'

FORWARD(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type)
