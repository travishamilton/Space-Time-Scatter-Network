import numpy as np
from main import *

##################################################################
################## 1D Lithium Niobate Example ####################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 3000
n_z = 1 

n_t = 10000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
location = (0,10,0)
polarization = 2
wavelength = 1.500e-6
injection_axis = 1
injection_direction = 0
fwhm = 3.5*wavelength/c0
fwhm_mode = 15
n_m = 2
center_mode = 0
mode_axis = 0
source_type = 'Line'
del_l = 5e-9

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

n_r = 2
n_m = 2
inf_x_mat = 0
inf_x_mat = 1.25
w_0_mat = np.array([10105,9037])*10**12
damp_mat = 0.0*w_0_mat
a_i_mat = np.array([3.6613,0.1776])*0
del_x_mat = a_i_mat
x_nl = 10e-12/del_l
mask_start = np.array([0,75,0])
mask_end = np.array([0,125,0])

mask_start = (0,500,0)
mask_end = (0,3900,0)

mat_par = [n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl,mask_start,mask_end,n_m]

# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

#LARGE_FIELD_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)

##################################################################
###################### 1D High n Material ########################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 1000
n_z = 1 

n_t = 1000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
location = (0,500,0)
polarization = 2
wavelength = 1.500e-6
injection_axis = 1
injection_direction = 0
fwhm = 0.5*wavelength/c0
fwhm_mode = 15
n_m = 2
center_mode = 0
mode_axis = 0
source_type = 'Line'
del_l = 5e-9

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

n_r = 1
n_m = 2
n = 20

#multiple materials have the following array format
# [ [ material 1 ] , [ material 2 ] ]
#each material is time dependent and changes at time step t_change
#if we look at on material array [ material 1 ], it has two components:
# [ material 1 ] = [ material 1 before t_change , material 1 after t_change]
# 
# for example, if there are two materials both with a inf_x_mat value of 3 before 
# t_change and a value of 1.5 for materail 1 and 2.33 for materail 2 after t_change,
# the inf_x_mat will look like this:
# [[3,1.5],[3,2.33]]
#
# if you wanted a non-time dependent material, simply keep the parameters the same
# after the time change. For example, if you have a material with infinite susceptibility
# of 3 and 2 respectivelly, their inf_x_mat would look like
# [[3,3],[2,2]]

inf_x_mat = np.array([[n**2-1,n**2-1],[n**2-1,n**2-1]])
w_0_mat = np.array([[0,0],[0,0]])
damp_mat = np.array([[0,0],[0,0]])
x_nl_mat = np.array([[0,0],[0,0]])
del_x_mat = np.array([[0,0],[0,300]])
 
x_nl = 0/del_l

mat_start = np.array([[0,450,0],[0,650,0]]) 
mat_end = np.array([[0,550,0],[0,750,0]])
t_change = 500

print('0 mat_start: ',mat_start[0,:])
print('1 mat_start: ',mat_start[1,:])
 
mat_par = [inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl_mat,mat_start,mat_end,n_m,t_change]


# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

TIME_DEP_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)