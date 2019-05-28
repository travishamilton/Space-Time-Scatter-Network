import numpy as np
from main import *
import time

##################################################################
################## 1D Lithium Niobate Example ####################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 800
n_z = 1 

n_t = 3000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
location = (0,10,0)
polarization = 2
wavelength = 0.9e-6
injection_axis = 1
injection_direction = 0
fwhm = 1.5*wavelength/c0
fwhm_mode = 15
n_m = 2
center_mode = 0
mode_axis = 0
source_type = 'Line'
del_l = 10.0e-9

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

n_r = 2
n_m = 2
inf_x_mat = 0
w_0_mat = np.array([10105,9037])*1.0e12
damp_mat = 0.0*w_0_mat
a_i_mat = np.array([3.7916,0.0575])
del_x_mat = a_i_mat
x_nl = 40e-12/del_l
x_nl = 0

mask_start = (0,0,0)
mask_end = (0,n_y*2,0)

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

n_t = 20000

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
del_l = 1.0e-9

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
f_0_mat = np.array([[0,0],[100e12,100e12]])
w_0_mat = 2*np.pi*f_0_mat
damp_mat = 0.5*w_0_mat
x_nl_mat = np.array([[0,0],[0,0]])
del_x_mat = np.array([[300,300],[300,300]])
 
#x_nl = 0/del_l

mat_start = np.array([[0,0,0],[0,750,0]]) 
mat_end = np.array([[0,749,0],[0,1000,0]])
t_change = 500
 
mat_par = [inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl_mat,mat_start,mat_end,n_m,t_change]


# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

#TIME_DEP_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)

##################################################################
################### 2D Linear Nondispersive ######################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 500
n_y = 500
n_z = 1 

n_t = 2000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
del_l = 30.0e-9

location = (0,5,0)
polarization = 2
wavelength = 1.500e-6
injection_axis = 1
injection_direction = 0
fwhm = 0.5*wavelength/c0
fwhm_mode = 70*del_l
n_m = 2
center_mode = n_x//2*del_l
mode_axis = 0
source_type = 'Mode'


source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

inf_x_mat = 1.25
mat_start = np.array([20,15,0])
mat_end = np.array([25,175,0])

mat_par = [inf_x_mat,mat_start,mat_end]


# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

#FORWARD_2D_LINEAR_NONDISPERSIVE(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)

##################################################################
### 2D Linear Nondispersive Multiple Materials Time Dependent ####
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 200
n_y = 200
n_z = 1 

n_t = 600

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
del_l = 10.0e-9

location = (100,100,0)
polarization = 2
wavelength = 1.500e-6
injection_axis = 1
injection_direction = 0
fwhm = 0.5*wavelength/c0
fwhm_mode = 70*del_l
n_m = 2
center_mode = n_x//2*del_l
mode_axis = 0
source_type = 'Mode'


source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

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

n1 = 1.2  #refractive inedex - material 1
n2 = 1.2  #refractive index - material 2
inf_x_mat = np.array([[n1**2-1,n1**2-1],[n2**2-1,n2**2-1]])

mat_start = np.array([[0,0,0],[196,196,0]]) 
mat_end = np.array([[195,195,0],[200,200,0]])
t_change = 500
n_m,_ = np.shape(inf_x_mat)
 
mat_par = [inf_x_mat,mat_start,mat_end,n_m,t_change]


# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

TIME_DEP_FORWARD_2D_LINEAR_NONDISPERSIVE(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)
