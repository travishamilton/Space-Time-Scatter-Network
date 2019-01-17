import numpy as np
from main import FORWARD , INVERSE

##################################################################
################## 1D Lithium Niobate Example ####################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 200
n_z = 1 

n_t = 1000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
location = (0,100,0)
polarization = 2
wavelength = 1.500e-6
n_wavelength = 50
injection_axis = 1
injection_direction = 0
fwhm = 0.5*wavelength/c0
fwhm_mode = 15
n_m = 2
center_mode = 0
mode_axis = 0
source_type = 'Line'
del_l = wavelength/n_wavelength

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

n_r = 2
n_m = 2
inf_x_mat = 0
w_0_mat = np.array([10105,9037])*10**12
damp_mat = 0.0*w_0_mat
a_i_mat = np.array([3.6613,0.1776])
del_x_mat = a_i_mat
x_nl = 10e-12/del_l
x_nl = 0
mask_start = np.array([0,75,0])
mask_end = np.array([0,125,0])

mask_start = (0,75,0)
mask_end = (0,125,0)

mat_par = [n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl,mask_start,mask_end,n_m]

# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

#FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)

##################################################################
############## Inverse 1D Lithium Niobate Example ################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 200
n_z = 1

n_t = 300

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

c0 = 2.99792458e8
location = (0,100,0)
polarization = 2
wavelength = 1.500e-6
injection_axis = 1
injection_direction = 0
fwhm = 0.5*wavelength/c0
fwhm_mode = 15
n_m = 0
center_mode = 0
mode_axis = 0
source_type = 'Line'
del_l = 25e-9

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

n_r = 2
n_m = 2
inf_x_mat = 0
w_0_mat = np.array([10105,9037])*10**12
damp_mat = 0.0*w_0_mat
a_i_mat = np.array([3.6613,0.1776])
del_x_mat = a_i_mat
x_nl = 10e-12/del_l
x_nl = 0
mask_start = np.array([0,75,0])
mask_end = np.array([0,125,0])

mat_par = [n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl,mask_start,mask_end,n_m]

# ------------------------------- ------------------------------ #
# --------------------- Training Parameters -------------------- #
# ------------------------------- ------------------------------ #

lr = 0.01
epochs = 400
freq_1_end = c0/1.2e-6
freq_1_start = c0/1.8e-6
freq_2_start = 2*freq_1_start
freq_2_end = 2*freq_1_end

loss_path = "testing_R1/nt_"+str(n_t)+"_nw_"+str(n_wavelength)+"_epochs_"+str(epochs)

train_par = [lr,epochs,loss_path,freq_1_start,freq_1_end,freq_2_start,freq_2_end]


# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Inverse ------------------- #
# ------------------------------- ------------------------------ #

INVERSE(n_x,n_y,n_z,n_t,del_l,source_par,mat_par,train_par)

# ------------------------------- ------------------------------ #
# ---------------------- Relevant Parameters ------------------- #
# ------------------------------- ------------------------------ #
del_t = del_l/(2*c0)
device_size_y = del_l*(mask_end[1] - mask_start[1] + 1)
source_location_y = location[1]*del_l
source_fwhm = fwhm
del_freq = 1/(del_t*n_t)
space_size_y = del_l*n_y
time_size = n_t*del_t

