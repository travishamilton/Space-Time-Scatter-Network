import numpy as np
from main import FORWARD , COMPARE , MULTIPLE_FORWARD
from lumerical_read import readFields

# ----------------------- Simulation Constants ------------------#
n_c = 12    # number of voltage components per node
n_f = 6     # number of field components per node

n_w = 1     # number of weights per node

##################################################################
################## 1D Single Resonances Example ##################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 150
n_z = 1 

n_t = 300

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

location = (0,1,0)
polarization = 2
wavelength = 1550e-6
n_wavelength = 30
injection_axis = 1
injection_direction = 0
fwhm = 1.5*wavelength
fwhm_mode = 15
n_m = 0
center_mode = 0
mode_axis = 0
source_type = 'Line'
del_l = wavelength/n_wavelength

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

#set materail parameters - for each resonance
inf_x_mat = 0
w_0_mat = 2*np.pi*np.array([14192])*10**12
damp_mat = 0.01*w_0_mat
a_i_mat = np.array([2.9804])
del_x_mat = a_i_mat
n_r = 1

mat_par = [n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat]

# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Forward ------------------- #
# ------------------------------- ------------------------------ #

MULTIPLE_FORWARD(n_x,n_y,n_z,n_t,del_l,source_par,mat_par)
