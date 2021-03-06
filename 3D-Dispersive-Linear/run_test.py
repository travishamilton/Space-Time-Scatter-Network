import numpy as np
from main import FORWARD , INVERSE , INVERSE_ANNEALING
import pickle
#from files import SAVE_PARAMS

##################################################################
################## 1D Lithium Niobate Example ####################
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

n_x = 1
n_y = 200
n_z = 1 

n_t = 10000

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

#simulaion points (x)
n_x = 1
#simulaion points (y)
n_y = 300
#simulaion points (z)
n_z = 1

#time points
n_t = 4000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

#location of source (pts.)
location = np.zeros((3,1),dtype = np.int32)
location[0,:] = np.array([0])
location[1,:] = np.array([25])
location[2,:] = np.array([0])
#polarization of source
polarization = 2
#center of wavelength
wavelength = 1.550e-6
#injection axis
injection_axis = 1
#injection direction
injection_direction = 0
#fwhm of source (s)
fwhm = 0.5*wavelength/c0

###not needed for now###
fwhm_mode = 15
n_m = 0
center_mode = 0
mode_axis = 0
### ---------------- ###

#source type
source_type = 'Line'
#step size (m)
del_l = 25e-9

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

#number of Lorentz resonances
n_r = 2
#ignore
n_m = 2
#infinite susceptibility
inf_x_mat = 0
#Lorentz resonances
w_0_mat = np.array([10105,9037])*10**12
#Lorentz damping resonances
damp_mat = 0.0*w_0_mat
a_i_mat = np.array([3.6613,0.1776])
#Lorentz change in susceptibility
del_x_mat = a_i_mat
x_nl = 10e-12/del_l
#Chi 2
x_nl = 0
#Device start
mask_start = np.array([0,50,0])
#Device end
mask_end = np.array([0,250,0])

mat_par = [n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl,mask_start,mask_end,n_m]

# ------------------------------- ------------------------------ #
# --------------------- Training Parameters -------------------- #
# ------------------------------- ------------------------------ #

#learning rate
lr = 0.01
#epochs
epochs = 4000
#Pumping end frequency
freq_1_end = c0/1.45e-6
#Pumping start frequency
freq_1_start = c0/1.65e-6
#SHG start frequency
freq_2_start = 2*freq_1_start
#SHG end frequency
freq_2_end = 2*freq_1_end

loss_path = "broad_band/nt_"+str(n_t)+"_ny_"+str(n_y)+"_lr_"+str(lr)

#print frequency range
print('starting freq (THz): ',freq_1_start*10**-12)
print('ending freq (THz): ',freq_1_end*10**-12)

#initial weight
w_0 = 0

#initial slope of weight sigmoid
init_slope = 100
#change to slope 
change_slope = 100
#number of epochs before slope changes
epoch_step = 100

train_par = [lr,epochs,loss_path,freq_1_start,freq_1_end,freq_2_start,freq_2_end,w_0,init_slope,change_slope,epoch_step]

n_train = 1

# ------------------------------- ------------------------------ #
# ---------------------- Relevant Parameters ------------------- #
# ------------------------------- ------------------------------ #
#time step (s)
del_t = del_l/(2*c0)
#device size (m)
device_size_y = del_l*(mask_end[1] - mask_start[1] + 1)
#source location (m)
source_location_y = location[1]*del_l
#frequency step (Hz)
del_freq = 1/(del_t*n_t)
#space length (m)
space_size_y = del_l*n_y
#time length (s)
time_size = n_t*del_t


# ------------------------------- ------------------------------ #
# --------------------- Run Multiple Inverse ------------------- #
# ------------------------------- ------------------------------ #

for i in range(n_train):

	INVERSE(n_x,n_y,n_z,n_t,del_l,source_par,mat_par,train_par)

	#SAVE_PARAMS(n_x,n_y,n_z,n_t,c0,location,polarization,wavelength,injection_axis,injection_direction,fwhm,source_type,del_l,n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl,mask_start,mask_end,lr,epochs,freq_1_start,freq_1_end,freq_2_start,freq_2_end,w_0,del_t,device_size_y,source_location_y,del_freq,space_size_y,time_size,train_par[2])

##################################################################
############# Inverse 1D Lithium Niobate Annealing ###############
##################################################################

# ------------------------------- ------------------------------ #
# -------------------- Simulation Parameters ------------------- #
# ------------------------------- ------------------------------ #

#simulaion points (x)
n_x = 1
#simulaion points (y)
n_y = 300
#simulaion points (z)
n_z = 1

#time points
n_t = 2000

# ------------------------------- ------------------------------ #
# ---------------------- Source Parameters --------------------- #
# ------------------------------- ------------------------------ #

#location of source (pts.)
location = np.zeros((3,1),dtype = np.int32)
location[0,:] = np.array([0])
location[1,:] = np.array([25])
location[2,:] = np.array([0])
#polarization of source
polarization = 2
#center of wavelength
wavelength = 1.550e-6
#injection axis
injection_axis = 1
#injection direction
injection_direction = 0
#fwhm of source (s)
fwhm = 0.5*wavelength/c0

###not needed for now###
fwhm_mode = 15
n_m = 0
center_mode = 0
mode_axis = 0
### ---------------- ###

#source type
source_type = 'Line'
#step size (m)
del_l = 25e-9

source_par = [polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis]

# ------------------------------- ------------------------------ #
# --------------------- Material Parameters -------------------- #
# ------------------------------- ------------------------------ #

#number of Lorentz resonances
n_r = 2
#ignore
n_m = 2
#infinite susceptibility
inf_x_mat = 0
#Lorentz resonances
w_0_mat = np.array([10105,9037])*10**12
#Lorentz damping resonances
damp_mat = 0.0*w_0_mat
a_i_mat = np.array([3.6613,0.1776])
#Lorentz change in susceptibility
del_x_mat = a_i_mat
x_nl = 10e-12/del_l
#Chi 2
x_nl = 0
#Device start
mask_start = np.array([0,50,0])
#Device end
mask_end = np.array([0,250,0])

mat_par = [n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl,mask_start,mask_end,n_m]

# ------------------------------- ------------------------------ #
# --------------------- Training Parameters -------------------- #
# ------------------------------- ------------------------------ #

#learning rate
lr = 0.03
#epochs
epochs = 100
#Pumping end frequency
freq_1_end = c0/1.55e-6
#Pumping start frequency
freq_1_start = c0/1.55e-6
#SHG start frequency
freq_2_start = 2*freq_1_start
#SHG end frequency
freq_2_end = 2*freq_1_end
#location and name of data folder
loss_path = "annealing/nt_"+str(n_t)+"_ny_"+str(n_y)+"_lr_"+str(lr)
#initial weight
w_0 = 0
#initial slope of weight sigmoid
init_slope = 100
#change to slope 
change_slope = 100
#number of epochs before slope changes
epoch_step = 100

train_par = [lr,epochs,loss_path,freq_1_start,freq_1_end,freq_2_start,freq_2_end,w_0,init_slope,change_slope,epoch_step]

n_train = 1

# ------------------------------- ------------------------------ #
# ---------------------- Relevant Parameters ------------------- #
# ------------------------------- ------------------------------ #
#time step (s)
del_t = del_l/(2*c0)
#device size (m)
device_size_y = del_l*(mask_end[1] - mask_start[1] + 1)
#source location (m)
source_location_y = location[1]*del_l
#frequency step (Hz)
del_freq = 1/(del_t*n_t)
#space length (m)
space_size_y = del_l*n_y
#time length (s)
time_size = n_t*del_t


# ------------------------------- ------------------------------ #
# ------------------------- Run Inverse ------------------------ #
# ------------------------------- ------------------------------ #


# weights_tens,weights_train_tens = INVERSE_ANNEALING(n_x,n_y,n_z,n_t,del_l,source_par,mat_par,train_par,[],[],0)


# for j in range(1,15):
# 	train_par[8] = train_par[8] + train_par[9]
# 	weights_tens,weights_train_tens = INVERSE_ANNEALING(n_x,n_y,n_z,n_t,del_l,source_par,mat_par,train_par,weights_tens,weights_train_tens,j)

