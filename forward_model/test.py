import numpy as np
import os
import matplotlib.pyplot as plt
import time

os.system('cls' if os.name == 'nt' else 'clear')

from layers import PROPAGATE
from parameters import ALPHA , REFRACTIVE_INDEX
from fields import POINT_SOURCE , SCATTER_2_ELECTRIC_NODES
from plots import PLOT_BOUNDARIES , PLOT_TIME_SOURCE , PLOT_VIDEO


n_i = 70
n_j = 70

n_k = 1

n_c = 12
n_t = 380

location = (5,1,0)
polarization = 2
wavelength = 30
full_width_half_maximum = 3*wavelength

alpha = ALPHA(n_i,n_j,n_k)
n = REFRACTIVE_INDEX(n_i,n_j,n_k,0)

source_scatter_field_vector,source_time = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum)
#plot time source
PLOT_TIME_SOURCE(source_time,n[location],alpha[location],fig_num = 1)

#location = (1,2,0)

#source_scatter_field_vector_temp = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum)
 
#source_scatter_field_vector = source_scatter_field_vector + source_scatter_field_vector_temp



scatter_field_vector , scatter_field_vector_time , transfer_matrix = PROPAGATE(alpha,n,n_c,source_scatter_field_vector,n_t)




E0,E1,E2 = SCATTER_2_ELECTRIC_NODES(scatter_field_vector,n_c,n,alpha)


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

# E2_time_monitor = np.zeros(n_t,dtype = float)

# for t in range(n_t):

#     E0,E1,E2 = SCATTER_2_ELECTRIC_NODES(scatter_field_vector_time[:,t],n_c,n,alpha)

#     E2_time_monitor[t] = E2[35,1,0]

#fft of point time monitor
# E2_fft_monitor = np.fft.fft(E2_time_monitor)
# time = np.arange(0,n_t*2.965e-12,2.965e-12)
# freq = np.fft.fftfreq(time.shape[-1], d = 2.965e-12)

# #find the frequency response
# n_f = 200
# f = np.linspace(1,100,n_f)
# E2_fft_monitor = np.zeros(n_f,dtype = complex)

# for index_f in range(n_f):
#     for k in range(n_t):
    
#         E2_fft_monitor[index_f] = E2_time_monitor[k]*complex(np.cos(2*np.pi*k*f[index_f]) , np.sin(2*np.pi*k*f[index_f])) + E2_fft_monitor[index_f]

# plt.figure(5)
# plt.plot(time*10**12,E2_time_monitor)
# plt.title('E2 time monitor')
# plt.xlabel('time (ps)')

# plt.figure(4)
# plt.plot(f,np.abs(E2_fft_monitor))
# plt.title('E2 time monitor')
# plt.xlabel('lambda over delat l')

# plt.figure(5)
# plt.plot(10**-9/(f*2.965e-12*2),np.abs(E2_fft_monitor))
# plt.title('E2 time monitor')
# plt.xlabel('GHz')

# plt.figure(6)
# plt.plot(freq*10**-9,np.abs(E2_fft_monitor))
# plt.title('E2 fft monitor')
# plt.xlabel('GHz')

plt.figure(7)
plt.imshow(n[:,:,0,0])
plt.title('Refractive Index')
plt.colorbar()

#PLOT_BOUNDARIES(transfer_matrix,n_i,n_j,n_k,n_c,fig_num = 8)

PLOT_VIDEO(scatter_field_vector_time,n_c,n,alpha)

#plt.show()
