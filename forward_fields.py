import numpy as np
from forward_layers import COMPONENT_2_INDEX , NORMALIZED_CAPACITANCE , INDEX_2_COMPONENT

import matplotlib.pyplot as plt

def GET_PERMUTATIONS():

    out = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])

    return out

def VECTORIZE(field_tensor):
    #vectorizes the field tensor into a single arry using C - like indexing (last index changes fastest)
    #field_tensor: 4 dimensional tensor holding field data at position (i,j,k) with c field components  - float, shape (i,j,k,c)
    #field_array: field_tensor in an array sequentially listing each c long field components for each position (i,j,k) float, shape (i*j*k*c,)
    
    field_array = np.reshape(field_tensor,np.size(field_tensor))

    return field_array

def ADMITTANCE(n,alpha):
    #creates the characteristic admittance matrix corresponding to a scatter sub matrix
    #alpha: array of space to time ratios for each dimension - float, shape (3,) 
    #n: refractive index for the scatter sub cell - float, shape (1,) 
    #C: normalized capacatice of scatter sub matrix - float, shape (3,3)

    C = NORMALIZED_CAPACITANCE(alpha,n)

    #characteristic impedance
    impedance = np.ones((3,3),dtype = float)

    perm = GET_PERMUTATIONS()

    for l in range(6):

        i = perm[l,:][0]
        j = perm[l,:][1]
        k = perm[l,:][2]
        
        impedance[i,j] = alpha[j]/(alpha[i]*alpha[k]*C[i,j]*n**2)

    #characteristic admittance
    admittance = 1/impedance

    return admittance
 
def TENSORIZE(field_vector,n_i,n_j,n_k,n_c):

    field_tensor = np.reshape(field_vector,(n_i,n_j,n_k,n_c))

    return field_tensor
    
def TIME_SOURCE(polarization,n_c,n_t,wavelength,full_width_half_maximum,n,alpha,location,injection_axis,injection_direction):
    #produces the time source for a Gaussian wave packet
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #n_c: number of scatter components - int, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #wavelength: number of points used to define one wavelength - int, shape(1,)
    #full_width_half_maximum: full width at half maximum in the time domain - int, shape(1,)
    #n: refractive index distribution for each spatial location (i,j,k) - np.array float, shape(n_i,n_j,n_k,1)
    #alpha: ratio of space/time steps in units of c - np.array float, shape(n_i,n_j,n_k,3)
    #location: gives the location (i,j,k) of the source - tuple int, shape (3,)
    #injection_axis: t

    #get normalized capacitance
    location_i = location[0]
    location_j = location[1]
    location_k = location[2]

    normalized_capacitance = NORMALIZED_CAPACITANCE(alpha[location_i,location_j,location_k,:],n[location_i,location_j,location_k,:])

    #time
    t = np.arange(0,n_t,1,dtype = float)
    #angular frequency
    omega = 2*np.pi/wavelength
    #standard deviation
    sigma = full_width_half_maximum / 2.35482
    #electric field time source multiplicatvely scaled by the time step (E * del_t)
    electric_field_time_source = np.sin(-1*omega*t)*np.exp(-(t-sigma*2.5)**2/(2*sigma**2))

    #voltage polarized in the polarization direction
    voltage = -electric_field_time_source*alpha[location_i,location_j,location_k,polarization]

    #initilize a time source
    time_source = np.zeros((n_t,n_c),dtype = float)

    #build time source for each component
    for t in range(n_t):
    #for t in range(1):
        for direction in range(3):
            for polarity in range(2):

                #non-equal direction/polarization components only
                if not direction == polarization:

                    if direction == injection_axis:
                        if polarity == injection_direction:
                            #get the scatter component for a given polarization,direction and polarity
                            c = INDEX_2_COMPONENT(direction,polarization,polarity)

                            time_source[t,c] = voltage[t]/(4*normalized_capacitance[direction,polarization])
                            #time_source[t,c] = 1

    return time_source

def POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_axis,injection_direction):
    #produces a dipole point source at a given location and polarization
    #location: gives the location (i,j,k) of the source - tuple int, shape (3,)
    #alpha: ratio of space/time steps in units of c - np.array float, shape(n_i,n_j,n_k,3)
    #n_c: number of scatter components - int, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #n: refractive index distribution for each spatial location (i,j,k) - np.array float, shape(n_i,n_j,n_k,1)
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #wavelength: number of points used to define one wavelength - int, shape(1,)
    #full_width_half_maximum: full width at half maximum in the time domain - int, shape(1,)

    #get spatial information
    n_i,n_j,n_k,_ = np.shape(alpha)
    i_location = location[0]
    j_location = location[1]
    k_location = location[2]

    #initilize sources
    source_space = np.zeros((n_i,n_j,n_k,n_c),dtype = float)
    source_time = TIME_SOURCE(polarization,n_c,n_t,wavelength,full_width_half_maximum,n,alpha,location,injection_axis,injection_direction)
    source_space_time = np.zeros((n_i*n_j*n_k*n_c,n_t),dtype = float)

    #build space-time source
    for t in range(n_t):

        source_space[i_location,j_location,k_location,:] = source_time[t,:]

        source_space_time[:,t] = VECTORIZE(source_space)

    return source_space_time , source_time

def MODE_SHAPE(full_width_half_maximum,n_m,center):
    # creates the transverse shape of the mode
    # full_width_half_maximum: the full width at half maximum of the mode
    # n_m: number of points used to define the mode
    # center: center of the mode

    x = np.arange(0,n_m,1)
    sigma = full_width_half_maximum / 2.35482
    mode_shape = np.exp(-(x-center)**2/(2*sigma**2))

    plt.figure(101)
    plt.plot(mode_shape)

    return mode_shape

def MODE_SOURCE(center,injection_axis,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,full_width_half_maximum_mode):
    #produces a mode source
    # center: the center location of the plane wave
    # injection_axis: the axis direction in which the plane wave travels
    #alpha: ratio of space/time steps in units of c - np.array float, shape(n_i,n_j,n_k,3)
    #n_c: number of scatter components - int, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #n: refractive index distribution for each spatial location (i,j,k) - np.array float, shape(n_i,n_j,n_k,1)
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #wavelength: number of points used to define one wavelength - int, shape(1,)
    #full_width_half_maximum: full width at half maximum in the time domain - int, shape(1,)
    #full_width_half_maximum_mode: full width at half maximum for the transverse mode - int, shape(1,)

    n_x,n_y,n_z,_ = np.shape(alpha)

    #initilize sources
    source_space_time = np.zeros((n_x*n_y*n_z*n_c,n_t),dtype = float)

    
    # find x and y axis index values corresponding to the line source location
    if injection_axis == 0:

        #get mode shape
        mode_shape = MODE_SHAPE(full_width_half_maximum_mode,n_y,center[1])

        for y in range(n_y): 

            location = (center[0],y,0)
            
            source_space_time_tmp , source_time = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_axis)
            source_space_time = mode_shape[y]*source_space_time_tmp + source_space_time


    elif injection_axis == 1:

        #get mode shape
        mode_shape = MODE_SHAPE(full_width_half_maximum_mode,n_x,center[0])

        for x in range(n_x): 

            location = (x,center[1],0)
            
            source_space_time_tmp , source_time = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum)
            source_space_time = mode_shape[x]*source_space_time_tmp + source_space_time

    else:
        print('WARNING: injection_axis value is not recognized by LINE_SOURCE function')

    return source_space_time , source_time

def LINE_SOURCE(center,injection_axis,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_direction):
    #produces a mode source
    # center: the center location of the plane wave
    # injection_axis: the axis direction in which the plane wave travels
    #alpha: ratio of space/time steps in units of c - np.array float, shape(n_i,n_j,n_k,3)
    #n_c: number of scatter components - int, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #n: refractive index distribution for each spatial location (i,j,k) - np.array float, shape(n_i,n_j,n_k,1)
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #wavelength: number of points used to define one wavelength - int, shape(1,)
    #full_width_half_maximum: full width at half maximum in the time domain - int, shape(1,)
    #full_width_half_maximum_mode: full width at half maximum for the transverse mode - int, shape(1,)

    n_x,n_y,n_z,_ = np.shape(alpha)

    #initilize sources
    source_space_time = np.zeros((n_x*n_y*n_z*n_c,n_t),dtype = float)

    
    # find x and y axis index values corresponding to the line source location
    if injection_axis == 0:

        for y in range(n_y): 

            location = (center[0],y,0)
            
            source_space_time_tmp , source_time = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_axis,injection_direction)
            source_space_time = np.roll(source_space_time_tmp,0,axis = 1) + source_space_time

    elif injection_axis == 1:

        for x in range(n_x): 

            location = (x,center[1],0)
            
            source_space_time_tmp , source_time = POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum,injection_axis,injection_direction)
            source_space_time = np.roll(source_space_time_tmp,0,axis = 1) + source_space_time

    else:
        print('WARNING: injection_axis value is not recognized by LINE_SOURCE function')

    return source_space_time , source_time


def SCATTER_2_ELECTRIC_LINK_LINES(scatter_field_vector,n_i,n_j,n_k,n_c,n,alpha):

    scatter_field_tensor = TENSORIZE(scatter_field_vector,n_i,n_j,n_k,n_c)

    V = np.zeros((n_i,n_j,n_k,3,3,2),dtype = float)
    Y = np.zeros((n_i,n_j,n_k,3,3),dtype = float)

    V_0 = np.zeros((n_i,2*n_j-1,2*n_k-1),dtype = float)
    V_1 = np.zeros((2*n_i-1,n_j,2*n_k-1),dtype = float)
    V_2 = np.zeros((2*n_i-1,2*n_j-1,n_k),dtype = float)

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):

                admittance= ADMITTANCE(n[i,j,k,:],alpha[i,j,k,:])

                for c in range(n_c):
                    
                    direction,polarization,polarity = COMPONENT_2_INDEX(c)

                    V[i,j,k,direction,polarization,polarity] = scatter_field_tensor[i,j,k,c]
                    Y[i,j,k,direction,polarization] = admittance[direction,polarization]
    
                #polarized in the ith direction

    for i in range(n_i-1):
        for j in range(n_j-1):
            for k in range(n_k-1):

                polarit = 1

                #polarized in the zeroth direction
                dir = 1
                pol = 0

                V_0[i,2*j+1,2*k] = ( V[i,j,k,dir,pol,polarit] * Y[i,j,k,dir,pol] + V[i,j+1,k,dir,pol,polarit-1] * Y[i,j+1,k,dir,pol] ) / ( Y[i,j,k,dir,pol] + Y[i,j+1,k,dir,pol] ) 
                
                dir = 2

                V_0[i,2*j,2*k+1] = ( V[i,j,k,dir,pol,polarit] * Y[i,j,k,dir,pol] + V[i,j,k+1,dir,pol,polarit-1] * Y[i,j,k+1,dir,pol] ) / ( Y[i,j,k,dir,pol] + Y[i,j,k+1,dir,pol] ) 

                #polarized in the first direction
                dir = 0
                pol = 1

                V_1[2*i+1,j,2*k] = ( V[i,j,k,dir,pol,polarit] * Y[i,j,k,dir,pol] + V[i+1,j,k,dir,pol,polarit-1] * Y[i+1,j,k,dir,pol] ) / ( Y[i,j,k,dir,pol] + Y[i+1,j,k,dir,pol] ) 
                
                dir = 2

                V_1[2*i,j,2*k+1] = ( V[i,j,k,dir,pol,polarit] * Y[i,j,k,dir,pol] + V[i,j,k+1,dir,pol,polarit-1] * Y[i,j,k+1,dir,pol] ) / ( Y[i,j,k,dir,pol] + Y[i,j,k+1,dir,pol] ) 

                #polarized in the second direction
                dir = 0
                pol = 2

                V_2[2*i+1,2*j,k] = ( V[i,j,k,dir,pol,polarit] * Y[i,j,k,dir,pol] + V[i+1,j,k,dir,pol,polarit-1] * Y[i+1,j,k,dir,pol] ) / ( Y[i,j,k,dir,pol] + Y[i+1,j,k,dir,pol] ) 
                
                dir = 1

                V_2[2*i,2*j+1,k] = ( V[i,j,k,dir,pol,polarit] * Y[i,j,k,dir,pol] + V[i,j+1,k,dir,pol,polarit-1] * Y[i,j+1,k,dir,pol] ) / ( Y[i,j,k,dir,pol] + Y[i,j+1,k,dir,pol] ) 

            
    #Electric fields time time step delta t polarized in the zeroth, first, and second directions
    
    E_0 = V_0#/alpha[:,:,:,0]
    E_1 = V_1#/alpha[:,:,:,1]
    E_2 = V_2#/alpha[:,:,:,2]

    return E_0,E_1,E_2

def SCATTER_2_ELECTRIC_NODES(scatter_field_vector,n_c,n,alpha):
    #produces the electric field values at each node given the scatter components

    #get spatial parameters
    n_i,n_j,n_k,_ = np.shape(n)

    scatter_field_tensor = TENSORIZE(scatter_field_vector,n_i,n_j,n_k,n_c)

    V_0 = np.zeros((n_i,n_j,n_k),dtype = float)
    V_1 = np.zeros((n_i,n_j,n_k),dtype = float)
    V_2 = np.zeros((n_i,n_j,n_k),dtype = float)

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):

                normalized_capacitance = NORMALIZED_CAPACITANCE(alpha[i,j,k,:],n[i,j,k,:])

                for c in range(n_c):
                    
                    direction,polarization,_ = COMPONENT_2_INDEX(c)

                    if polarization == 0:
                        V_0[i,j,k] = normalized_capacitance[direction,polarization]*scatter_field_tensor[i,j,k,c] + V_0[i,j,k]
                    elif polarization == 1:
                        V_1[i,j,k] = normalized_capacitance[direction,polarization]*scatter_field_tensor[i,j,k,c] + V_1[i,j,k]
                    elif polarization == 2:
                        V_2[i,j,k] = normalized_capacitance[direction,polarization]*scatter_field_tensor[i,j,k,c] + V_2[i,j,k]


    #Electric fields time time step delta t polarized in the zeroth, first, and second directions
    
    E_0 = V_0#/alpha[:,:,:,0]
    E_1 = V_1#/alpha[:,:,:,1]
    E_2 = V_2#/alpha[:,:,:,2]



    
    return E_0,E_1,E_2

def FIELD_ENERGY(E_0,E_1,E_2,n,mask):
    # determines the field energy out side the masked region relative to the total energy
    # E_0: electric field polarizedin the 0th direction - shape(n_x,n_y,n_z)
    # E_1: electric field polarizedin the 1st direction - shape(n_x,n_y,n_z)
    # E_2: electric field polarizedin the 2nd direction - shape(n_x,n_y,n_z)
    # n: refractive index distribuiton - shape(n_x,n_y,n_z)
    # mask: smallest and largest coordinates of masked region - shape(3,3)

    E_squared_total = E_0**2 + E_1**2 + E_2**2
    E_squared_total = E_2**2
    E_squared_masked = E_squared_total
    E_squared_masked[mask[0,0]:mask[1,0],mask[0,1]:mask[1,1],mask[0,2]:mask[1,2]] = 0

    total_energy = np.trapz(np.trapz(np.trapz((n**2)*E_squared_total,axis = 0),axis = 0),axis = 0)
    masked_energy = np.trapz(np.trapz(np.trapz((n**2)*E_squared_masked,axis = 0),axis = 0),axis = 0)

    print('The energy outside the mask accounts for ',masked_energy/total_energy)

