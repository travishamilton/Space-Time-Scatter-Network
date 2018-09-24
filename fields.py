import numpy as np
from layers import COMPONENT_2_INDEX , NORMALIZED_CAPACITANCE , INDEX_2_COMPONENT

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
    
def TIME_SOURCE(polarization,n_c,n_t,wavelength,full_width_half_maximum,n,alpha,location):
    #produces the time source for a Gaussian wave packet
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #n_c: number of scatter components - int, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #wavelength: number of points used to define one wavelength - int, shape(1,)
    #full_width_half_maximum: full width at half maximum in the time domain - int, shape(1,)
    #n: refractive index distribution for each spatial location (i,j,k) - np.array float, shape(n_i,n_j,n_k,1)
    #alpha: ratio of space/time steps in units of c - np.array float, shape(n_i,n_j,n_k,3)
    #location: gives the location (i,j,k) of the source - tuple int, shape (3,)

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
                    #get the scatter component for a given polarization,direction and polarity
                    c = INDEX_2_COMPONENT(direction,polarization,polarity)

                    time_source[t,c] = voltage[t]/(4*normalized_capacitance[direction,polarization])
                    #time_source[t,c] = 1

    return time_source

def POINT_SOURCE(location,alpha,n_c,n_t,n,polarization,wavelength,full_width_half_maximum):
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
    source_time = TIME_SOURCE(polarization,n_c,n_t,wavelength,full_width_half_maximum,n,alpha,location)
    source_space_time = np.zeros((n_i*n_j*n_k*n_c,n_t),dtype = float)

    #build space-time source
    for t in range(n_t):
        source_space[i_location,:,k_location,:] = source_time[t,:]
        source_space_time[:,t] = VECTORIZE(source_space)

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