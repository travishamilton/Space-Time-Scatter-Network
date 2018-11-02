import numpy as np
from layers import INDEX_2_COMPONENT , COMPONENT_2_INDEX

def GET_PERMUTATIONS():

    out = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])

    return out

def GET_C_PERMUTATIONS():

    out = np.array([[0,1,2],[1,2,0],[2,0,1]])

    return out

def NORMALIZED_CAPACITANCE(alpha,n):
    #finds the normalized capacitance based off of alpha
    #alpha: array of alpha values for each direction - float, shape (3,0)
    #alpha_l: the alpha of the equivalent unit cell length - float, shape (1,0)
    #C normalized capacitance for a given scatter sub matrix - float, shape (3,3)

    #define equivalent cubic cell parameter to propagation delay ratio
    alpha_l = 0.5*2/n

    #define constant A
    A = ( alpha[0]*alpha[1]*alpha[2] )**2 * ( 4/(alpha_l)**2 - np.sum(alpha**-2) )

    #check parameters
    parameter_check = A**2 - (alpha[0]*alpha[1]*alpha[2]*alpha_l)**2
    if parameter_check <= 0:
        print('WARNING: parameter check for scatter sub matrix failed. parameter check value: ', parameter_check)
    
    #define constant B
    B = A - np.sqrt( A**2 - (alpha[0]*alpha[1]*alpha[2])**2 )

    #normalized capacitance
    C = np.zeros((3,3),dtype = float)
    perm = GET_C_PERMUTATIONS()

    for l in range(3):
        
        i = perm[l,:][0]
        j = perm[l,:][1]
        k = perm[l,:][2]

        C[i,j] = ( 2 * alpha[j]**2 * alpha[k]**2 + B ) / ( 2 * alpha[i]**2 * alpha[k]**2 * ((2*alpha[j]/alpha_l)**2 - 1) )

    C[0,2] = 1 - C[1,2]
    C[1,0] = 1 - C[2,0]
    C[2,1] = 1 - C[0,1]

    return C

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

def SCATTER_2_ELECTRIC_NODES(scatter_field_tensor,n_c,n,alpha):
    #produces the electric field values at each node given the scatter components

    #get spatial parameters
    n_i,n_j,n_k,_ = np.shape(n)

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

# ---------------------- Source Functions -------------------------#

def TIME_SOURCE(polarization,n_c,n_t,wavelength,fwhm,n,alpha,location,injection_axis,injection_direction):
    #produces the time source for a Gaussian wave packet traveling in one direction
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #n_c: number of scatter components - int, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #wavelength: number of points used to define one wavelength - int, shape(1,)
    #fwhm: full width at half maximum in the time domain - int, shape(1,)
    #n: refractive index distribution for each spatial location (i,j,k) - np.array float, shape(n_i,n_j,n_k,1)
    #alpha: ratio of space/time steps in units of c - np.array float, shape(n_i,n_j,n_k,3)
    #location: gives the location (i,j,k) of the source - tuple int, shape (3,)
    #injection_axis: gives the axis of injection - int, shape(1,)
    #injection_direction: gives the direction (positive or negative) the source travels on the injection axis - int, shape(1,)
    #
    #time source: the source in time - np.array float, shape(n_t,n_c)

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
    sigma = fwhm / 2.35482
    #electric field time source multiplicatvely scaled by the time step (E * del_t)
    electric_field_time_source = np.sin(-1*omega*t)*np.exp(-(t-sigma*2.5)**2/(2*sigma**2))

    #voltage polarized in the polarization direction
    voltage = -electric_field_time_source*alpha[location_i,location_j,location_k,polarization]

    #initilize a time source
    time_source = np.zeros((n_t,n_c),dtype = float)

    #build time source for each component
    for t in range(n_t):
        for direction in range(3):
            for polarity in range(2):

                #non-equal direction/polarization components only
                if not direction == polarization:

                    if direction == injection_axis:
                        if polarity == injection_direction:
                            #get the scatter component for a given polarization,direction and polarity
                            c = INDEX_2_COMPONENT(direction,polarization,polarity)

                            time_source[t,c] = voltage[t]/(4*normalized_capacitance[direction,polarization])

    return time_source

def POINT_SOURCE(location,time_source,n_x,n_y,n_z):
    #produces a dipole point source at a given location and polarization
    #location: gives the location (i,j,k) of the source - tuple int, shape (3,)
    #time_source: the source in time - np.array float, shape(n_t,n_c)
    #n_x: number of space steps along first axis - int, shape(1,)
    #n_y: number of space steps along second axis - int, shape(1,)
    #n_z: number of space steps along third axis - int, shape(1,)
    #
    #space_time_source: source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)

    #get spatial information
    n_t,n_c = np.shape(time_source)
    i_location = location[0]
    j_location = location[1]
    k_location = location[2]

    #initilize sources
    space_time_source = np.zeros((n_x,n_y,n_z,n_c,n_t),dtype = float)

    #build space-time source
    for t in range(n_t):

        space_time_source[i_location,j_location,k_location,:,t] = time_source[t,:]

    return space_time_source

def MODE_SHAPE(fwhm,n_m,center):
    # creates the transverse shape of the mode
    # fwhm: the full width at half maximum of the mode
    # n_m: number of points used to define the mode
    # center: center of the mode
    #
    # mode_shape: shape of mode along mode axis - np.array float, shape (n_m,)

    #mode axis
    x = np.arange(0,n_m,1)
    #standard deviation
    sigma = fwhm / 2.35482
    #create guassian mode shape
    mode_shape = np.exp(-(x-center)**2/(2*sigma**2))

    return mode_shape

def MODE_SOURCE(space_time_source,mode_shape,mode_axis):
    #produces a mode source
    # space_time_source: source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)
    # mode_shape: shape of mode along mode axis - np.array float, shape (n_m,)
    # mode_axis: axis on which the mode exists - int, shape(1,)
    #
    # space_time_source_mode: mode source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)


    if mode_axis == 0:

            space_time_source_mode = np.einsum('i,ijk->ijk',mode_shape,space_time_source)

    elif mode_axis == 1:

        space_time_source_mode = np.einsum('j,ijk->ijk',mode_shape,space_time_source)

    else:
        print('WARNING: injection_axis value is not recognized by LINE_SOURCE function')

    return space_time_source_mode

def LINE_SOURCE(location,injection_axis,time_source,n_x,n_y,n_z):
    #produces a plane wave line source
    # location: a point on the line source
    # injection_axis: the axis direction in which the plane wave travels
    # time_source: the source in time - np.array float, shape(n_t,n_c)
    # n_x: number of space steps along first axis - int, shape(1,)
    # n_y: number of space steps along second axis - int, shape(1,)
    # n_z: number of space steps along third axis - int, shape(1,)
    #
    #space_time_source: source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)
   
    #get time/component information
    n_t,n_c = np.shape(time_source)

    #initilize sources
    space_time_source = np.zeros((n_x,n_y,n_z,n_c,n_t),dtype = float)

    #get postion of source
    i_location = location[0]
    j_location = location[1]
    k_location = location[2]

    # find x and y axis index values corresponding to the line source location
    if injection_axis == 0:

        #build space-time source
        for t in range(n_t):

            space_time_source[i_location,:,k_location,:,t] = time_source[t,:]


    elif injection_axis == 1:

        #build space-time source
        for t in range(n_t):

            space_time_source[:,j_location,k_location,:,t] = time_source[t,:]

    else:
        print('WARNING: injection_axis value is not recognized by LINE_SOURCE function')

    return space_time_source

def SOURCE(polarization,n_c,n_t,wavelength,fwhm,n,alpha,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis):
    #calculates the source 

    time_source = TIME_SOURCE(polarization,n_c,n_t,wavelength,fwhm,n,alpha,location,injection_axis,injection_direction)

    #get position parameters
    n_x,n_y,n_z,_ = np.shape(alpha)

    if source_type == 'Line':

        space_time_source = LINE_SOURCE(location,injection_axis,time_source,n_x,n_y,n_z)

        return space_time_source , time_source

    elif source_type == 'Mode':

        space_time_source = LINE_SOURCE(location,injection_axis,time_source,n_x,n_y,n_z)

        mode_shape = MODE_SHAPE(fwhm_mode,n_m,center_mode)

        space_time_source_mode = MODE_SOURCE(space_time_source,mode_shape,mode_axis)

        return space_time_source_mode , time_source

    else: 

        print('WARNING: source type not recognized')
        
        return 0

