import numpy as np

data_type = np.float32

# ---------------------- Source Functions -------------------------#
def TIME_SOURCE_LUMERICAL(polarization,n_f,del_t,n_t,del_l,e_time_source):
    #produces the time source for a given Guassian wave packet source
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #n_f: number of field components - int, shape(1,)
    #del_t: time step - np.float32, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #del_l: space step in all three directions - np.float32, shape(1,)
    #e_time_source: the electric field time source - float32, shape(n_t,)
    #
    #v_f: the free source fields for all time - np.array float, shape(n_t,n_f)

    #impedance of free space
    eta_0 = 119.9169832*np.pi
    c0 = 2.99792458e8

    #time
    t = np.arange(0,n_t*del_t,del_t,dtype = float)

    #current density required to produce a plane wave with the given electric field time source
    current_density = -2*e_time_source/eta_0

    #initilize the free-source fields
    v_f = np.zeros((n_t,n_f),dtype = float)

    #build free-source fields for each componet for all time (space not included yet)
    for t_index in range(n_t):

        v_f[t_index,polarization] = del_l**2 * eta_0 * current_density[t_index]

    return v_f

def TIME_SOURCE_E(polarization,n_f,del_t,n_t,wavelength,fwhm,del_l,location,injection_axis,injection_direction):
    #produces the electric free source for a Gaussian wave packet traveling in one direction
    #polarization: gives the polarization direction of the point source - int, shape(1,)
    #n_f: number of field components - int, shape(1,)
    #del_t: time step - np.float32, shape(1,)
    #n_t: number of time steps - int, shape(1,)
    #wavelength: center wavelenght of wave packket - float32 , shape(1,)
    #fwhm: full width at half maximum in the time domain - int, shape(1,)
    #del_l: space step in all three directions - np.float32, shape(1,)
    #location: gives the location (i,j,k) of the source - tuple int, shape (3,)
    #injection_axis: gives the axis of injection - int, shape(1,)
    #injection_direction: gives the direction (positive or negative) the source travels on the injection axis - int, shape(1,)
    #
    #time source: the free voltage sources in time (V) - np.array float, shape(n_t,n_f)

    #impedance of free space and speed of light
    eta_0 = 119.9169832*np.pi
    c0 = 2.99792458e8

    #time
    t = np.arange(0,n_t*del_t,del_t,dtype = data_type)
    t = np.linspace(0,(n_t-1)*del_t,n_t)
    #angular frequency
    omega = 2*np.pi*c0/wavelength
    #standard deviation
    sigma = fwhm / 2.35482

    #free electric current density guassian wavepacket
    current_density = np.sin(-omega*t)*np.exp(-(t-2*sigma)**2/(2*sigma**2)) / (-del_l * eta_0 * 0.5 )

    #initilize the free source
    time_source = np.zeros((n_t,n_f),dtype = data_type)

    #build free source for each electric component
    for t_index in range(n_t):

        time_source[t_index,polarization] = -del_l**2 * eta_0 * current_density[t_index]

    return time_source , current_density

# def POINT_SOURCE(location,time_source,n_x,n_y,n_z):
#     #produces a dipole point source at a given location and polarization
#     #location: gives the location (i,j,k) of the source - tuple int, shape (3,)
#     #time_source: the source in time - np.array float, shape(n_t,n_f)
#     #n_x: number of space steps along first axis - int, shape(1,)
#     #n_y: number of space steps along second axis - int, shape(1,)
#     #n_z: number of space steps along third axis - int, shape(1,)
#     #
#     #space_time_source: source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)

#     #get spatial information
#     n_t,n_f = np.shape(time_source)
#     i_location = location[0]
#     j_location = location[1]
#     k_location = location[2]

#     #initilize sources
#     space_time_source = np.zeros((n_x,n_y,n_z,n_f,n_t),dtype = float)

#     #build space-time source
#     for t in range(n_t):

#         space_time_source[i_location,j_location,k_location,:,t] = time_source[t,:]

#     return space_time_source

# def MODE_SHAPE(fwhm,n_m,center):
#     # creates the transverse shape of the mode
#     # fwhm: the full width at half maximum of the mode
#     # n_m: number of points used to define the mode
#     # center: center of the mode
#     #
#     # mode_shape: shape of mode along mode axis - np.array float, shape (n_m,)

#     #mode axis
#     x = np.arange(0,n_m,1)
#     #standard deviation
#     sigma = fwhm / 2.35482
#     #create guassian mode shape
#     mode_shape = np.exp(-(x-center)**2/(2*sigma**2))

#     return mode_shape

# def MODE_SOURCE(space_time_source,mode_shape,mode_axis):
#     #produces a mode source
#     # space_time_source: source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)
#     # mode_shape: shape of mode along mode axis - np.array float, shape (n_m,)
#     # mode_axis: axis on which the mode exists - int, shape(1,)
#     #
#     # space_time_source_mode: mode source for all space and time - np.array float, shape (n_x,n_y,n_z,n_c,n_t)


#     if mode_axis == 0:

#             space_time_source_mode = np.einsum('i,ijk->ijk',mode_shape,space_time_source)

#     elif mode_axis == 1:

#         space_time_source_mode = np.einsum('j,ijk->ijk',mode_shape,space_time_source)

#     else:
#         print('WARNING: injection_axis value is not recognized by LINE_SOURCE function')

#     return space_time_source_mode

def LINE_SOURCE_E(location,injection_axis,time_source,n_x,n_y,n_z):
    # produces a plane wave line source with electric field components only
    # location: a point on the line source
    # injection_axis: the axis direction in which the plane wave travels
    # time_source: the free voltage sources in time - np.array float, shape(n_t,n_c)
    # n_x: number of space steps along first axis - int, shape(1,)
    # n_y: number of space steps along second axis - int, shape(1,)
    # n_z: number of space steps along third axis - int, shape(1,)
    #
    #space_time_source: source for all space and time - np.array float, shape (n_x,n_y,n_z,n_f,n_t)
   
    #get time/component information
    n_t,n_f = np.shape(time_source)

    #initilize sources
    space_time_source = np.zeros((n_x,n_y,n_z,n_f,n_t),dtype = data_type)

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

def SOURCE(n_f,n_t,del_t,del_l,n_x,n_y,n_z,polarization,wavelength,fwhm,location,injection_axis,injection_direction,source_type,fwhm_mode,n_m,center_mode,mode_axis):
    #calculates the source 
    #
    # space_time_source: the voltage and current free sources injected into the simulation over all space and time (V and A) - shape(n_x,n_y,n_z,n_f,n_t)
    # time_source: the voltage and current free sources injected into the simulation over all time (V and A) - shape(n_t,n_f)
    # current_density: the free source current density (A/m^2) - shape(n_t)

    time_source , current_density = TIME_SOURCE_E(polarization,n_f,del_t,n_t,wavelength,fwhm,del_l,location,injection_axis,injection_direction)
    
    if source_type == 'Line':

        space_time_source = LINE_SOURCE_E(location,injection_axis,time_source,n_x,n_y,n_z)

        return space_time_source , time_source , current_density

    # elif source_type == 'Mode':

    #     space_time_source = LINE_SOURCE_E(location,injection_axis,time_source,n_x,n_y,n_z) 

    #     mode_shape = MODE_SHAPE(fwhm_mode,n_m,center_mode)

    #     space_time_source_mode = MODE_SOURCE(space_time_source,mode_shape,mode_axis)

    #     return space_time_source_mode , time_source

    else: 

        print('WARNING: source type not recognized')
        
        return 0


