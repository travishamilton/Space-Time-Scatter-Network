import numpy as np

dtype = np.float32

def DISTANCE(x1,x2):
    #determines the distance between two points in 3D space
    #x1: point one - np.array int, shape(1,1,1)
    #x2: point two - np.array int, shape(1,1,1)
    #D: distance between point x1 and x2

    D = np.sqrt( (x2[0]-x1[0])**2 + (x2[1]-x1[1])**2 + (x2[2]-x1[2])**2 )

    return D

def MAKE_CYLINDER(radius,center,n_background,n_cylinder):
    #creates a refractive index distribution of a 2 1/2 D slice of a cylinder along a specified axis
    #radius: the radius of the cylinder - int, shape (1,)
    #center: center of cylinder - np.array int, shape (1,1,1)
    #n_background: background refractive index  - np.array float, shape (n_i,n_j,n_k,1), where n_i,n_j or n_k == 0
    #n_cylinder: cylinder's rafractive index - int, shape(1,)
    #n: refractive index distribution of cylinder in n_background

    #initilize n to background
    n = n_background

    #get simulation size parameters
    n_i,n_j,n_k,_ = np.shape(n_background)

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                #get distance from center
                d = DISTANCE(np.array([i,j,k],dtype = int),center)
                
                if d <= radius:
                    n[i,j,k,0] = n_cylinder
    return n

def REFRACTIVE_INDEX(n_i,n_j,n_k,distribution_type,mask_start,mask_stop,initial_weight):
    #produces the refractive index values for each postion in space
    #n_i: number of spatial steps in the ith dimension - int, shape (1,)
    #n_j: number of spatial steps in the jth dimension - int, shape (1,)
    #n_k: number of spatial steps in the kth dimension - int, shape (1,)
    #n: refractive index - np.array float, shape (n_i,n_j,n_k,1)

    #set n to free space
    n = np.ones((n_i,n_j,n_k,1))

    #select distribution type
    if distribution_type == 'cylinder':
        radius = 15
        n_cylinder = 1.5
        center = np.array([n_i//2,n_j//2,n_k//2],dtype = int)
        n = MAKE_CYLINDER(radius,center,n,n_cylinder)

    if distribution_type == 'waveguide':
        half = 15
        n_waveguide = 2
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    if j > n_j//2 - half and j < n_j//2 + half:
                        n[:,j,k,0] = n_waveguide

    if distribution_type == 'mask':
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    if i >= mask_start[0] and j >= mask_start[1] and k >= mask_start[2] and i <= mask_stop[0] and j <= mask_stop[1] and k <= mask_stop[2]:
                        n[i,j,k,0] = 1./np.sqrt(initial_weight)
    return n

def REFRACTIVE_INDEX_DISPERSION(n_i,n_j,n_k,n_f,distribution_type,mask_start,mask_stop,initial_weight):
    #produces the refractive index values for each postion in space
    #n_i: number of spatial steps in the ith dimension - int, shape (1,)
    #n_j: number of spatial steps in the jth dimension - int, shape (1,)
    #n_k: number of spatial steps in the kth dimension - int, shape (1,)
    #n: refractive index - np.array float, shape (n_i,n_j,n_k,1)

    #set n to free space
    n = np.ones((n_i,n_j,n_k,1))
    inf_x = np.zeros((n_i,n_j,n_k,n_f//2))
    w_0 = np.zeros((n_i,n_j,n_k,n_f//2))
    damp = np.zeros((n_i,n_j,n_k,n_f//2))
    del_x = np.zeros((n_i,n_j,n_k,n_f//2))
    absorption_coeff = np.zeros((n_i,n_j,n_k,12),dtype = np.float32)

    #select distribution type
    if distribution_type == 'cylinder':
        radius = 15
        n_cylinder = 1.5
        center = np.array([n_i//2,n_j//2,n_k//2],dtype = int)
        n = MAKE_CYLINDER(radius,center,n,n_cylinder)

    if distribution_type == 'waveguide':
        half = 15
        n_waveguide = 2
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    if j > n_j//2 - half and j < n_j//2 + half:
                        n[:,j,k,0] = n_waveguide


    if distribution_type == 'mask':
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    inf_x[i,j,k,:] = np.array([0.5,0.5,0.5],dtype = np.float32)
                    w_0[i,j,k,:] = 2*np.pi*20*10**9*np.array([1,1,1],dtype = np.float32)
                    damp[i,j,k,:] = 0.0*w_0[i,j,k,:]
                    del_x = np.array([1.5,1.5,1.5],dtype = np.float32)

                    if i >= mask_start[0] and j >= mask_start[1] and k >= mask_start[2] and i <= mask_stop[0] and j <= mask_stop[1] and k <= mask_stop[2]:
                        n[i,j,k,0] = 1./np.sqrt(initial_weight)
                        absorption_coeff[i,j,k,:] = 0*np.array([0,0,0,0,1,1,1,0,0,1,0,0],dtype = np.float32)

    return n , inf_x,w_0,damp,del_x
                
def MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat):
    #produces the parameters for dispersive medium
    #n_x: number of spatial steps in the 0th dimension - int, shape (1,)
    #n_y: number of spatial steps in the 1st dimension - int, shape (1,)
    #n_z: number of spatial steps in the 2nd dimension - int, shape (1,)
    #n_r: number of Lorentz resonances within the material - int, shape (1,)
    #
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # w_0: Lorentz resonance frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # damp: Lorentz damping frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # del_x: change in susceptibility tensor (unitless) - np.constant, shape(n_x,n_y,n_z,n_r)

    #set values to free space
    inf_x = np.zeros((n_x,n_y,n_z),dtype = dtype)
    w_0 = np.zeros((n_x,n_y,n_z,n_r),dtype = dtype)
    damp = np.zeros((n_x,n_y,n_z,n_r),dtype = dtype)
    del_x = np.zeros((n_x,n_y,n_z,n_r),dtype = dtype)

    #assign same values across all space
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                inf_x[i,j,k] = inf_x_mat
                w_0[i,j,k,:] = w_0_mat
                damp[i,j,k,:] = damp_mat
                del_x[i,j,k,:] = del_x_mat

    return inf_x,w_0,damp,del_x

def REFLECTION():
    #produces the reflections at each wall
    #reflection: the reflection for a wave along a certian direction with a given polarization and polarity - np.array float, shape (3,3)

    reflection = np.zeros((3,3,2), dtype = float)

    reflection[0,1,0] = 0
    reflection[0,1,1] = 0
    reflection[0,2,0] = 0
    reflection[0,2,1] = 0

    reflection[1,0,0] = 1
    reflection[1,0,1] = 1
    reflection[1,2,0] = 1
    reflection[1,2,1] = 1

    reflection[2,0,0] = -1
    reflection[2,0,1] = -1
    reflection[2,1,0] = -1
    reflection[2,1,1] = -1
    

    return reflection
    