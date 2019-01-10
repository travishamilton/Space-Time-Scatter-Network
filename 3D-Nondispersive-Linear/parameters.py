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

def NL_MULTIPLE_DISPERSION_PARAMETERS(n_x,n_y,n_z,n_r,inf_x_mat,w_0_mat,damp_mat,del_x_mat,x_nl_mat,mask_start,mask_end,n_m):
    # produces the parameters for dispersive and non-linear medium
    # n_x: number of spatial steps in the 0th dimension - int, shape (1,)
    # n_y: number of spatial steps in the 1st dimension - int, shape (1,)
    # n_z: number of spatial steps in the 2nd dimension - int, shape (1,)
    # n_r: number of Lorentz resonances within the material - int, shape (1,)
    # mask_start: smallest coordinates of the masked region - tuple int, shape(3,)
    # mask_end: largest coordinates of the masked region - tuple int, shape(3,)
    # n_m: number of materials present in simulation - int. shape(1,)
    #
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # w_0: Lorentz resonance frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # damp: Lorentz damping frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # del_x: change in susceptibility tensor (unitless) - np.constant, shape(n_x,n_y,n_z,n_r,n_m)

    #set values to free space
    inf_x = np.zeros((n_x,n_y,n_z),dtype = dtype)
    w_0 = np.zeros((n_x,n_y,n_z,n_r),dtype = dtype)
    damp = np.zeros((n_x,n_y,n_z,n_r),dtype = dtype)
    del_x = np.zeros((n_x,n_y,n_z,n_r),dtype = dtype)
    x_nl = np.zeros((n_x,n_y,n_z),dtype = dtype)

    #check to make sure mask makes sense
    if mask_start[0] <= mask_end[0] and mask_start[1] <= mask_end[1] and mask_start[2] <= mask_end[2]:

        #assign same values across all space
        for i in range(n_x):
            if i >= mask_start[0] and i <= mask_end[0]:
                for j in range(n_y):
                    if j >= mask_start[1] and j <= mask_end[1]:
                        for k in range(n_z):
                            if k >= mask_start[2] and k <= mask_end[2]:
                                inf_x[i,j,k] = inf_x_mat
                                w_0[i,j,k,:] = w_0_mat
                                damp[i,j,k,:] = damp_mat
                                x_nl[i,j,k] = x_nl_mat
                                del_x[i,j,k,:] = del_x_mat
        

    else:
	    raise ValueError("Starting index must be smaller than or equal to ending index.")

    return inf_x,w_0,damp,del_x,x_nl
    