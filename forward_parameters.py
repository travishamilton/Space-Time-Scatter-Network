import numpy as np

def ALPHA(n_i,n_j,n_k):
    #produces the spatial step to temportal step ratios called alpha in three dimensions for all spaces in time
    #n_i: number of spatial steps in the ith dimension - int, shape (1,)
    #n_j: number of spatial steps in the jth dimension - int, shape (1,)
    #n_k: number of spatial steps in the kth dimension - int, shape (1,)
    #alpha: the alpha variable mentioned above - np.array float, shape (n_i,n_j,n_k,3)

    #set dx/dt = c
    alpha = np.ones((n_i,n_j,n_k,3),dtype = float)

    #adjust as needed
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                alpha[i,j,k,:] = np.array([2.0,2.0,2.0])

    #return alpha
    return alpha
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

def REFRACTIVE_INDEX(n_i,n_j,n_k,distribution_type):
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

    if distribution_type == 3:
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    if j > 3*n_j//8 and j < 5*n_j//8:
                        n[i,j,k,0] = 1.3
    return n

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
    