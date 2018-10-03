import tensorflow as tf
import numpy as np
import pickle

data_type = np.float32

# ---------------------- Tensors and Tensor Operations ------------------#
def CONSTANT_MATRICES():
    #produces the matrices of constants used to construct the scatter tensor

    a = 0
    c = -1
    b = 1
    d = 0

    ones_matrix =  np.array([[a,b,d,0,0,0,0,0,b,0,-d,c],
            [b,a,0,0,0,d,0,0,c,-d,0,b],
            [d,0,a,b,0,0,0,b,0,0,c,-d],
            [0,0,b,a,d,0,-d,c,0,0,b,0],
            [0,0,0,d,a,b,c,-d,0,b,0,0],
            [0,d,0,0,b,a,b,0,-d,c,0,0],
            [0,0,0,-d,c,b,a,d,0,b,0,0],
            [0,0,b,c,-d,0,d,a,0,0,b,0],
            [b,c,0,0,0,-d,0,0,a,d,0,b],
            [0,-d,0,0,b,c,b,0,d,a,0,0],
            [-d,0,c,b,0,0,0,b,0,0,a,d],
            [c,b,-d,0,0,0,0,0,b,0,d,a]])

    a = -1
    c = -1
    b = -1
    d = 0

    d1_matrix =  np.array([[a,b,d,0,0,0,0,0,b,0,-d,c],
            [b,a,0,0,0,d,0,0,c,-d,0,b],
            [d,0,a,b,0,0,0,b,0,0,c,-d],
            [0,0,b,a,d,0,-d,c,0,0,b,0],
            [0,0,0,d,a,b,c,-d,0,b,0,0],
            [0,d,0,0,b,a,b,0,-d,c,0,0],
            [0,0,0,-d,c,b,a,d,0,b,0,0],
            [0,0,b,c,-d,0,d,a,0,0,b,0],
            [b,c,0,0,0,-d,0,0,a,d,0,b],
            [0,-d,0,0,b,c,b,0,d,a,0,0],
            [-d,0,c,b,0,0,0,b,0,0,a,d],
            [c,b,-d,0,0,0,0,0,b,0,d,a]])

    a = -1
    c = 1
    b = 0
    d = 1

    d2_matrix =  np.array([[a,b,d,0,0,0,0,0,b,0,-d,c],
            [b,a,0,0,0,d,0,0,c,-d,0,b],
            [d,0,a,b,0,0,0,b,0,0,c,-d],
            [0,0,b,a,d,0,-d,c,0,0,b,0],
            [0,0,0,d,a,b,c,-d,0,b,0,0],
            [0,d,0,0,b,a,b,0,-d,c,0,0],
            [0,0,0,-d,c,b,a,d,0,b,0,0],
            [0,0,b,c,-d,0,d,a,0,0,b,0],
            [b,c,0,0,0,-d,0,0,a,d,0,b],
            [0,-d,0,0,b,c,b,0,d,a,0,0],
            [-d,0,c,b,0,0,0,b,0,0,a,d],
            [c,b,-d,0,0,0,0,0,b,0,d,a]])

    return d1_matrix, d2_matrix, ones_matrix

def IDENTITY_TENSOR(n_x,n_y,n_z,n_c):
    # produces a tensor with an identity matrix at location index
    # n_x: # of spatial steps in x - int, shape(1,)
    # n_y: # of spatial steps in y - int, shape(1,)
    # n_z: # of spatial steps in z - int, shape(1,)
    # n_c: # of field components - int, shape(1,)

    identity_tensor = tf.eye(n_c , batch_shape = [n_x,n_y,n_z])

    return identity_tensor
def MESH_INDEX(c):
    #get the mesh index for a given field component c
    #c: field component - int shape(1,)

    #table relating c index to mesh direction (0-x, 1-y, 2-z)
    table = np.array( [[0,1,0,2] , [1,2,0,1] , [2,0,1,2] , [3,2,1,0] , [4,1,2,0] , [5,0,2,1] , [6,1,2,0] , [7,2,1,0], [8,2,0,1], [9,0,2,1], [10,0,1,2], [11,1,0,2]] )

    return table[c,:]

def CONSTANT_TENSORS(mesh,n_c,layers):
    #produces the constant tensors used to construct the scatter tensor
    #mesh: contains the alpha mesh variables (del_space/del_time/c0) in each direction - numpy.array shape(n_x,n_y,n_z,3)
    #n_c: number of field components

    #produce matrices for filter the d1 and d2 matrix along with the ones matrix
    d1_matrix , d2_matrix , ones_matrix = CONSTANT_MATRICES()
    #spatial parameters
    n_x,n_y,n_z,_ = np.shape(mesh)

    #initilize constant tensors
    a1 = np.zeros((layers,n_x,n_y,n_z,n_c), dtype = data_type)
    a21 = np.zeros((layers,n_x,n_y,n_z,n_c), dtype = data_type)
    a22 = np.zeros((layers,n_x,n_y,n_z,n_c), dtype = data_type)
    a3 = np.zeros((layers,n_x,n_y,n_z,n_c), dtype = data_type)
    a4 = np.zeros((layers,n_x,n_y,n_z,n_c), dtype = data_type)

    f1 = np.zeros((layers,n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    f2 = np.zeros((layers,n_x,n_y,n_z,n_c,n_c), dtype = data_type)

    O = np.zeros((layers,n_x,n_y,n_z,n_c,n_c), dtype = data_type)

    #build constant tensors based of relationship between mesh index and field component c
    for t in range(layers):
        for x in range(n_x):
            for y in range(n_y):
                for z in range(n_z):

                    #build filter and ones tensors
                    f1[t,x,y,z,:,:] = d1_matrix
                    f2[t,x,y,z,:,:] = d2_matrix
                    O[t,x,y,z,:,:] = ones_matrix

                    for c in range(n_c):

                        _,i,j,k = MESH_INDEX(c)
                        a1[t,x,y,z,c] = 2 * mesh[x,y,z,j]**2 * mesh[x,y,z,k]**2 
                        a21[t,x,y,z,c] = 2 * mesh[x,y,z,i]**2 * mesh[x,y,z,k]**2 
                        a22[t,x,y,z,c] = 2 * mesh[x,y,z,i]**2 * mesh[x,y,z,j]**2 
                        a3[t,x,y,z,c] = ( mesh[x,y,z,i] * mesh[x,y,z,j] * mesh[x,y,z,k] ) ** 2
                        a4[t,x,y,z,c] = mesh[x,y,z,i] ** -2 + mesh[x,y,z,j] ** -2 + mesh[x,y,z,k] ** -2

    return a1,a21,a22,a3,a4,f1,f2,O
# ---------------------- Scatter Tensor Generation ----------------------#

def SCATTER(weight_tens,a1,a21,a22,a3,a4,f1,f2,O,n_c,n_x,n_y,n_z,n_w,layers):
    #produces the scatter tensor to operate on the field tensor
    #weight_tens: weight tensor - shape (n_x,n_y,n_z,n_w)

    #assign a weight to each field component
    w = tf.tile(weight_tens,[1,1,1,1,n_c//n_w])

    #calculate protions of the d tensors
    A = tf.multiply(a3,tf.reciprocal(w) - a4)
    B = A + tf.sqrt( tf.multiply(A,A) - tf.multiply(a3,w) )
    C1 = tf.reciprocal( tf.multiply( tf.multiply(2.0,a3),tf.reciprocal(w) ) - a21)
    C2 = tf.reciprocal( tf.multiply( tf.multiply(2.0,a3),tf.reciprocal(w) ) - a22)

    #combine to produce the d tensors
    d1 = tf.tile( tf.reshape( tf.multiply( a1+B , C1 ) , [layers,n_x,n_y,n_z,n_c,1] ) , [1,1,1,1,1,n_c] )
    d2 = tf.tile( tf.reshape( tf.multiply( a1+B , C2 ) , [layers,n_x,n_y,n_z,n_c,1] ) , [1,1,1,1,1,n_c] )

    #produce the scatter tensor
    S = tf.multiply(f1,d1) + tf.multiply(f2,d2) + O

    return S



