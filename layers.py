import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

from scipy.linalg import block_diag
from numpy.linalg import matrix_power
from parameters import REFLECTION
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

############################################################################################
##                                                                                        ##
##                                                                                        ##
##                          layer functions of STSN forward model                         ##
##                                                                                        ##
##                                                                                        ##
############################################################################################
##Notes: These layers define the strucutre of STSN. STSN space is defined using 4 indexes:
#   i - x-axis in real space
#   j - y-axis in real space
#   k - z-axis in real space
#   c - field components in real space
#       c = 0 - x polarized , y traveling , - polarity
#       c = 1 - x polarized , z traveling , - polarity
#       c = 2 - y polarized , x traveling , - polarity
#       c = 3 - y polarized , z traveling , - polarity
#       c = 4 - z polarized , y traveling , - polarity
#       c = 5 - z polarized , x traveling , - polarity
#       c = 6 - z polarized , y traveling , + polarity
#       c = 7 - y polarized , z traveling , + polarity
#       c = 8 - x polarized , z traveling , + polarity
#       c = 9 - z polarized , x traveling , + polarity
#       c = 10 - y polarized , x traveling , + polarity
#       c = 11 - x polarized , y traveling , + polarity


def GET_PERMUTATIONS():

    out = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])

    return out

def GET_C_PERMUTATIONS():

    out = np.array([[0,1,2],[1,2,0],[2,0,1]])

    return out

def COMPONENT_2_INDEX(c):
    if c == 0 or c == 11:
        i = 1
        j = 0
    if c == 1 or c == 8:
        i = 2
        j = 0
    if c == 2 or c == 10:
        i = 0
        j = 1
    if c == 3 or c == 7:
        i = 2
        j = 1
    if c == 4 or c == 6:
        i = 1
        j = 2
    if c == 5 or c == 9:
        i = 0
        j = 2
    if c <= 5:
        s = 0
    else:
        s = 1

    #direction - i , polarization - j and polarity - s of the scatter component c
    return i,j,s

def INDEX_2_COMPONENT(i,j,s):
    #direction - i , polarization - j and polarity - s of the scatter component c
    if i == 1 and j == 0 and s == 0:
        c = 0
    if i == 1 and j == 0 and s == 1:
        c = 11
    if i == 2 and j == 0 and s == 0:
        c = 1
    if i == 2 and j == 0 and s == 1:
        c = 8
    if i == 0 and j == 1 and s == 0:
        c = 2
    if i == 0 and j == 1 and s == 1:
        c = 10
    if i == 2 and j == 1 and s == 0:
        c = 3
    if i == 2 and j == 1 and s == 1:
        c = 7
    if i == 1 and j == 2 and s == 0:
        c = 4
    if i == 1 and j == 2 and s == 1:
        c = 6
    if i == 0 and j == 2 and s == 0:
        c = 5
    if i == 0 and j == 2 and s == 1:
        c = 9

    return c

def TRANSFER_INDEX_CHECK(transfer_index,n_i,n_j,n_k):
    #checks transfer index to see if it is out of bounds
    #transfer index: the corrdinates the field will be transfered to - tuple int, shape (4,)

    out_of_bounds = 0

    if transfer_index[0] >= n_i or transfer_index[0] < 0:
        out_of_bounds = 1
    elif transfer_index[1] >= n_j or transfer_index[1] < 0:
        out_of_bounds = 1
    elif transfer_index[2] >= n_k or transfer_index[2] < 0:
        out_of_bounds = 1

    return out_of_bounds

def TRANSFER_OPERATIONS(index):
    #transfer rules of the STSN model
    #index: tuple containing the index of the ith, jth, kth dimension and the cth field component in that order - int, shape (4,)
    #transfer_index: same values and shape as index but updated to follow the transfer operation rules - int, shape (4,)

    #retrive individual index values
    i = index[0]
    j = index[1]
    k = index[2]
    c = index[3]

    #first field component
    if c == 0:
        c_transfered = 11
        i_transfered = i
        j_transfered = j-1
        k_transfered = k
    #scond field component
    elif c == 1:
        c_transfered = 8
        i_transfered = i
        j_transfered = j
        k_transfered = k-1
    #third field component
    elif c == 2:
        c_transfered = 10
        i_transfered = i-1
        j_transfered = j
        k_transfered = k
    #fourth field component
    elif c == 3:
        c_transfered = 7
        i_transfered = i
        j_transfered = j
        k_transfered = k-1
    #fifth field component
    elif c == 4:
        c_transfered = 6
        i_transfered = i
        j_transfered = j-1
        k_transfered = k
    #sixth field component
    elif c == 5:
        c_transfered = 9
        i_transfered = i-1
        j_transfered = j
        k_transfered = k
    #seventh field component
    elif c == 6:
        c_transfered = 4
        i_transfered = i
        j_transfered = j+1
        k_transfered = k
    #eight field component
    elif c == 7:
        c_transfered = 3
        i_transfered = i
        j_transfered = j
        k_transfered = k+1
    #nineth field component
    elif c == 8:
        c_transfered = 1
        i_transfered = i
        j_transfered = j
        k_transfered = k+1
    #tenth field component
    elif c == 9:
        c_transfered = 5
        i_transfered = i+1
        j_transfered = j
        k_transfered = k
    #eleventh field component
    elif c == 10:
        c_transfered = 2
        i_transfered = i+1
        j_transfered = j
        k_transfered = k
    #twelvth field component
    elif c == 11:
        c_transfered = 0
        i_transfered = i
        j_transfered = j+1
        k_transfered = k
    #unrecognized field component
    else:
        print('c value not recognized in TRANSFER_OPERATIONS') 
        c_transfered = c
        i_transfered = i
        j_transfered = j
        k_transfered = k
    
    #updated index values
    transfer_index = (i_transfered,j_transfered,k_transfered,c_transfered)
    
    
    return transfer_index

def VECTORIZE(field_tensor):
    #vectorizes the field tensor into a single arry using C - like indexing (last index changes fastest)
    #field_tensor: 4 dimensional tensor holding field data at position (i,j,k) with c field components  - float, shape (i,j,k,c)
    #field_array: field_tensor in an array sequentially listing each c long field components for each position (i,j,k) float, shape (i*j*k*c,)
    
    field_array = np.reshape(field_tensor,np.size(field_tensor))

    return field_array

def TENSOR_INDEX_TO_VECTOR_INDEX(tensor_index,n_i,n_j,n_k,n_c):
    #calculates the index a value in a tensor would have after vecotrization
    #tensor_index: the value's tensor index - tuple int, shape (1,1,1,1)

    c = tensor_index[3]
    k = tensor_index[2]
    j = tensor_index[1]
    i = tensor_index[0]

    vector_index = c + k*n_c + j*n_c*n_k + i*n_c*n_k*n_j

    return vector_index

def TRANSFER_MATRIX_INDEX(basis_tensor,transfer_basis_tensor):
    #creates the index where the transfer matrix should equal 1 such that the given basis vector is transformed properly
    #basis_tensor: the basis vector of the field vectors in tensor form
    #transfer_basis_tensor: the basis vector operated on by the transfer matrix in tensor form
    #transfer_matrix_index: the index where the transfer matrix should equal 1

    #create the basis vector
    basis_vector = VECTORIZE(basis_tensor)
    #determine the column index
    column_index = np.flatnonzero(basis_vector)
    #create the transfered basis vector vector
    transfer_basis_vector = VECTORIZE(transfer_basis_tensor)
    #determin the column vector row index
    row_index = np.flatnonzero(transfer_basis_vector)
    #package as int tuple
    transfer_matrix_index = (int(row_index),int(column_index))

    return transfer_matrix_index

def NORMALIZED_CAPACITANCE(alpha,n):
    #finds the normalized capacitance based off of alpha
    #alpha: array of alpha values for each direction - float, shape (3,0)
    #alpha_l: the alpha of the equivalent unit cell length - float, shape (1,0)
    #C normalized capacitance for a given scatter sub matrix - float, shape (3,3)

    #define equivalent cubic cell parameter to propagation delay ratio
    alpha_l = 0.49*2/n

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

def NORMALIZED_IMPEDANCE(alpha,n):
    
    normalized_capacitance = NORMALIZED_CAPACITANCE(alpha,n)

    perm = GET_PERMUTATIONS()

    normalized_inductance = np.zeros((3,3),dtype = float)

    for l in range(6):
        
        i = perm[l,:][0]
        j = perm[l,:][1]
        k = perm[l,:][2]

        normalized_inductance[i,k] = normalized_capacitance[i,j]

    normalized_impedance = normalized_inductance/normalized_capacitance

    return normalized_impedance

def SUM_R(r,sum_type):
    #sums the variable r in SCATTER without the identical spatial permuation
    #r: r variable from SCATTER - float, shape (3,3)
    #sum_type: chooses the type of summation

    if sum_type == 1:

        sum_1 = 0
        for i in range(3):
            for j in range(3):
                    if i == j:
                        sum_1 = sum_1
                    else:
                        sum_1 = r[i,j] + sum_1

        return sum_1

    elif sum_type == 2:

        sum_2 = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == j or i == k or j == k:
                        sum_2 = sum_2
                    else:
                        sum_2 = r[i,j]*r[i,k] + sum_2

        return sum_2

    elif sum_type == 3:

        sum_3 = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == j or i == k or j == k:
                        sum_3 = sum_3
                    else:
                        sum_3 = r[i,j]*r[j,k] + sum_3

        return sum_3

def TEST_EQUATIONS(C):
    #use the equations explicitly to compare with the matrix
    #C normalized capacitance for a given scatter sub matrix - float, shape (3,3)

    V_i_n = np.zeros((3,3))
    V_i_p = np.zeros((3,3))
    V_i_n[1,0] = 1

    V = np.zeros((3),dtype = float)
    IZ = np.zeros((3),dtype = float)

    V_r_n = np.zeros((3,3))
    V_r_p = np.zeros((3,3))

    perm = GET_PERMUTATIONS()

    for l in range(6):
        i = perm[l,:][0]
        j = perm[l,:][1]
        k = perm[l,:][2]

        V[j] = C[i,j] * ( V_i_n[i,j] + V_i_p[i,j] ) + C[k,j] * ( V_i_n[k,j] + V_i_p[k,j] ) 
        IZ[k] = C[i,k] * ( V_i_p[i,j] - V_i_n[i,j] + V_i_n[j,i] - V_i_p[j,i] )

        V_r_n[i,j] = V[j] + IZ[k] - V_i_p[i,j]
        V_r_p[i,j] = V[j] - IZ[k] - V_i_n[i,j]

    V_r = [V_r_n[1,0],V_r_n[2,0],V_r_n[0,1],V_r_n[2,1],V_r_n[1,2],V_r_n[0,2],V_r_p[1,2],V_r_p[2,1],V_r_p[2,0],V_r_p[0,2],V_r_p[0,1],V_r_p[1,0]]
    print('V_r:\n', np.around(V_r,5))

#def SCATTER_SUB_MATRIX_test(alpha,eps):
    #     #produces the scatter sub matrix for one time step and one position
    #     #alpha: array of space to time ratios for each dimension - float, shape (3,) 
    #     #eps: array of relative (unitless) dielectric constant for each dimension - float, shape (3,) 
    #     #scatter_sub_matrix: array of scatter sub matrix values - float, shape (12,12)

    #     #define variable r (assuming permeability is equal to free space) over all permutations of spatial dimension pairs

    #     r = np.zeros((3,3),dtype = float)

    #     for i in range(3):
    #         for j in range(3):
    #             r[i,j] = alpha[i]**-2 / eps[j]
                
    #     print('r: ', r)
        
    #     #define variable A

    #     A = 1 - SUM_R(r,1) + SUM_R(r,2) + SUM_R(r,3)

    #     #define normalized capacitance C

    #     C = np.zeros((3,3),dtype = float)

    #     for i in range(3):

    #         j = (i+1)%3
    #         k = (i+2)%3

    #         C[i,j] = ( 2*r[i,j]*(1 - r[j,k] - r[k,i] - r[i,k]) + A - np.sqrt( A * ( A - (np.sqrt(r[0,1]*r[1,2]*r[2,0]) + np.sqrt(r[0,2]*r[1,0]*r[2,1]))**2 ) ) ) / ( 2*(1 - r[j,k] - r[k,i] - r[j,i] - r[i,k] + r[j,k]*r[k,i] + r[j,i]*r[i,k] + r[j,i]*r[j,k]) )
    #         print('A - R > 0:', A - (np.sqrt(r[0,1]*r[1,2]*r[2,0]) + np.sqrt(r[0,2]*r[1,0]*r[2,1]))**2)
            

    #     print('A:',A)

    #     C[0,2] = 1 - C[1,2]
    #     C[1,0] = 1 - C[2,0]
    #     C[2,1] = 1 - C[0,1]

    #     print('C:',C)

    #     #define matrix variables

    #     b = np.ones((3,3),dtype = float)*0.5
    #     #d = np.zeros((3,3),dtype = float)
    #     d = np.ones((3,3),dtype = float)*0.5

    #     # for i in range(3):
    #     #     for j in range(3):
    #     #         for k in range(3):
    #     #             b[i,j] = C[k,j]
    #     #             d[i,j] = C[i,k]

    #     for i in range (3):
    #         for j in range(3):
    #             for k in range(3):
    #                 C[i,k] = d[i,j]
    #                 C[i,j] = 1-b[i,j]

    #     a = 1 - b - d
    #     c = d - b

    #     #build sub matrix

    #     # scatter_sub_matrix = [[a[0,1],b[0,1],d[0,1],0,0,0,0,0,b[0,1],0,-d[0,1],c[0,1]],
    #     #     [b[0,2],a[0,2],0,0,0,d[0,2],0,0,c[0,2],-d[0,2],0,b[0,2]],
    #     #     [d[1,0],0,a[1,0],b[1,0],0,0,0,b[1,0],0,0,c[1,0],-d[1,0]],
    #     #     [0,0,b[1,2],a[1,2],d[1,2],0,-d[1,2],c[1,2],0,0,b[1,2],0],
    #     #     [0,0,0,d[2,1],a[2,1],b[2,1],c[2,1],-d[2,1],0,b[2,1],0,0],
    #     #     [0,d[2,0],0,0,b[2,0],a[2,0],b[2,0],0,-d[2,0],c[2,0],0,0],
    #     #     [0,0,0,-d[2,1],c[2,1],b[2,1],a[2,1],d[2,1],0,b[2,1],0,0],
    #     #     [0,0,b[1,2],c[1,2],-d[1,2],0,d[1,2],a[1,2],0,0,b[1,2],0],
    #     #     [b[0,2],c[0,2],0,0,0,-d[0,2],0,0,a[0,2],d[0,2],0,b[0,2]],
    #     #     [0,-d[2,0],0,0,b[2,0],c[2,0],b[2,0],0,d[2,0],a[2,0],0,0],
    #     #     [-d[1,0],0,c[1,0],b[1,0],0,0,0,b[1,0],0,0,a[1,0],d[1,0]],
    #     #     [c[0,1],b[0,1],-d[0,1],0,0,0,0,0,b[0,1],0,d[0,1],a[0,1]]]

    #     scatter_sub_matrix = np.array( [[a[1,0],b[1,0],d[1,0],0,0,0,0,0,b[1,0],0,-d[1,0],c[1,0]],
    #         [b[2,0],a[2,0],0,0,0,d[2,0],0,0,c[2,0],-d[2,0],0,b[2,0]],
    #         [d[0,1],0,a[0,1],b[0,1],0,0,0,b[0,1],0,0,c[0,1],-d[0,1]],
    #         [0,0,b[2,1],a[2,1],d[2,1],0,-d[2,1],c[2,1],0,0,b[2,1],0],
    #         [0,0,0,d[1,2],a[1,2],b[1,2],c[1,2],-d[1,2],0,b[1,2],0,0],
    #         [0,d[0,2],0,0,b[0,2],a[0,2],b[0,2],0,-d[0,2],c[0,2],0,0],
    #         [0,0,0,-d[1,2],c[1,2],b[1,2],a[1,2],d[1,2],0,b[1,2],0,0],
    #         [0,0,b[2,1],c[2,1],-d[2,1],0,d[2,1],a[2,1],0,0,b[2,1],0],
    #         [b[2,0],c[2,0],0,0,0,-d[2,0],0,0,a[2,0],d[2,0],0,b[2,0]],
    #         [0,-d[0,2],0,0,b[0,2],c[0,2],b[0,2],0,d[0,2],a[0,2],0,0],
    #         [-d[0,1],0,c[0,1],b[0,1],0,0,0,b[0,1],0,0,a[0,1],d[0,1]],
    #         [c[1,0],b[1,0],-d[1,0],0,0,0,0,0,b[1,0],0,d[1,0],a[1,0]]] )

    #     #conservation of energy of test
    #     admittance_sub_matrix = ADMITTANCE(C,eps,alpha)
    #     zero_matrix = np.matmul( np.matmul( scatter_sub_matrix.transpose(),admittance_sub_matrix ), scatter_sub_matrix ) - admittance_sub_matrix

    #     print('Admittance sub matrix:', admittance_sub_matrix)

    #     print('zero matrix:', zero_matrix)
    
    #return scatter_sub_matrix

def SCATTER_SUB_MATRIX(alpha,n):
    #produces the scatter sub matrix for one time step and one position
    #alpha: array of space to time ratios for each dimension - float, shape (3,) 
    #n: refractive index of scatter sub matrix - float, shape (1,) 
    #scatter_sub_matrix: array of scatter sub matrix values - float, shape (12,12)

    #define normalized capacitance
    C = NORMALIZED_CAPACITANCE(alpha,n)

    #define matrix variables
    b = np.ones((3,3),dtype = float)
    d = np.ones((3,3),dtype = float)

    perm = GET_PERMUTATIONS()

    for l in range(6):
        i = perm[l,:][0]
        j = perm[l,:][1]
        k = perm[l,:][2]
    
        b[i,j] = C[k,j]
        d[i,j] = C[i,k]

    a = 1 - b - d
    c = d - b

    #build sub matrix
    scatter_sub_matrix =  [[a[1,0],b[1,0],d[1,0],0,0,0,0,0,b[1,0],0,-d[1,0],c[1,0]],
        [b[2,0],a[2,0],0,0,0,d[2,0],0,0,c[2,0],-d[2,0],0,b[2,0]],
        [d[0,1],0,a[0,1],b[0,1],0,0,0,b[0,1],0,0,c[0,1],-d[0,1]],
        [0,0,b[2,1],a[2,1],d[2,1],0,-d[2,1],c[2,1],0,0,b[2,1],0],
        [0,0,0,d[1,2],a[1,2],b[1,2],c[1,2],-d[1,2],0,b[1,2],0,0],
        [0,d[0,2],0,0,b[0,2],a[0,2],b[0,2],0,-d[0,2],c[0,2],0,0],
        [0,0,0,-d[1,2],c[1,2],b[1,2],a[1,2],d[1,2],0,b[1,2],0,0],
        [0,0,b[2,1],c[2,1],-d[2,1],0,d[2,1],a[2,1],0,0,b[2,1],0],
        [b[2,0],c[2,0],0,0,0,-d[2,0],0,0,a[2,0],d[2,0],0,b[2,0]],
        [0,-d[0,2],0,0,b[0,2],c[0,2],b[0,2],0,d[0,2],a[0,2],0,0],
        [-d[0,1],0,c[0,1],b[0,1],0,0,0,b[0,1],0,0,a[0,1],d[0,1]],
        [c[1,0],b[1,0],-d[1,0],0,0,0,0,0,b[1,0],0,d[1,0],a[1,0]]]

    #conservation of energy of test
    #scatter_sub_matrix_nparray = np.array(scatter_sub_matrix)
    #admittance_sub_matrix = ADMITTANCE(C,n**2,alpha)
    #zero_matrix = np.matmul( np.matmul( scatter_sub_matrix_nparray.transpose(),admittance_sub_matrix ), scatter_sub_matrix ) - admittance_sub_matrix

    #print('zero matrix:\n', np.around(zero_matrix,8))
    
    return scatter_sub_matrix

def SCATTER_MATRIX(alpha,n):
    #produces th3 scatter matrix for the entire physical region
    #alpha - the spatial step to temportal step ratios for each dimenions for each location - float, shape (n_i,n_j,n_k,3)
    #n -the refractive index values for each location - float, shape (n_i,n_j,n_k,1)

    n_i,n_j,n_k,_ = np.shape(alpha)

    scatter_matrix = SCATTER_SUB_MATRIX(alpha[0,0,0,0:3],n[0,0,0,0])

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):

                if not (i == 0 and j == 0 and k == 0):

                    scatter_sub_matrix = SCATTER_SUB_MATRIX(alpha[i,j,k,0:3],n[i,j,k,0])
                    scatter_matrix = sp.block_diag((scatter_matrix,scatter_sub_matrix))

    return scatter_matrix.tocsc()

#def TRANSFER_MATRIX(n_c,alpha,ref_index):
    #produces the transfer matrix responsible for transfering field values to their adjecent nodes
    #n_c: number of field components at each node - int, shape(1,)
    #alpha - the spatial step to temportal step ratios for each dimenions for each location - float, shape (n_i,n_j,n_k,3)
    #ref_index -the refractive index values for each location - float, shape (n_i,n_j,n_k,1)

    n_i,n_j,n_k,_ = np.shape(alpha)
    
    #size of transfer matrix (n X n)
    n = n_i*n_j*n_k*n_c

    #initilize basis and tansfered basis tensors
    basis_tensor = np.zeros((n_i,n_j,n_k,n_c),dtype = np.float32)
    transfer_basis_tensor = np.zeros((n_i,n_j,n_k,n_c),dtype = np.float32)

    #initilize row, col and data arrays for transfer matrix
    transfer_matrix_row = np.zeros(n,dtype = np.float32)
    transfer_matrix_col = np.zeros(n,dtype = np.float32)
    transfer_matrix_data = np.zeros(n,dtype = np.float32)
    ii = 0

    #get reflection at each wall
    reflection = REFLECTION()

    #determine column vectors by operating the transfer matrix on basis vectors
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                for c in range(n_c):
                    
                    #define basis tensor
                    index = (i,j,k,c)
                    basis_tensor[index] = 1
                    
                    #find index of transfered basis tensor that should be 1
                    transfer_index = TRANSFER_OPERATIONS(index)

                    #check transfer index to see if it sends light out of bounds
                    out_of_bounds = TRANSFER_INDEX_CHECK(transfer_index,n_i,n_j,n_k)

                    if not out_of_bounds:

                        #define transfered basis tensor
                        transfer_basis_tensor[transfer_index] = 1
                        
                        #determine index for transfer matrix
                        transfer_matrix_index = TRANSFER_MATRIX_INDEX(basis_tensor,transfer_basis_tensor)
                        
                        #determine row/column values for specific element in transfer matrix
                        transfer_matrix_row[ii] = transfer_matrix_index[0]
                        transfer_matrix_col[ii] = transfer_matrix_index[1]
                        transfer_matrix_data[ii] = 1
                        
                        #zero out transfered basis tensor
                        transfer_basis_tensor[transfer_index] = 0

                    else:
                        
                        #define normalized inductance for given component
                        z_hat = NORMALIZED_IMPEDANCE(alpha[i,j,k,:],ref_index[i,j,k,:])
                        direction,polarization,polarity = COMPONENT_2_INDEX(c)
                        z_hat = z_hat[direction,polarization]
                        
                        #reflection coefficient
                        rho = ( (1 + reflection[direction,polarization,polarity]) - z_hat * (1 - reflection[direction,polarization,polarity]) ) / ( (1 + reflection[direction,polarization,polarity]) + z_hat*(1 - reflection[direction,polarization,polarity]) )

                        #determine index for transfer matrix
                        transfer_matrix_index = TRANSFER_MATRIX_INDEX(basis_tensor,basis_tensor)

                        #determine row/column values for specific element in transfer matrix
                        transfer_matrix_row[ii] = transfer_matrix_index[0]
                        transfer_matrix_col[ii] = transfer_matrix_index[1]
                        transfer_matrix_data[ii] = rho

                    #zero out basis tensors
                    basis_tensor[index] = 0

                    #increment ii
                    ii = ii + 1

 
    transfer_matrix = coo_matrix((transfer_matrix_data, (transfer_matrix_row, transfer_matrix_col)), shape=(n, n)).tocsc()


    return transfer_matrix

def TRANSFER_MATRIX_2(n_c,alpha,ref_index):
    #produces the transfer matrix responsible for transfering field values to their adjecent nodes by calculating the transfer
    #basis vectors 
    #n_c: number of field components at each node - int, shape(1,)
    #alpha - the spatial step to temportal step ratios for each dimenions for each location - float, shape (n_i,n_j,n_k,3)
    #ref_index -the refractive index values for each location - float, shape (n_i,n_j,n_k,1)

    n_i,n_j,n_k,_ = np.shape(alpha)
    
    #size of transfer matrix (n X n)
    n = n_i*n_j*n_k*n_c

    #initilize row, col and data arrays for transfer matrix
    transfer_matrix_row = np.zeros(n,dtype = np.float32)
    transfer_matrix_col = np.zeros(n,dtype = np.float32)
    transfer_matrix_data = np.zeros(n,dtype = np.float32)
    ii = 0

    #get reflection at each wall
    reflection = REFLECTION()

    #determine column vectors by operating the transfer matrix on basis vectors
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                for c in range(n_c):
                    
                    #define index
                    index = (i,j,k,c)
                    
                    #find index of transfered basis tensor that should be 1
                    transfer_index = TRANSFER_OPERATIONS(index)

                    #check transfer index to see if it sends light out of bounds
                    out_of_bounds = TRANSFER_INDEX_CHECK(transfer_index,n_i,n_j,n_k)

                    if not out_of_bounds:

                        #convert original tensor index to a vector index. This index is the only row the basis vector 
                        #is non-zero and equal to one. This row value corresponds to the column the transfered basis vector
                        #will be placed into.
                        transfer_matrix_col[ii] = TENSOR_INDEX_TO_VECTOR_INDEX(index,n_i,n_j,n_k,n_c)

                        #convert transfer tensor index to a vector index. This index is the only row the transfered basis 
                        # vector is non-zero and equal to one.
                        transfer_matrix_row[ii] = TENSOR_INDEX_TO_VECTOR_INDEX(transfer_index,n_i,n_j,n_k,n_c)
                        
                        #determine row/column values for specific element in transfer matrix
                        transfer_matrix_data[ii] = 1

                    else:
                        
                        #define normalized inductance for given component
                        z_hat = NORMALIZED_IMPEDANCE(alpha[i,j,k,:],ref_index[i,j,k,:])
                        direction,polarization,polarity = COMPONENT_2_INDEX(c)
                        z_hat = z_hat[direction,polarization]
                        
                        #reflection coefficient
                        rho = ( (1 + reflection[direction,polarization,polarity]) - z_hat * (1 - reflection[direction,polarization,polarity]) ) / ( (1 + reflection[direction,polarization,polarity]) + z_hat*(1 - reflection[direction,polarization,polarity]) )

                        #convert original tensor index to a vector index. This index is the only row the basis vector 
                        #is non-zero and equal to one. This row value corresponds to the column the transfered basis vector
                        #will be placed into.
                        transfer_matrix_col[ii] = TENSOR_INDEX_TO_VECTOR_INDEX(index,n_i,n_j,n_k,n_c)

                        #the wave is reflected back to itself...
                        transfer_matrix_row[ii] = transfer_matrix_col[ii]
                        
                        #...with a given rho reflection value
                        transfer_matrix_data[ii] = rho

                    #increment ii
                    ii = ii + 1

    transfer_matrix = coo_matrix((transfer_matrix_data, (transfer_matrix_row, transfer_matrix_col)), shape=(n, n)).tocsc()


    return transfer_matrix

def ADMITTANCE(C,n,alpha):
    #creates the characteristic admittance matrix corresponding to a scatter sub matrix
    #alpha: array of space to time ratios for each dimension - float, shape (3,) 
    #n: refractive index for the scatter sub cell - float, shape (1,) 
    #C: normalized capacatice of scatter sub matrix - float, shape (3,3)

    #characteristic impednace
    Z = np.ones((3,3),dtype = float)

    perm = GET_PERMUTATIONS()

    for l in range(6):

        i = perm[l,:][0]
        j = perm[l,:][1]
        k = perm[l,:][2]
        
        Z[i,j] = alpha[j]/(alpha[i]*alpha[k]*C[i,j]*n**2)

    #characteristic admittance
    Y = 1/Z

    #characteristic admittance matrix
    admittance_sub_matrix = np.diag([Y[1,0],Y[2,0],Y[0,1],Y[2,1],Y[1,2],Y[0,2],Y[1,2],Y[2,1],Y[2,0],Y[0,2],Y[0,1],Y[1,0]])

    return admittance_sub_matrix

def PROPAGATION_MATRIX(alpha,n,n_c):
    #propagate the input signal through the simulation


    start = time.time()
    scatter_matrix = SCATTER_MATRIX(alpha,n)
    end = time.time()

    print('Time for scatter matrix creation: ',end-start)

    start = time.time()
    transfer_matrix = TRANSFER_MATRIX_2(n_c,alpha,n)
    end = time.time()

    print('Time for transfer matrix creation: ',end-start)

    start = time.time()
    propagation_matrix = transfer_matrix @ scatter_matrix
    end = time.time()

    print('Time for propagation matrix creation: ',end-start)

    return propagation_matrix , transfer_matrix

def PROPAGATE(alpha,n,n_c,source_scatter_field_vector,n_t):

    propagation_matrix, transfer_matrix = PROPAGATION_MATRIX(alpha,n,n_c)

    start = time.time()
    l = np.shape(source_scatter_field_vector[:,0])
    scatter_field_vector = coo_matrix(np.zeros((l[0],1),dtype = np.float32)).tocsc()

    source_scatter_field_vector = coo_matrix(source_scatter_field_vector).tocsc()
    scatter_field_vector_time = lil_matrix(np.zeros((l[0],n_t),dtype = np.float32))

    for t in range(n_t):
        source = source_scatter_field_vector[:,t]
        source.shape = (l[0],1)
        scatter_field_vector = scatter_field_vector + source
        scatter_field_vector = propagation_matrix @ scatter_field_vector
        scatter_field_vector_time[:,t] = scatter_field_vector

    end = time.time()

    print('Time for propagation calculation: ',end-start)

    return scatter_field_vector.toarray() , scatter_field_vector_time.toarray(), transfer_matrix



