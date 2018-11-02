import tensorflow as tf
import numpy as np
import pickle

data_type = np.float32

# ---------------------- Index Assignment -------------------------------#
def INDEX_2_COMPONENT(i,j,s):
    # convert direction, polarization and polarity index to field component
    # i - direction, int 
    # j - polarization, int
    # s - polarity
    #
    # c - scatter field component

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

def COMPONENT_2_INDEX(c):
    # convert scatter field component to direction, polarizaiton and polarity index
    # c - scatter field component
    #
    # i - direction, int 
    # j - polarization, int
    # s - polarity
    
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

    return i,j,s

# ---------------------- Tensors and Tensor Operations ------------------#
def CONSTANT_MATRICES():
    #produces the matrices of constants used to construct the scatter tensor

    a = 1
    c = 0
    b = 0
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
    c = 1
    b = 0
    d = 1

    filter_d_matrix =  np.array([[a,b,d,0,0,0,0,0,b,0,-d,c],
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
    b = 1
    d = 0

    filter_b_matrix =  np.array([[a,b,d,0,0,0,0,0,b,0,-d,c],
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

    beta1_vector = np.array([1,-1,-1,1,-1,1,-1,1,-1,1,-1,1])
    beta2_vector = np.array([0,1,1,0,1,0,1,0,1,0,1,0])

    beta1_matrix = np.zeros((12,12))
    beta2_matrix = np.zeros((12,12))

    for i in range(12):
        beta1_matrix[:,i] = beta1_vector
        beta2_matrix[:,i] = beta2_vector

    return filter_d_matrix, filter_b_matrix, ones_matrix , beta1_matrix , beta2_matrix

def MESH_INDEX(c):
    #get the mesh index for a given field component c
    #c: field component - int shape(1,)

    #table relating c index to mesh direction (0-x, 1-y, 2-z)
    index_d = np.array( [[1,2,0],[0,1,2],[1,2,0],[2,0,1],[2,0,1],[0,1,2],[2,0,1],[2,0,1],[0,1,2],[0,1,2],[1,2,0],[1,2,0]] )
    index_b = np.array( [[2,0,1],[2,0,1],[0,1,2],[0,1,2],[1,2,0],[1,2,0],[1,2,0],[0,1,2],[2,0,1],[1,2,0],[0,1,2],[2,0,1]] )

    return index_d[c,:] , index_b[c,:]

def CONSTANT_TENSORS(mesh,n_c):
    #produces the constant tensors used to construct the scatter tensor and boundary tensor
    #mesh: contains the alpha mesh variables (del_space/del_time/c0) in each direction - numpy.array shape(n_x,n_y,n_z,3)
    #n_c: number of field components

    #produce matrices for filter the d1 and d2 matrix along with the ones matrix
    filter_d_matrix, filter_b_matrix, ones_matrix , beta1_matrix , beta2_matrix = CONSTANT_MATRICES()
    #spatial parameters
    n_x,n_y,n_z,_ = np.shape(mesh)

    #initilize boundary tensor
    boundary = np.ones((n_x,n_y,n_z,n_c),dtype = data_type)

    #initilize constant tensors
    a1 = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    a2 = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    a3 = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    a4 = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    a5 = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    a6 = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
  
    beta1_tens = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    beta2_tens = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)

    filter_d_tens = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)
    filter_b_tens = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)

    ones_tens = np.zeros((n_x,n_y,n_z,n_c,n_c), dtype = data_type)

    #build constant tensors based on relationship between mesh index and field component c
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                #build boundary tensor
                if y == 0 or x == n_y - 1:
                      boundary[x,y,z,:] = np.zeros(n_c,dtype = data_type)

                #build filter and ones tensors
                filter_d_tens[x,y,z,:,:] = filter_d_matrix
                filter_b_tens[x,y,z,:,:] = filter_b_matrix
                ones_tens[x,y,z,:,:] = ones_matrix
                beta1_tens[x,y,z,:,:] = beta1_matrix
                beta2_tens[x,y,z,:,:] = beta2_matrix

                for c1 in range(n_c):

                    d_index,b_index = MESH_INDEX(c1)

                    kd = d_index[2]
                    jd = d_index[1]
                    id = d_index[0]

                    kb = b_index[2]
                    jb = b_index[1]
                    ib = b_index[0]

                    for c2 in range(n_c):
                        
                        a1[x,y,z,c1,c2] = ( mesh[x,y,z,0] * mesh[x,y,z,1] * mesh[x,y,z,2] ) ** 2
                        a2[x,y,z,c1,c2] = mesh[x,y,z,0] ** -2 + mesh[x,y,z,1] ** -2 + mesh[x,y,z,2] ** -2

                        a3[x,y,z,c1,c2] = 2 * mesh[x,y,z,kd]**2 * mesh[x,y,z,jd]**2 
                        a4[x,y,z,c1,c2] = 2 * mesh[x,y,z,id]**2 * mesh[x,y,z,kd]**2 
                        a5[x,y,z,c1,c2] = 2 * mesh[x,y,z,jb]**2 * mesh[x,y,z,kb]**2 
                        a6[x,y,z,c1,c2] = 2 * mesh[x,y,z,ib]**2 * mesh[x,y,z,kb]**2 

    return a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,beta1_tens,beta2_tens,boundary

# ---------------------- Transfer Operation -----------------------------#
def TRANSFER_OPERATION(field):
    #transfers field components for all points in space
    #field: the field to be transfered - shape(n_x,n_y,n_z,n_c)

    with tf.name_scope("transfer_op"):
        #explain shifts here
        tmp0 = tf.manip.roll(field[:,:,:,0],shift=-1,axis=1)
        tmp1 = tf.manip.roll(field[:,:,:,1],shift=-1,axis=2)
        tmp2 = tf.manip.roll(field[:,:,:,2],shift=-1,axis=0)
        tmp3 = tf.manip.roll(field[:,:,:,3],shift=-1,axis=2)
        tmp4 = tf.manip.roll(field[:,:,:,4],shift=-1,axis=1)
        tmp5 = tf.manip.roll(field[:,:,:,5],shift=-1,axis=0)
        tmp6 = tf.manip.roll(field[:,:,:,6],shift=1,axis=1)
        tmp7 = tf.manip.roll(field[:,:,:,7],shift=1,axis=2)
        tmp8 = tf.manip.roll(field[:,:,:,8],shift=1,axis=2)
        tmp9 = tf.manip.roll(field[:,:,:,9],shift=1,axis=0)
        tmp10 = tf.manip.roll(field[:,:,:,10],shift=1,axis=0)
        tmp11 = tf.manip.roll(field[:,:,:,11],shift=1,axis=1)

        transferred_field = tf.stack([tmp11,tmp8,tmp10,tmp7,tmp6,tmp9,tmp4,tmp3,tmp1,tmp5,tmp2,tmp0],axis=3,name="stack")

        return transferred_field
        
# ---------------------- Scatter Tensor Generation ----------------------#
def SCATTER(weight_tens,a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,n_c,n_x,n_y,n_z,n_w,beta1_tens,beta2_tens):
    #produces the scatter tensor to operate on the field tensor
    #weight_tens: weight tensor - shape (layers,n_x,n_y,n_z,n_w)
    #S: scatter tensor - shape (n_x,n_y,n_z,n_c,n_c)

    #assign a weight to each field component
    w = tf.reshape(weight_tens,[n_x,n_y,n_z,n_w,n_w])
    w = tf.tile(w,[1,1,1,n_c//n_w,n_c//n_w])

    #calculate portions of the Cd and Cb tensors
    A = tf.multiply(a1,4.0*tf.reciprocal(w) - a2)
    B = A - tf.sqrt( tf.multiply(A,A) - tf.multiply(a1,w) )

    #calculate the normalized capacitance tensors
    Cd = tf.divide ( a3 + B , tf.multiply( tf.multiply (8.0,a1),tf.reciprocal(w) ) - a4 )
    Cb = tf.divide ( a5 + B , tf.multiply( tf.multiply (8.0,a1),tf.reciprocal(w) ) - a6 )

    #combine to produce the d and b tensors
    d = beta2_tens + tf.multiply(beta1_tens,Cd)
    b = beta2_tens + tf.multiply(beta1_tens,Cb)

    #produce the scatter tensor
    scatter_tensor = tf.multiply(filter_d_tens,d) + tf.multiply(filter_b_tens,b) + ones_tens

    return scatter_tensor

# -------------------------- Propagation Operation ------------------------#
def PROPAGATE_1D(field_in_tens,mesh,n_c,weight_tens,n_t,n_w):
    # propagte the fields in STSN assuming a 1D simulation
    # field_in_tens: input field tensor containing the field at all points in space and time - tf.constant tf.float##, shape(n_x,n_y,n_z,n_c,n_t)
    # mesh: holds the mesh length values for all space - np.array int, shape(n_x,n_y,n_z,3)
    # n_c: number of field componentes - int, shape(1)
    # weight_tens: weights in all regions - tf.Variable tf.float##, shape(n_x,n_y,n_z,n_w)
    # n_t: number of time steps - int, shape(1,)
    # n_w: number of weight steps - int, shape(1,)

    #spatial steps
    n_x,n_y,n_z,_ = np.shape(mesh)

    #produce constant tensors for scatter and transfer operations
    a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,beta1_tens,beta2_tens,boundary = CONSTANT_TENSORS(mesh,n_c)

    #create scatter tensor
    scatter_tensor = SCATTER(weight_tens,a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,n_c,n_x,n_y,n_z,n_w,beta1_tens,beta2_tens)
    
    #keep field_out_tens_tmp a rank 4 tensor
    field_out_tens_tmp = 0*field_in_tens[:,:,:,:,0]

    #initilize field_out_tens
    source = field_in_tens[:,:,:,:,0]
    field_out_tens_tmp = source + field_out_tens_tmp

    field_out_tens_tmp = tf.einsum('ijkm,ijknm->ijkn',field_out_tens_tmp,scatter_tensor)
    field_out_tens_tmp = TRANSFER_OPERATION(field_out_tens_tmp)

    field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c,1))
    field_out_tens = field_out_tens_tmp
    field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c))

    #perform scatter operation
    for t in np.arange(1,n_t):
        source = field_in_tens[:,:,:,:,t]
        field_out_tens_tmp = source + field_out_tens_tmp

        field_out_tens_tmp = tf.einsum('ijkm,ijknm->ijkn',field_out_tens_tmp,scatter_tensor)
        field_out_tens_tmp = tf.multiply(field_out_tens_tmp,boundary)
        field_out_tens_tmp = TRANSFER_OPERATION(field_out_tens_tmp)

        field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c,1))
        field_out_tens = tf.concat( [field_out_tens,field_out_tens_tmp] , 4 )
        field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c))

    return field_out_tens,boundary

def PROPAGATE(field_in_tens,mesh,n_c,weight_tens,n_t,n_w):
    # propagte the fields in STSN assuming a 1D simulation
    # field_in_tens: input field tensor containing the field at all points in space and time - tf.constant tf.float##, shape(n_x,n_y,n_z,n_c,n_t)
    # mesh: holds the mesh length values for all space - np.array int, shape(n_x,n_y,n_z,3)
    # n_c: number of field componentes - int, shape(1)
    # weight_tens: weights in all regions - tf.Variable tf.float##, shape(n_x,n_y,n_z,n_w)
    # n_t: number of time steps - int, shape(1,)
    # n_w: number of weight steps - int, shape(1,)

    #spatial steps
    n_x,n_y,n_z,_ = np.shape(mesh)

    #produce constant tensors for scatter and transfer operations
    a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,beta1_tens,beta2_tens,_ = CONSTANT_TENSORS(mesh,n_c)

    #create scatter tensor
    scatter_tensor = SCATTER(weight_tens,a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,n_c,n_x,n_y,n_z,n_w,beta1_tens,beta2_tens)
    
    #keep field_out_tens_tmp a rank 4 tensor
    field_out_tens_tmp = 0*field_in_tens[:,:,:,:,0]

    #initilize field_out_tens
    source = field_in_tens[:,:,:,:,0]
    field_out_tens_tmp = source + field_out_tens_tmp

    field_out_tens_tmp = tf.einsum('ijkm,ijknm->ijkn',field_out_tens_tmp,scatter_tensor)
    field_out_tens_tmp = TRANSFER_OPERATION(field_out_tens_tmp)

    field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c,1))
    field_out_tens = field_out_tens_tmp
    field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c))

    #perform scatter operation
    for t in np.arange(1,n_t):
        source = field_in_tens[:,:,:,:,t]
        field_out_tens_tmp = source + field_out_tens_tmp

        field_out_tens_tmp = tf.einsum('ijkm,ijknm->ijkn',field_out_tens_tmp,scatter_tensor)
        field_out_tens_tmp = TRANSFER_OPERATION(field_out_tens_tmp)

        field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c,1))
        field_out_tens = tf.concat( [field_out_tens,field_out_tens_tmp] , 4 )
        field_out_tens_tmp = tf.reshape(field_out_tens_tmp,(n_x,n_y,n_z,n_c))

    return field_out_tens

def MASK(mask_start,mask_end,field_tens,n_x,n_y,n_z,n_c,data_type):

    mask_start_x = mask_start[0]
    mask_start_y = mask_start[1]
    mask_start_z = mask_start[2]

    mask_end_x = mask_end[0]
    mask_end_y = mask_end[1]
    mask_end_z = mask_end[2]

    mask = np.ones((n_x,n_y,n_z,n_c),dtype = data_type)
    mask[mask_start_x:mask_end_x+1,mask_start_y:mask_end_y+1,mask_start_z:mask_end_z+1,:] = np.zeros((mask_end_x-mask_start_x + 1,mask_end_y-mask_start_y + 1,mask_end_z-mask_start_z + 1,n_c),dtype = data_type)

    mask_field_tens = tf.multiply(mask,field_tens)

    return mask_field_tens





