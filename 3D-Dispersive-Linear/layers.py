import tensorflow as tf
import numpy as np
from numpy.polynomial import polynomial as p
import pickle

data_type = np.float32

# ---------------------------------------------------------------------- #
###         Dispersion Parameters

def ELECTRIC_DISPERSION_OPERATORS(res_freq,damp,del_x,del_t,inf_x):
    #converts the electrical dispersion physical parameters into relevent electric dispersion state variable operators
    #res_freq - resonant frequency (r/s) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    #damp - damping coefficient (r/s) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    #del_x - the change in susceptibility (unitless) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    #del_t - the time step of the simulation (s) - np.float32, shape(1,)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    #
    # sta_ope: a list of state operators - shape(4,)
    # tra_ope: a list of transmission operators - shape(2,)

    #secondary constants
    beta = np.sqrt(res_freq**2 - damp**2)
    A_e = np.exp(-damp*del_t)*np.cos(beta*del_t)
    B_e = np.exp(-damp*del_t)*np.sin(beta*del_t)
    K1 = del_x*(1-2*A_e+A_e**2+B_e**2)

    #primary constants
    a_1 = 2*A_e
    a_2 = -(A_e**2 + B_e**2)
    b_1 = -2*K1
    b_2 = 2*K1

    #calculate the lorentz transfer function
    tran_num_coeff , tran_den_coeff = LORENTZ_TRANSFER_FUNCTION(a_1,a_2,b_1,b_2)

    #convert the transfer function into state variable operators
    a , b , c , d = STATE_OPERATORS(tran_num_coeff , tran_den_coeff)

    #produce transmission operators
    t = 1/(2 + 2*inf_x)
    t = tf.stack([t,t,t],axis = -1)
    k = -(2-2*inf_x)
    k = tf.stack([k,k,k],axis = -1)

    #convert everything to tensors
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    c = tf.convert_to_tensor(c)
    d = tf.convert_to_tensor(d)
    t = tf.convert_to_tensor(t)
    k = tf.convert_to_tensor(k)

    #package operators
    sta_ope = [a,b,c,d]
    tra_ope = [t,k]

    return sta_ope , tra_ope

def STATE_OPERATORS(tran_num_coeff , tran_den_coeff):
    # produces tensors that dictate the operations governing the state variable
    # tran_num_coeff: transfer function numerator polynomial coefficients (w/ alpha_0 = 1) - np.array, shape (n_x,n_y,n_z,n_o+1)
    # tran_den_coeff: transfer function denominator polynomial coefficients (w/ alpha_0 = 1) - np.array, shape (n_x,n_y,n_z,n_o+1)
    #
    # a: constant tensor operating on x to update x - shape(n_x,n_y,n_z,n_s,n_s)
    # b: constant tensor operating on f to update x - shape(n_x,n_y,n_z,n_s)
    # c: constant tensor operating on x to update s_e_d - shape(n_x,n_y,n_z,n_s)
    # d: constant tensor operating on f to update s_e_d - shape(n_x,n_y,n_z)

    #get spatial constants, number of electric field components and polynomial order + 1
    #note: number of state variables (n_s) is equal to the polynomial order of the transfer function
    n_x,n_y,n_z,n_o_tmp = np.shape(tran_num_coeff)
    n_o = n_o_tmp - 1
    n_s = n_o

    #define alpha and beta values
    alpha = tran_den_coeff
    beta = tran_num_coeff

    #initilize tensors
    a = np.zeros((n_x,n_y,n_z,n_s,n_s),dtype = data_type)
    b = np.zeros((n_x,n_y,n_z,n_s),dtype = data_type)
    c = np.zeros((n_x,n_y,n_z,n_s),dtype = data_type)
    d = np.zeros((n_x,n_y,n_z),dtype = data_type)

    #matrix that acts on state variable matrix to update the state variable
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                a[x,y,z,:,:] = np.diag(np.ones(n_s-1), 1)
                b[x,y,z,:] = np.zeros(n_s)
                b[x,y,z,-1] = 1
                d[x,y,z] = beta[x,y,z,0]

                for s in range(n_s):
                    a[x,y,z,n_s-1,s] = -alpha[x,y,z,-s-1]
                    c[x,y,z,s:s+1] = beta[x,y,z,-s-1] - alpha[x,y,z,-s-1]*beta[x,y,z,0]

    return a , b , c , d

def LORENTZ_TRANSFER_FUNCTION(a_1,a_2,b_1,b_2):
    #determines the Lorentz model transfer function in the z domain in terms
    #of polynomial coefficients in the numerator and denominator
    #a_1,a_2,b_1,b_2: a primary constant of the Lorentz model - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    #
    # tran_num_coeff: transfer function numerator polynomial coefficients (w/ alpha_0 = 1) - np.array, shape (n_x,n_y,n_z,n_o+1)
    # tran_den_coeff: transfer function denominator polynomial coefficients (w/ alpha_0 = 1) - np.array, shape (n_x,n_y,n_z,n_o+1)

    #get spatial constants, number of electric field components and number of resonances
    n_x,n_y,n_z,n_r = np.shape(a_1)

    #determine highest order of transfer function
    n_o = 2*n_r

    #initilize the transmission numerator and denomenator polynomial coefficients
    tran_num_coeff = np.zeros((n_x,n_y,n_z,n_o+1))
    tran_den_coeff = np.zeros((n_x,n_y,n_z,n_o+1))

    #determine polynomial coefficients of each resonance function at each electric field component over all space
    num_coeff = np.stack((b_1,b_2,0*b_1),axis = -1)
    den_coeff = np.stack((a_1*0+1,-a_1,-a_2),axis = -1)

    #determine sum of resonance functions in terms of polynomial coeff for each electric field component over all space
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                num_results , den_results = SUM_RATIONAL_POLY(num_coeff[x,y,z,:,:],den_coeff[x,y,z,:,:])

                #zeropad lenght to ensure it includes coeff up to the highest order of the transfer function
                if len(num_results) < n_o + 1:
                    num_results = np.pad(num_results,(n_o + 1 - len(num_results),0),'constant')
                if len(den_results) < n_o + 1:
                    den_results = np.pad(den_results,(n_o + 1 - len(den_results),0),'constant')
                
                #save values normalized by the highest order denomantor coefficient
                tran_num_coeff[x,y,z,:] = num_results/den_results[0]
                tran_den_coeff[x,y,z,:] = den_results/den_results[0]

    return tran_num_coeff , tran_den_coeff

# ---------------------------------------------------------------------- #
###         Misc. Mathematical Operations

def SUM_RATIONAL_POLY(num_coeff,den_coeff):
    #determines the numerator and denomanator polynomial coefficients after summing rational polynomials.
    #assumes zero over zero polynomials are just 0
    #num_coeff: the coefficients (highest oder first) of the numerators for each rational polynomial - np.float32 np.array, shape (number of rational polynomials,highest order numerator)
    #den_coeff: the coefficients (highest oder first) of the denominators for each rational polynomial - np.float32 np.array, shape (number of rational polynomials,highest order denominator)
    #
    #numerator_results: the coefficients (highest oder first) of the numerator - np.float32 np.array, shape(order of numerator)
    #denominator_results: the coefficients (highest oder first) of the denominator - np.float32 np.array, shape(order of denominator)

    #get number of polynomials and highest order of polynomials
    n_p,n_o_tmp = np.shape(num_coeff)
    n_p_tmp,_= np.shape(den_coeff)
    n_o = n_o_tmp - 1

    #make sure there are an equal number of numerator and denominator polynomials
    if n_p != n_p_tmp:
        raise ValueError("Starting index must be smaller than or equal to ending index.")

    #make sure polynomials with all zeros are set to 0/1 = 0
    for i in range(n_p):
        if np.array_equal(den_coeff[i,:] , np.zeros(n_o+1)) and np.array_equal( num_coeff[i,:] , np.zeros(n_o+1) ):
            den_coeff[i,n_o] = 1
        elif np.array_equal(den_coeff[i,:] , np.zeros(n_o+1)) and ~np.array_equal( num_coeff[i,:] , np.zeros(n_o+1) ):
            raise ValueError("Denominator is zero and numerator is non-zero")


    #denominator
    d_results = np.poly1d([1])
    
    for i in range(n_p):
        d = np.poly1d(den_coeff[i,:])
        
        d_results = d_results*d

    #numerator 
    n_results = np.poly1d([0])

    for i in range(n_p):
        d = np.poly1d(den_coeff[i,:])
        n = np.poly1d(num_coeff[i,:])
        
        d_contribution,_ = d_results/d
        n_results = d_contribution * n + n_results

    return n_results.c , d_results.c

# ---------------------------------------------------------------------- #
###         Tensors and Tensor Operations

def CONSTANT_MATRICES():
    #produces the matrices of constants used to construct the scatter tensor

    #incident voltages to reflected fields
    r_1_t = np.array([[1,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,1,1,1],
                [0,0,0,0,0,0,-1,1,1,-1,0,0],
                [1,-1,0,0,0,0,0,0,0,0,-1,1],
                [0,0,-1,1,1,-1,0,0,0,0,0,0]],dtype = data_type)

    #total fields to reflected voltages
    r = np.array([[1,0,0,0,-1,0],
            [1,0,0,0,1,0],
            [1,0,0,0,0,1],
            [1,0,0,0,0,-1],
            [0,1,0,0,0,-1],
            [0,1,0,0,0,1],
            [0,1,0,1,0,0],
            [0,1,0,-1,0,0],
            [0,0,1,-1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,0,1,0],
            [0,0,1,0,-1,0]],dtype = data_type)

    #flip incident voltages
    p = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,1,0]],dtype = data_type)

    return r_1_t , r , p

def CONSTANT_TENSORS(inf_x,n_c,n_f):
    # produces the constant tensors used to scatter the fields and construct the boundary tensor
    # n_c: number of voltage components - shape(1,)
    # n_f: number of field components - shape(1,)

    #initilize matrices
    r_1_t_matrix , r_matrix , p_matrix = CONSTANT_MATRICES()

    #spatial parameters
    n_x,n_y,n_z = np.shape(inf_x)

    #initilize tensor
    boundary = np.ones((n_x,n_y,n_z,n_c),dtype = data_type)
    r_1_t = np.zeros((n_x,n_y,n_z,n_f,n_c),dtype = data_type)
    r = np.zeros((n_x,n_y,n_z,n_c,n_f),dtype = data_type)
    p = np.zeros((n_x,n_y,n_z,n_c,n_c),dtype = data_type)


    #build constant tensors based on relationship between mesh index and field component c
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                #incident voltages to reflected fields tensor
                r_1_t[x,y,z,:,:] = r_1_t_matrix
                r[x,y,z,:,:] = r_matrix
                p[x,y,z,:,:] = p_matrix


                #build boundary tensor
                if y == 0 or x == n_y - 1:
                      boundary[x,y,z,:] = np.zeros(n_c,dtype = data_type)

                        
    return tf.convert_to_tensor(r_1_t) , tf.convert_to_tensor(r) , tf.convert_to_tensor(p) , tf.convert_to_tensor(boundary)

# ---------------------------------------------------------------------- #
###            Transfer Operation

def TRANSFER(field):
    #transfers field components for all points in space
    #field: the field to be transfered - shape(n_x,n_y,n_z,n_c)

    with tf.name_scope("transfer_op"):
        #explain shifts here

        tmp0 = tf.manip.roll(field[:,:,:,0],shift=-1,axis=2)
        tmp1 = tf.manip.roll(field[:,:,:,1],shift=1,axis=2)
        tmp2 = tf.manip.roll(field[:,:,:,2],shift=-1,axis=1)
        tmp3 = tf.manip.roll(field[:,:,:,3],shift=1,axis=1)
        tmp4 = tf.manip.roll(field[:,:,:,4],shift=-1,axis=0)
        tmp5 = tf.manip.roll(field[:,:,:,5],shift=1,axis=0)
        tmp6 = tf.manip.roll(field[:,:,:,6],shift=-1,axis=2)
        tmp7 = tf.manip.roll(field[:,:,:,7],shift=1,axis=2)
        tmp8 = tf.manip.roll(field[:,:,:,8],shift=-1,axis=1)
        tmp9 = tf.manip.roll(field[:,:,:,9],shift=1,axis=1)
        tmp10 = tf.manip.roll(field[:,:,:,10],shift=-1,axis=0)
        tmp11 = tf.manip.roll(field[:,:,:,11],shift=1,axis=0)

        transferred_field = tf.stack([tmp1,tmp0,tmp3,tmp2,tmp5,tmp4,tmp7,tmp6,tmp9,tmp8,tmp11,tmp10],axis=3)

        return transferred_field

# ---------------------------------------------------------------------- #
###         Scattering

def LORENTZ(f,w_0,damp,del_x,del_t,x_1_pre,x_2_pre):
    # calculates a dielectric accumulator
    # f: total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # w_0: resonant frequency tensors  - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # damp: damping frequency tensors - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # del_x: change in susceptibility tensors - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # x_1_pre: previous first state variable - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # x_2_pre: previous second state variable - np.constant, shape (n_x,n_y,n_z,n_f/2)

    #first set of constants
    beta = tf.sqrt( tf.multiply(w_0,w_0) - tf.multiply(damp,damp) )
    a = tf.multiply( tf.exp(-damp*del_t) , tf.cos(beta*del_t) )
    b = tf.multiply( tf.exp(-damp*del_t) , tf.sin(beta*del_t) )

    #second set of constants
    a_1 = tf.cast(2*a,data_type)
    a_2 = tf.cast(-( tf.multiply(a,a) + tf.multiply(b,b) ),data_type)
    k_1 = tf.multiply( del_x , (1-a_1-a_2) )
    b_1 = -2*k_1
    b_2 = 2*k_1

    #calculate state variable
    x_1 = tf.multiply(a_1,x_1_pre) + tf.multiply(a_2,x_2_pre) + f
    x_2 = x_1_pre

    #calculate the dielectric accumulator
    s_d = tf.multiply(b_1,x_1) + tf.multiply(b_2,x_2)

    #update 
    x_1_pre = x_1
    x_2_pre = x_2

    return s_d , x_1_pre , x_2_pre

def MULTIPLE_LORENTZ(f,x,a,b,c,d):
    # calculates an electrical dielectric accumulator for Lorentz model with multiple resonances
    # f: total fields - np.constant, shape (n_x,n_y,n_z)
    # x: state variable tensor - np.constant, shape (n_x,n_y,n_z,n_o)
    # a: constant tensor operating on x to update x - shape(n_x,n_y,n_z,n_o,n_o)
    # b: constant tensor operating on f to update x - shape(n_x,n_y,n_z,n_o)
    # c: constant tensor operating on x to update s_e_d - shape(n_x,n_y,n_z,n_o)
    # d: constant tensor operating on f to update s_e_d - shape(n_x,n_y,n_z)
    #
    # s_e_d: the electrical dielectric accumulator - shape(n_x,n_y,n_z)
    # x_next: state variable tensor at the next time step - shape (n_x,n_y,n_z,n_o)

    #calculate state variable
    x_next = tf.einsum('ijko,ijkmo->ijkm',x,a) + tf.einsum('ijko,ijk->ijko ',b,f)

    #calculate the electrical dielectric accumulator
    s_e_d = tf.einsum('ijko,ijko->ijk',c,x) + tf.multiply(d,f)

    return s_e_d , x_next

def MULTIPLE_LORENTZ_2(f,x,a,b,c,d):
    # calculates an electrical dielectric accumulator for Lorentz model with multiple resonances
    # f: total fields - np.constant, shape (n_x,n_y,n_z)
    # x: state variable tensor - np.constant, shape (n_x,n_y,n_z,n_o)
    # a: constant tensor operating on x to update x - shape(n_x,n_y,n_z,n_o,n_o)
    # b: constant tensor operating on f to update x - shape(n_x,n_y,n_z,n_o)
    # c: constant tensor operating on x to update s_e_d - shape(n_x,n_y,n_z,n_o)
    # d: constant tensor operating on f to update s_e_d - shape(n_x,n_y,n_z)
    #
    # s_e_d: the electrical dielectric accumulator - shape(n_x,n_y,n_z)
    # x_next: state variable tensor at the next time step - shape (n_x,n_y,n_z,n_o)

    #calculate state variable
    x_next = a[0,0,0,:,:]@x + tf.multiply(b[0,0,0,:],f[0,0,0])


    #calculate the electrical dielectric accumulator
    s_e_d = tf.multiply(c[0,0,0,:],x) + tf.multiply(d[0,0,0],f[0,0,0])

    return s_e_d , x_next

def TRANSMISSION(f_r,inf_x,s_pre,x_1_pre,x_2_pre,w_0,damp,del_x,del_t):
    # transmitts the reflected fields into the total fields
    # f_r: reflected fields
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z,n_f)

    n_x,n_y,n_z,n_f = np.shape(inf_x)

    #build constant tensors
    t = tf.math.reciprocal(2.0 + 2.0*inf_x)
    t = tf.cast(t,data_type)
    k = tf.cast(-2 + 2*inf_x,data_type)

    #calculate total fields
    f = tf.multiply((f_r + s_pre), t)
    
    #calculate dielectric accumulator
    s_d,x_1_pre,x_2_pre = LORENTZ(f,w_0,damp,del_x,del_t,x_1_pre,x_2_pre)

    #calculate the main accumulator
    s = f_r + tf.multiply(k,f) + s_d

    #update
    s_pre = s

    return f,s_pre,x_1_pre,x_2_pre

def MULTIPLE_TRANSMISSION(f_r,sta_ope,x,t,k,s_e_pre):
    # transmitts the reflected fields into the total fields for a multiple 
    # resonance Lorentz model
    # f_r: reflected fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # sta_ope: a list of state operators - shape(4,)
    # x: state variable tensor - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)
    # t: constant tensor operating on f_r and s_pre to update f - shape(n_x,n_y,n_z,n_f/2)
    # k: constant tensor operating on f to update s - shape(n_x,n_y,n_z,n_f/2)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    #
    # f: total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    # x_next: the state variable matrix value at the next time step - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)

    #calculate total fields
    f = tf.multiply((f_r + s_e_pre), t)
    
    #calculate dielectric accumulator in x
    s_e_d_x , x_next_x = MULTIPLE_LORENTZ(f[:,:,:,0],x[:,:,:,:,0],sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])

    #calculate dielectric accumulator in y
    s_e_d_y , x_next_y = MULTIPLE_LORENTZ(f[:,:,:,1],x[:,:,:,:,1],sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])

    #calculate dielectric accumulator in z
    s_e_d_z , x_next_z = MULTIPLE_LORENTZ(f[:,:,:,2],x[:,:,:,:,2],sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])

    #combine x,y and z components
    s_e_d = tf.stack([s_e_d_x,s_e_d_y,s_e_d_z],axis = -1)
    x_next = tf.stack([x_next_x,x_next_y,x_next_z],axis = -1)

    #calculate the main accumulator
    s_e_pre = f_r + tf.multiply(k,f) + s_e_d

    return f , s_e_pre , x_next

def SCATTER(v_i,v_f,r_1_t,r,p,inf_x,s_pre,x_1_pre,x_2_pre,w_0,damp,del_x,del_t):
    # scatters the incident voltages and free-source fields into reflected
    # voltages
    # v_i: incident voltages , tf.constant - shape(n_x,n_y,n_z,n_c)
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f)
    # r_1_t: converts incident voltages to reflected fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_c)
    # r: converts total fields to reflected voltages , tf.constant - shape(n_x,n_y,n_z,n_c,n_f)
    # p: flips incidnet voltages , tf.constant - shape(n_x,n_y,n_z,n_c,n_c)
    # v_r: reflected voltages , tf.constant - shape(n_x,n_y,n_z,n_c)

    #calculate the reflected fields
    f_r = tf.einsum('ijkm,ijknm->ijkn',v_i,r_1_t) - 0.5*v_f

    #seperate out electric and magnetic reflected fields
    f_r_e = f_r[:,:,:,0:3]
    f_r_m = f_r[:,:,:,3:6]

    #calculate total fields 
    f_e,s_pre,x_1_pre,x_2_pre = TRANSMISSION(f_r_e,inf_x,s_pre,x_1_pre,x_2_pre,w_0,damp,del_x,del_t)
    f_m = 0.5*f_r_m

    #recombine total fields
    f = tf.concat([f_e,f_m],3)

    #calculate reflected voltages
    v_r = tf.einsum('ijkm,ijknm->ijkn',f,r) - tf.einsum('ijkm,ijknm->ijkn',v_i,p)

    return v_r,s_pre,x_1_pre,x_2_pre,f

def MULTIPLE_SCATTER(v_i,v_f,r_1_t,r,p,x,tra_ope,s_e_pre,sta_ope):
    # scatters the incident voltages and free-source fields into reflected
    # voltages for a Lorentz model with multiple resonances
    # v_i: incident voltages , tf.constant - shape(n_x,n_y,n_z,n_c)
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f)
    # r_1_t: converts incident voltages to reflected fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_c)
    # r: converts total fields to reflected voltages , tf.constant - shape(n_x,n_y,n_z,n_c,n_f)
    # p: flips incidnet voltages , tf.constant - shape(n_x,n_y,n_z,n_c,n_c)
    # x: state variable tensor - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)
    # tra_ope: a list of transmission operators - list, shape (2,)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    # sta_ope: a list of state operators - shape(4,)
    #
    # v_r: reflected voltages , tf.constant - shape(n_x,n_y,n_z,n_c)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    # x_next: the state variable matrix value at the next time step - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)
    # f: total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)

    #calculate the reflected fields
    f_r = tf.einsum('ijkm,ijknm->ijkn',v_i,r_1_t) - 0.5*v_f

    #seperate out electric and magnetic reflected fields
    f_r_e = f_r[:,:,:,0:3]
    f_r_m = f_r[:,:,:,3:6]

    #calculate total fields 
    f_e,s_e_pre,x_next = MULTIPLE_TRANSMISSION(f_r_e,sta_ope,x,tra_ope[0],tra_ope[1],s_e_pre)
    f_m = 0.5*f_r_m

    #recombine total fields
    f = tf.concat([f_e,f_m],3)

    #calculate reflected voltages
    v_r = tf.einsum('ijkm,ijknm->ijkn',f,r) - tf.einsum('ijkm,ijknm->ijkn',v_i,p)

    return v_r,s_e_pre,x_next,f

# -------------------------- Propagation Operation ----------------------#
def PROPAGATE(v_f,inf_x,w_0,damp,del_x,del_t,n_c,n_t):
    # propagte the voltages (both scattering and transfer)
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_t)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z,n_f)
    # w_0: Lorentz resonance frequency tensor - np.constant, shape(n_x,n_y,n_z,n_f)
    # damp: Lorentz damping frequency tensor - np.constant, shape(n_x,n_y,n_z,n_f)
    # del_x: change in susceptibility tensor - np.constant, shape(n_x,n_y,n_z,n_f)
    # del_t: change in time - np.constant, shape(1)
    # n_c: number of voltage components
    # n_t: number of time steps

    #get spatial steps and number of field components
    n_x,n_y,n_z,n_f = np.shape(inf_x)
    n_f = 2*n_f

    #produce constant tensors for scattering
    r_1_t,r,p,boundary = CONSTANT_TENSORS(inf_x,n_c,n_f)

    #initial conditions
    s_pre = 0*v_f[:,:,:,0:3,0]
    x_1_pre = s_pre
    x_2_pre = x_1_pre

    #initilize field
    f_time = np.zeros((n_x,n_y,n_z,n_f,0),dtype = np.float32)
    v_i = tf.zeros((n_x,n_y,n_z,n_c),dtype = np.float32)

    for t in range(n_t):

        v_r,s_pre,x_1_pre,x_2_pre,f = SCATTER(v_i,v_f[:,:,:,:,t],r_1_t,r,p,inf_x,s_pre,x_1_pre,x_2_pre,w_0,damp,del_x,del_t)    

        v_i = TRANSFER(v_r)

        f_tmp = tf.reshape(f,(n_x,n_y,n_z,n_f,1))
        f_time = tf.concat( [f_time,f_tmp] , 4 )

    return v_i , f_time , f_tmp

def MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,del_t,n_c,n_t,n_f):
    # propagte the voltages for a multiple resonant lorentz model (both scattering and transfer)
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_t)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # w_0: Lorentz resonance frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # damp: Lorentz damping frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # del_x - the change in susceptibility (unitless) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # del_t - the time step of the simulation (s) - np.float32, shape(1,)
    # n_c: number of voltage components - int, shape(1,)
    # n_t: number of time steps - int, shape(1,)
    # n_f: number of field components at each point in space - int, shape(1,)
    #
    # v_i: voltage incident - tf.constant, shape(n_x,n_y,n_z,n_c)
    # f_time: total fields for all time - tf.constant, shape(n_x,n_y,n_z,n_f,n_t)
    # f_final: total fields at last time step - tf.constant, shape(n_x,n_y,n_z,n_f)
    

    #get spatial steps and number of resonances
    n_x,n_y,n_z,n_r = np.shape(w_0)

    #determine the electrical dispersion state variable operators
    sta_ope , tra_ope = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)

    #produce constant tensors for scattering
    r_1_t,r,p,boundary = CONSTANT_TENSORS(inf_x,n_c,n_f)

    #initial conditions of total electric accumulator and state variable
    s_e_pre = tf.zeros([n_x,n_y,n_z,n_f//2],dtype = data_type)
    x = tf.zeros([n_x,n_y,n_z,n_r*2,n_f//2],dtype = data_type)

    #initilize field
    f_time = np.zeros((n_x,n_y,n_z,n_f,0),dtype = np.float32)
    v_i = tf.zeros((n_x,n_y,n_z,n_c),dtype = np.float32)

    for t in range(n_t):

        v_r,s_e_pre,x,f = MULTIPLE_SCATTER(v_i,v_f[:,:,:,:,t],r_1_t,r,p,x,tra_ope,s_e_pre,sta_ope)    

        v_i = TRANSFER(v_r)

        f_tmp = tf.reshape(f,(n_x,n_y,n_z,n_f,1))
        f_time = tf.concat( [f_time,f_tmp] , 4 )

    f_final = f_tmp

    return v_i , f_time , f_final

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





