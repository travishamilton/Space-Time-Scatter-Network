import numpy as np
import tensorflow as tf
from numpy.polynomial import polynomial as p
import pickle
import matplotlib.pyplot as plt
from weights import WEIGHT_INDEXING

data_type = np.float32

alpha = 1000

# ---------------------------------------------------------------------- #
###         Dispersion Parameters

def NL_ELECTRIC_DISPERSION_OPERATORS(res_freq,damp,del_x,del_t,inf_x,x_nl):
    # converts the electrical dispersion and non-linear physical parameters into relevent state and transmission operators
    # res_freq - resonant frequency (r/s) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # damp - damping coefficient (r/s) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # del_x - the change in susceptibility (unitless) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # del_t - the time step of the simulation (s) - np.float32, shape(1,)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # x_nl: non-linear susceptibility - np.constant, shape(n_x,n_y,n_z)
    #
    # sta_ope: a list of state operators - shape(4,)
    # tra_ope: a list of transmission operators - shape(3,)

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
    t = np.stack((t,t,t),axis = -1)
    k = -(2-2*inf_x)
    k = np.stack((k,k,k),axis = -1)
    x_nl = np.stack((x_nl,x_nl,x_nl),axis = -1)

    #convert everything to tensors
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    c = tf.convert_to_tensor(c)
    d = tf.convert_to_tensor(d)
    t = tf.convert_to_tensor(t)
    k = tf.convert_to_tensor(k)

    #package operators
    sta_ope = [a,b,c,d]
    tra_ope = [t,k,x_nl]

    return sta_ope , tra_ope

def NL_ELECTRIC_DISPERSION_OPERATORS_TRAIN(res_freq,damp,del_x,del_t,inf_x,x_nl,weights,slope):
    # converts the electrical dispersion and non-linear physical parameters into relevent state and transmission operators
    # res_freq - resonant frequency (r/s) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # damp - damping coefficient (r/s) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # del_x - the change in susceptibility (unitless) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # del_t - the time step of the simulation (s) - np.float32, shape(1,)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # x_nl: non-linear susceptibility - np.constant, shape(n_x,n_y,n_z)
    # weights: (n_x,n_y,n_z)
    # slope: the slope of the sigmoid weight function when weights equal to zero, shape(1,)
    #
    # sta_ope: a list of state operators - shape(4,)
    # tra_ope: a list of transmission operators - shape(3,)

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
    t = np.stack((t,t,t),axis = -1)
    k = -(2-2*inf_x)
    k = np.stack((k,k,k),axis = -1)
    x_nl = np.stack((x_nl,x_nl,x_nl),axis = -1)

    #convert everything to tensors
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    c = tf.convert_to_tensor(c)
    d = tf.convert_to_tensor(d)
    t = tf.convert_to_tensor(t)
    k = tf.convert_to_tensor(k)

    #multiply weights by sigmoid function to round them to either 1 or zero, then multiply them by the c and d coefficients
    c = tf.einsum('ijkl,ijk->ijkl',c,tf.sigmoid(2*slope*weights))
    d = d*tf.sigmoid(2*slope*weights)

    #package operators
    sta_ope = [a,b,c,d]
    tra_ope = [t,k,x_nl]

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

                #a = num_coeff[x,y,z,:,:]
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
###         Non-linear Operations and Parameters

def CHI_2_NON_LINEAR(x_nl,t,u,f_pre):
    # solve the non-linear chi 2 update equation for f
    # x_nl: the non-linear second order susceptibility - np.constant, shape(n_x,n_y,n_z,n_f/2)
    # t: constant tensor operating on f , s_pre and f**2 to update f - shape(n_x,n_y,n_z,n_f/2)
    # u: forcing function of non-linear f update equation - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # f_pre: the previous total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    #
    # f: total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)

    
    #solve update equation for both roots
    f_0 =  ( 1 + tf.sqrt(1-8.0*t*x_nl*u) ) / ( 4.0*t*x_nl )
    f_1 = ( 1 - tf.sqrt(1-8.0*t*x_nl*u) ) / ( 4.0*t*x_nl )

    #determine perturbation
    f_per_0 = f_0 - f_pre
    f_per_1 = f_1 - f_pre
    f_per = tf.stack([f_per_0,f_per_1],axis=-1)
    f_per = tf.reduce_min(f_per,axis = -1)

    #assign value with smallest perturbation
    f = f_per+f_pre

    return f

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

def SPECTRUM_Z(f_time,del_t,n_t,freq_1_start,freq_1_end,freq_2_start,freq_2_end):
    # determines the spectrum of the z-polarized field along all space
    # fig_num: the figure number
    # f_time: the field over all time and space - shape(n_t,n_x,n_y,n_z,n_f)
    # del_t: the time step (s) - shape(1,)
    # n_t: number of time steps - shape(1,)
    # freq_1: the first frequency value to calculate (Hz)
    # freq_2: the second frequency value to calculate (Hz)
    #
    # sp_1: the spectrum over the y - axis for freq_1 - shape(n_y,n_f_1)
    # sp_2: the spectrum over the y - axis for freq_2 - shape(n_y,n_f_2)

    # get the z polarized time signal
    time_signal = f_time[:,:,:,2,:]

    # time values
    t = np.arange(0,del_t*n_t,del_t)

    #calculate spectrum
    sp = tf.fft(time_signal)
    freq = np.fft.fftfreq(t.shape[-1])/del_t
    l = len(freq)
     
    # only keep positive frequency values
    freq = freq[0:l//2-1]
    sp = sp[:,:,:,0:l//2-1]

    # get frequency step size
    df = freq[1] - freq[0]

    # get correct frequency
    begin_1,size_1 = FIND_CLOSEST(freq ,[freq_1_start,freq_1_end])
    begin_2,size_2 = FIND_CLOSEST(freq ,[freq_2_start,freq_2_end])

    #give spectrum vs. space for frequencty values
    sp_1 = sp[0,:,0,begin_1:begin_1+size_1]
    sp_2 = sp[0,:,0,begin_2:begin_2+size_1]

    return sp_1 , sp_2 , df

def TRAPZ(f,a,b,n):
    #calculates the definite integral using the trapezoidal rule along the first axis
    # f: function values - tf.constant, tf.float32, shape(n+1)
    # a: lower limit of definite integral - shape(1,)
    # b: upper limit of definite integral - shape(1,)
    # n: number of function points - shape(1,)
    #
    # output: approximate value of the definite integral of f from a to b

    #spatial step
    del_x = (b-a)/n

    #trapezoidal rule
    output = ( del_x/2 ) * ( f[0] + f[n] + 2.0*tf.reduce_sum(f[1:n]) )

    return output

def TRAPZ_2D(f,del_x,del_y):
    #calculates the definite integral using the trapezoidal rule along 2 dimensions
    # f: function values - tf.constant, tf.float32, shape(n+1,n_f+1)
    # del_x: step size in x - shape(1,)
    # del_y: step size in y - shape(1,)
    #
    # output: approximate value of the definite integral of f over all x and y

    #trapezoidal rule
    s1 = (del_x*del_y/4)*(f[0,0] + f[0,-1] + f[-1,0] + f[-1,-1])

    s2 = (del_x*del_y/2)*(tf.reduce_sum(f[1:-1,0]) + tf.reduce_sum(f[1:-1,-1]) + tf.reduce_sum(f[0,1:-1]) + tf.reduce_sum(f[-1,1:-1]))

    s3 = del_x*del_y*tf.reduce_sum(f)

    #sum components
    output = s1+s2+s3
    

    return output

def OVERLAP_INTEGRAL(f_1,f_2,del_x,del_freq):
    # calculates the overlap integral between to frequency modes for a list of frequency pairs
    # f_1: field 1 containing the spatial and frequency points of interest - tf.complex, shape(n+1,n_f+1)
    # f_2: field 2 containing the spatial and frequency points of interest - tf.complex, shape(n+1,n_f+1)
    # a: lower limit of region of interest - shape(1,)
    # b: upper limit of region of interest - shape(1,)
    # n: number of function points over space - shape(1,)
    # n_f: number of frequency - shape(1,)
    #
    # output: normalized overalp integral over space and frequency pairs - shape(n_f,)

    # mode product
    m_p = tf.multiply( tf.conj(f_1) , f_2 )

    # overlap integral 
    top = tf.abs( tf.complex( TRAPZ_2D(tf.real(m_p),del_x,del_freq) , TRAPZ_2D(tf.imag(m_p),del_x,del_freq) ) )**2

    # normalizing coefficients
    bottom = TRAPZ_2D(tf.abs(f_1)**2,del_x,del_freq) * TRAPZ_2D(tf.abs(f_2)**2,del_x,del_freq)

    # normalized overlap integral
    output = top/bottom

    return output

def NONLINEAR_OVERLAP_INTEGRAL(f_1,f_2,del_x,del_freq,weights,slope):
    # calculates the nonlinear overlap integral between to frequency modes for a list of frequency pairs
    # f_1: field 1 (pump) containing the spatial and frequency points of interest - tf.complex, shape(n+1,n_f+1)
    # f_2: field 2 (signal) containing the spatial and frequency points of interest - tf.complex, shape(n+1,n_f+1)
    # del_x: step size in space - shape(1,)
    # del_freq: step size in frequency - shape(1,)
    # weights: weight tensor over the mask - shape(n+1,)
    #
    # output: normalized nonllinear overalp integral over space and frequency pairs - shape(n_f,)

    # mode product
    m_p = tf.multiply(tf.multiply( tf.sigmoid(2*slope*tf.reshape(weights,(-1,1))) , f_1**2)  , tf.conj(f_2) )
    
    # overlap integral 
    top = tf.abs( tf.complex( TRAPZ_2D(tf.real(m_p),del_x,del_freq) , TRAPZ_2D(tf.imag(m_p),del_x,del_freq) ) )

    # normalizing coefficients
    bottom = TRAPZ_2D(tf.abs(f_1)**2,del_x,del_freq) * tf.sqrt( TRAPZ_2D(tf.abs(f_2)**2,del_x,del_freq) )

    # normalized overlap integral
    output = top/bottom

    return output

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

def MULTIPLE_CONSTANT_TENSORS(w_0,n_c,n_f):
    # produces the constant tensors used to scatter the fields and construct the boundary tensor
    # n_c: number of voltage components - shape(1,)
    # n_f: number of field components - shape(1,)

    #initilize matrices
    r_1_t_matrix , r_matrix , p_matrix = CONSTANT_MATRICES()

    #spatial parameters
    n_x,n_y,n_z,_ = np.shape(w_0)

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
                if y == 0 or y == n_y - 1:
                      boundary[x,y,z,:] = np.zeros(n_c,dtype = data_type)
   
    return tf.convert_to_tensor(r_1_t) , tf.convert_to_tensor(r) , tf.convert_to_tensor(p) , tf.convert_to_tensor(boundary)

def FIND_CLOSEST(tensor,value):
    # find the indicies in the tensor associated with the given value range
    # tensor - 1D tensor
    # value - a pair of values (highest,lowest) to be indexed

    # get first and last index value
    begin = tf.argmin(tf.abs(tensor - value[0]))
    last = tf.argmin(tf.abs(tensor - value[1]))

    # get size
    size = last - begin + 1

    return begin , size

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
###         Scattering Operations

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

def NL_MULTIPLE_TRANSMISSION(f_r,sta_ope,x,t,k,x_nl,s_e_pre,f_pre):
    # transmitts the reflected fields into the total fields for a non-linear multiple 
    # resonance Lorentz model
    # f_r: reflected fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # sta_ope: a list of state operators - shape(4,)
    # x: state variable tensor - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)
    # t: constant tensor operating on f_r and s_pre to update f - shape(n_x,n_y,n_z,n_f/2)
    # k: constant tensor operating on f to update s - shape(n_x,n_y,n_z,n_f/2)
    # x_nl: the non-linear second order susceptibility - np.constant, shape(n_x,n_y,n_z,n_f/2)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    # f_pre: previous time step total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    #
    # f: total fields - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    # x_next: the state variable matrix value at the next time step - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)

    #calculate the forcing function
    u = tf.multiply((f_r + s_e_pre), t)

    #update f
    #f = CHI_2_NON_LINEAR(x_nl,t,u,f_pre)
    f = u

    #calculate dielectric accumulator in x
    s_e_d_x , x_next_x = MULTIPLE_LORENTZ(f[:,:,:,0],x[:,:,:,:,0],sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])

    #calculate dielectric accumulator in y
    s_e_d_y , x_next_y = MULTIPLE_LORENTZ(f[:,:,:,1],x[:,:,:,:,1],sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])

    #calculate dielectric accumulator in z
    s_e_d_z , x_next_z = MULTIPLE_LORENTZ(f[:,:,:,2],x[:,:,:,:,2],sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])

    #combine x,y and z components
    s_e_d = tf.stack((s_e_d_x,s_e_d_y,s_e_d_z),axis = -1)
    x_next = tf.stack((x_next_x,x_next_y,x_next_z),axis = -1)

    #calculate the main accumulator
    s_e_pre = f_r + tf.multiply(k,f) - tf.multiply(2*x_nl,f**2) + s_e_d

    return f , s_e_pre , x_next

def NL_MULTIPLE_SCATTER(v_i,v_f,r_1_t,r,p,x,tra_ope,s_e_pre,sta_ope,f_pre):
    # scatters the incident voltages and free-source fields into reflected
    # voltages for a non-linear Lorentz model with multiple resonances
    # v_i: incident voltages , tf.constant - shape(n_x,n_y,n_z,n_c)
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f)
    # r_1_t: converts incident voltages to reflected fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_c)
    # r: converts total fields to reflected voltages , tf.constant - shape(n_x,n_y,n_z,n_c,n_f)
    # p: flips incidnet voltages , tf.constant - shape(n_x,n_y,n_z,n_c,n_c)
    # x: state variable tensor - np.constant, shape (n_x,n_y,n_z,# of state variables,n_f/2)
    # tra_ope: a list of transmission operators - list, shape (3,)
    # s_e_pre: previous total electric accumulator tensor - shape(n_x,n_y,n_z,n_f/2)
    # sta_ope: a list of state operators - shape(4,)
    # f: previous total fields - np.constant, shape (n_x,n_y,n_z,n_f)
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
    f_e_pre = f_pre[:,:,:,0:3]

    #calculate total fields 
    f_e,s_e_pre,x_next = NL_MULTIPLE_TRANSMISSION(f_r_e,sta_ope,x,tra_ope[0],tra_ope[1],tra_ope[2],s_e_pre,f_e_pre)

    f_m = 0.5*f_r_m

    #recombine total fields
    f = tf.concat([f_e,f_m],3)

    #calculate reflected voltages
    v_r = tf.einsum('ijkm,ijknm->ijkn',f,r) - tf.einsum('ijkm,ijknm->ijkn',v_i,p)

    return v_r,s_e_pre,x_next,f

# ---------------------------------------------------------------------- #
###         Propagation Operation

def NL_MULTIPLE_PROPAGATE(v_f,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f):
    # propagte the voltages for a non-linear multiple resonant lorentz model (both scattering and transfer)
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_t)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # w_0: Lorentz resonance frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # damp: Lorentz damping frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # del_x - the change in susceptibility (unitless) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # x_nl: non-linear susceptibility - np.constant, shape(n_x,n_y,n_z)
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
    sta_ope , tra_ope = NL_ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x,x_nl)

    #produce constant tensors for scattering
    r_1_t,r,p,boundary = MULTIPLE_CONSTANT_TENSORS(w_0,n_c,n_f)

    #initial conditions of total electric accumulator and state variable
    s_e_pre = tf.zeros([n_x,n_y,n_z,n_f//2],dtype = data_type)
    x = tf.zeros([n_x,n_y,n_z,n_r*2,n_f//2],dtype = data_type)

    #initilize field
    f_time = np.zeros((n_x,n_y,n_z,n_f,0),dtype = np.float32)
    v_i = tf.zeros([n_x,n_y,n_z,n_c],dtype = np.float32)
    f = tf.zeros([n_x,n_y,n_z,n_f],dtype = np.float32)

    for t in range(n_t):

        v_r,s_e_pre,x,f = NL_MULTIPLE_SCATTER(v_i,v_f[:,:,:,:,t],r_1_t,r,p,x,tra_ope,s_e_pre,sta_ope,f)    

        v_i = TRANSFER(v_r)

        f_tmp = tf.reshape(f,(n_x,n_y,n_z,n_f,1))
        f_time = tf.concat( [f_time,f_tmp] , 4 )

    f_final = f_tmp

    return v_i , f_time , f_final

def NL_MULTIPLE_PROPAGATE_TRAIN(v_f,inf_x,w_0,damp,del_x,x_nl,del_t,n_c,n_t,n_f,weights,slope):
    # propagte the voltages for a non-linear multiple resonant lorentz model (both scattering and transfer) with trained weights
    # v_f: free-source fields , tf.constant - shape(n_x,n_y,n_z,n_f,n_t)
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_x,n_y,n_z)
    # w_0: Lorentz resonance frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # damp: Lorentz damping frequency tensor (rad/s) - np.constant, shape(n_x,n_y,n_z,n_r)
    # del_x - the change in susceptibility (unitless) - np.float32, np.array, shape (n_x,n_y,n_z,n_r)
    # x_nl: non-linear susceptibility - np.constant, shape(n_x,n_y,n_z)
    # del_t - the time step of the simulation (s) - np.float32, shape(1,)
    # n_c: number of voltage components - int, shape(1,)
    # n_t: number of time steps - int, shape(1,)
    # n_f: number of field components at each point in space - int, shape(1,)
    # weights: material index for each position - shape(n_x,n_y,n_z)
    # slope: the slope of the weight sigmoid function at 0 - shape(1,)
    #
    # v_i: voltage incident - tf.constant, shape(n_x,n_y,n_z,n_c)
    # f_time: total fields for all time - tf.constant, shape(n_x,n_y,n_z,n_f,n_t)
    # f_final: total fields at last time step - tf.constant, shape(n_x,n_y,n_z,n_f)
    
    #get spatial steps and number of resonances
    n_x,n_y,n_z,n_r = np.shape(w_0)
    
    #determine the electrical dispersion state variable operators
    sta_ope , tra_ope = NL_ELECTRIC_DISPERSION_OPERATORS_TRAIN(w_0,damp,del_x,del_t,inf_x,x_nl,weights,slope)

    #produce constant tensors for scattering
    r_1_t,r,p,boundary = MULTIPLE_CONSTANT_TENSORS(w_0,n_c,n_f)

    #initial conditions of total electric accumulator and state variable
    s_e_pre = tf.zeros([n_x,n_y,n_z,n_f//2],dtype = data_type)
    x = tf.zeros([n_x,n_y,n_z,n_r*2,n_f//2],dtype = data_type)

    #initilize field
    f_time = np.zeros((n_x,n_y,n_z,n_f,0),dtype = np.float32)
    v_i = tf.zeros([n_x,n_y,n_z,n_c],dtype = np.float32)
    f = tf.zeros([n_x,n_y,n_z,n_f],dtype = np.float32)

    for t in range(n_t):

        v_r,s_e_pre,x,f = NL_MULTIPLE_SCATTER(v_i,v_f[:,:,:,:,t],r_1_t,r,p,x,tra_ope,s_e_pre,sta_ope,f)    

        v_i = TRANSFER(v_r)

        #perform boundary condition
        v_i = v_i*boundary

        f_tmp = tf.reshape(f,(n_x,n_y,n_z,n_f,1))
        f_time = tf.concat( [f_time,f_tmp] , 4 )

    f_final = f_tmp

    return v_i , f_time , f_final