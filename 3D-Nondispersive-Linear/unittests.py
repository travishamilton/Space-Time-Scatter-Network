import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import ss2tf

import tensorflow as tf

#from layers_3 import CONSTANT_TENSORS , LORENTZ_TRANSFER_FUNCTION , STATE_OPERATORS , MULTIPLE_LORENTZ , ELECTRIC_DISPERSION_OPERATORS , LORENTZ , CHI_2_NON_LINEAR, SUM_RATIONAL_POLY , MULTIPLE_TRANSMISSION , MULTIPLE_SCATTER , MULTIPLE_LORENTZ_2

from layers import SPECTRUM_Z  , TRAPZ

data_type = np.float32

# ----------------------- numpy functions------------------------ #
def REFLECTED_FIELD_NUMPY(x_inf,v_i,v_f,n_f):

    n_x,n_y,n_z,_ = np.shape(x_inf)
    f_r = np.zeros((n_x,n_y,n_z,n_f),dtype = np.float32)

    f_r[:,:,:,0:1] = v_i[:,:,:,0:1] + v_i[:,:,:,1:2] + v_i[:,:,:,2:3] + v_i[:,:,:,3:4] - 0.5*v_f[:,:,:,0:1]
    f_r[:,:,:,1:2] = v_i[:,:,:,4:5] + v_i[:,:,:,5:6] + v_i[:,:,:,6:7] + v_i[:,:,:,7:8] - 0.5*v_f[:,:,:,1:2]
    f_r[:,:,:,2:3] = v_i[:,:,:,8:9] + v_i[:,:,:,9:10] + v_i[:,:,:,10:11] + v_i[:,:,:,11:12] - 0.5*v_f[:,:,:,2:3]
    f_r[:,:,:,3:4] = -(v_i[:,:,:,6:7] - v_i[:,:,:,7:8] - v_i[:,:,:,8:9] + v_i[:,:,:,9:10]) - 0.5*v_f[:,:,:,3:4]
    f_r[:,:,:,4:5] = -(v_i[:,:,:,10:11] - v_i[:,:,:,11:12] - v_i[:,:,:,0:1] + v_i[:,:,:,1:2]) - 0.5*v_f[:,:,:,4:5]
    f_r[:,:,:,5:6] = -(v_i[:,:,:,2:3] - v_i[:,:,:,3:4] - v_i[:,:,:,4:5] + v_i[:,:,:,5:6]) - 0.5*v_f[:,:,:,5:6]

    return f_r

def REFLECTED_VOLTAGES_NUMPY(x_inf,v_i,f,n_c):

    n_x,n_y,n_z,_ = np.shape(x_inf)
    v_r = np.zeros((n_x,n_y,n_z,n_c))

    v_r[:,:,:,0:1] = f[:,:,:,0:1] - f[:,:,:,4:5] - v_i[:,:,:,1:2]
    v_r[:,:,:,1:2] = f[:,:,:,0:1] + f[:,:,:,4:5] - v_i[:,:,:,0:1]
    v_r[:,:,:,2:3] = f[:,:,:,0:1] + f[:,:,:,5:6] - v_i[:,:,:,3:4]
    v_r[:,:,:,3:4] = f[:,:,:,0:1] - f[:,:,:,5:6] - v_i[:,:,:,2:3]
    v_r[:,:,:,4:5] = f[:,:,:,1:2] - f[:,:,:,5:6] - v_i[:,:,:,5:6]
    v_r[:,:,:,5:6] = f[:,:,:,1:2] + f[:,:,:,5:6] - v_i[:,:,:,4:5]
    v_r[:,:,:,6:7] = f[:,:,:,1:2] + f[:,:,:,3:4] - v_i[:,:,:,7:8]
    v_r[:,:,:,7:8] = f[:,:,:,1:2] - f[:,:,:,3:4] - v_i[:,:,:,6:7]
    v_r[:,:,:,8:9] = f[:,:,:,2:3] - f[:,:,:,3:4] - v_i[:,:,:,9:10]
    v_r[:,:,:,9:10] = f[:,:,:,2:3] + f[:,:,:,3:4] - v_i[:,:,:,8:9]
    v_r[:,:,:,10:11] = f[:,:,:,2:3] + f[:,:,:,4:5] - v_i[:,:,:,11:12]
    v_r[:,:,:,11:12] = f[:,:,:,2:3] - f[:,:,:,4:5] - v_i[:,:,:,10:11]

    return v_r

def TENSOR_MULTIPLICATION_NUMPY(matrix_tensor,vector_tensor_1,vector_tensor_2,constant):

    n_x,n_y,n_z,n_1,n_2 = np.shape(matrix_tensor)
    n_x,n_y,n_z,n_2 = np.shape(vector_tensor_1)

    results_tensor_1 = np.zeros((n_x,n_y,n_z,n_1))
    results_tensor_2 = np.zeros((n_x,n_y,n_z))
    results_tensor_3 = np.zeros((n_x,n_y,n_z,n_2))

    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                results_tensor_1[x,y,z,:] = matrix_tensor[x,y,z,:,:] @ vector_tensor_1[x,y,z,:]
                results_tensor_2[x,y,z] = vector_tensor_1[x,y,z,:] @ vector_tensor_2[x,y,z,:]
                results_tensor_3[x,y,z,:] = vector_tensor_2[x,y,z,:]*constant[x,y,z]

    return results_tensor_1 , results_tensor_2 , results_tensor_3

def MULTIPLE_LORENTZ_NUMPY(f,x,a,b,c,d):
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
    
    #get shape parameters
    n_x,n_y,n_z,n_o = np.shape(x)
    x = x[:,:,:,:,np.newaxis]

    #initilize
    x_next = np.zeros((n_x,n_y,n_z,n_o,1),dtype = data_type)
    s_e_d = np.zeros((n_x,n_y,n_z),dtype = data_type)

    #calculate
    for xi in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                    x_next[xi,y,z,:,0] = a[xi,y,z,:,:]@x[xi,y,z,:,0] + b[xi,y,z,:]*f[xi,y,z]
                    s_e_d[xi,y,z] = c[xi,y,z,:]@x[xi,y,z,:,0] + d[xi,y,z]*f[xi,y,z]

    return s_e_d , x_next[:,:,:,:,0]

def MULTIPLE_LORENTZ_SINGLE_INPUT_NUMPY(f,x,a,b,c,d):
    # calculates an electrical dielectric accumulator for Lorentz model with multiple resonances
    # f: total fields - np.constant, shape (1,)
    # x: state variable tensor - np.constant, shape (n_o,)
    # a: constant tensor operating on x to update x - shape(n_o,n_o)
    # b: constant tensor operating on f to update x - shape(n_o,)
    # c: constant tensor operating on x to update s_e_d - shape(n_o,)
    # d: constant tensor operating on f to update s_e_d - shape(1,)
    #
    # s_e_d: the electrical dielectric accumulator - shape(1,)
    # x_next: state variable tensor at the next time step - shape (n_o,)
    
    #get shape parameters
    # x = x[:,np.newaxis]
        
    # x_next = a@x + b*f
    # s_e_d = c@x + d*f

    #calculate state variable
    x_next = np.einsum('mn,n->m',a,x) + b*f

    #calculate the electrical dielectric accumulator
    s_e_d = np.tensordot(c,x,[[0],[0]]) + d*f

    #return s_e_d,x_next[:,0]
    return s_e_d,x_next

def TRANSFER_FUNCTION_VARIABLES(w_0,damp,del_x,del_t):
    # calculates the transfer function variables
    # w_0: resonant frequency tensors  - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # damp: damping frequency tensors - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # del_x: change in susceptibility tensors - np.constant, shape (n_x,n_y,n_z,n_f/2)
    # del_t: the step size in timme (s) - np.constant, shape(1)
    #
    # a_1: shape(n_x,n_y,n_z,n_f/2)

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

    return a_1,a_2,b_1,b_2

def SUM_POLY_NUMPY(num,den,x):
    #sums two polynomials and evalutes them at x
    #num - shape(n_p,n_o+1)
    #den - shape(n_p,n_o+1)

    #get number of polynomials and order
    n_p,n_o_tmp = np.shape(num)
    n_o = n_o_tmp - 1

    #set results to zero
    poly_results = 0

    for p in range(n_p):
        num_results = 0
        den_results = 0
        for o in range(n_o+1):
            num_results = num[p,o]*x**(n_o-o) + num_results
            den_results = den[p,o]*x**(n_o-o) + den_results

        poly_results = num_results/den_results + poly_results

    return poly_results

def PERCENT_ERROR(A,B):

    difference = np.amax( np.abs( (A - B) ) )
    
    return difference

def LORENTZ_LINEARITY(U,X,A,B,C,D,n_t):
    X = tf.convert_to_tensor(X)
    X = X[:,:,:,:,0]
    for t in range(n_t):
        Y , X_next = MULTIPLE_LORENTZ(U[:,:,:,0,t],X,A,B,C,D)
        X = X_next

    return Y,X

def LORENTZ_LINEARITY_NUMPY(U,X,A,B,C,D,n_t):
    X = X[:,:,:,:,0]
    for t in range(n_t):
        Y , X_next = MULTIPLE_LORENTZ_NUMPY(U[:,:,:,0,t],X,A,B,C,D)
        X = X_next

    return Y,X

# ----------------------- unittest class ------------------------ #
class Test(unittest.TestCase):

    def test_einsum_check(self):

        success = True

        for _ in range(10):
            #position and field arguments
            n_c = 12
            n_f = 6
            n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        
            #build tensors
            matrix_tensor_np = np.float32(np.random.rand(n_x,n_y,n_z,n_c,n_f))
            vector_tensor_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
            vector_tensor_2_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
            constant_np = np.float32(np.random.rand(n_x,n_y,n_z))

            vector_np = np.float32(np.random.rand(n_c))
            matrix_np = np.float32(np.random.rand(n_c,n_c))

            matrix_tensor_tf = tf.convert_to_tensor(matrix_tensor_np,dtype = np.float32)
            vector_tensor_tf = tf.convert_to_tensor(vector_tensor_np,dtype = np.float32)
            vector_tensor_2_tf = tf.convert_to_tensor(vector_tensor_2_np,dtype = np.float32)
            constant_tf = tf.convert_to_tensor(constant_np,dtype = np.float32)
            vector_tf = tf.convert_to_tensor(vector_np,dtype = np.float32)
            matrix_tf = tf.convert_to_tensor(matrix_np,dtype = np.float32)

            #calculate einsum in both numpy and tensorflow using einsum and for loops
            results_1_np = np.einsum('ijkmn,ijkn -> ijkm',matrix_tensor_np,vector_tensor_np)
            results_2_np , results_3_np , results_4_np = TENSOR_MULTIPLICATION_NUMPY(matrix_tensor_np,vector_tensor_np,vector_tensor_2_np,constant_np)
            results_5_np = np.matmul(matrix_np,vector_np)
            results_tf = tf.einsum('ijkmn,ijkn -> ijkm',matrix_tensor_tf,vector_tensor_tf)
            results_2_tf = tf.einsum('ijkn,ijkmn -> ijkm',vector_tensor_tf,matrix_tensor_tf)
            results_3_tf = tf.einsum('ijko,ijko->ijk',vector_tensor_tf,vector_tensor_2_tf)
            results_4_tf = tf.einsum('ijko,ijk->ijko ',vector_tensor_2_tf,constant_tf)
            results_5_tf = tf.einsum('mn,n->m',matrix_tf,vector_tf)

            #convert tensorflow results back to numpy
            with tf.Session() as sess:

                results_tf = sess.run(results_tf)
                results_2_tf = sess.run(results_2_tf)
                results_3_tf = sess.run(results_3_tf)
                results_4_tf = sess.run(results_4_tf)
                results_5_tf = sess.run(results_5_tf)

                #compare tensorflow and with numpy results
                success = np.amax(results_1_np-results_tf) < 5.0e-6 and np.amax(results_2_np-results_1_np) < 5.0e-6 and np.amax(results_tf - results_2_tf) < 5.0e-6 and np.amax(results_3_tf - results_3_np) < 5.0e-6 and np.amax(results_4_tf - results_4_np) < 5.0e-6 and np.amax(results_5_tf - results_5_np) < 5.0e-6 and success

        self.assertEqual(success, True)

    def test_tensordot_check(self):

        success = True

        for _ in range(10):

            #numpy calculation
            n_c = 12
            vector_tensor_1_np = np.float32(np.random.rand(n_c))
            vector_tensor_2_np = np.float32(np.random.rand(n_c))
            result_np = np.sum(vector_tensor_1_np*vector_tensor_2_np)

            #tensorflow calculation
            vector_tensor_1 = tf.convert_to_tensor(vector_tensor_1_np)
            vector_tensor_2 = tf.convert_to_tensor(vector_tensor_2_np)
            result = tf.tensordot(vector_tensor_1,vector_tensor_2,[[0],[0]])

            with tf.Session() as sess:
                result = sess.run(result)

            success = np.amax(result-result_np) < 5.0e-6 and success

        self.assertEqual(success, True)

    def test_multiply_check(self):

        success = True

        #position and field arguments
        n_c = 12
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))

        #build random tensor/constants
        tensor_np = np.float32(np.random.rand(n_x,n_y,n_z,n_c))
        constant_np = np.float32(2.233)

        tensor_tf = tf.convert_to_tensor(tensor_np)
        constant_tf = tf.convert_to_tensor(constant_np)

        #calculate multiplication in both tensorflow and numpy
        results_np = tensor_np*constant_np
        results_tf = tensor_tf*constant_tf

        #convert tensorflow results back to numpy
        with tf.Session() as sess:

            results_tf = sess.run(results_tf)

            #compare tensorflow and with numpy results
            success = np.array_equal(results_np,results_tf) and success

        self.assertEqual(success, True)

    def test_main_accumulator(self):

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))

        #build random tensors
        f_r_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
        f_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
        k_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
        s_d_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))

        f_r_tf = tf.convert_to_tensor(f_r_np)
        f_tf = tf.convert_to_tensor(f_np)
        k_tf = tf.convert_to_tensor(k_np)
        s_d_tf = tf.convert_to_tensor(s_d_np)

        #calculate main accumulator
        s_np = 2*f_r_np + np.multiply(k_np,f_np) + s_d_np
        s_tf = 2*f_r_tf + tf.multiply(k_tf,f_tf) + s_d_tf

        #convert tensorflow results back to numpy
        with tf.Session() as sess:

            s_tf = sess.run(s_tf)

            #compare tensorflow and with numpy results
            success = np.array_equal(s_np,s_tf) and success

        self.assertEqual(success, True)

    def test_fields_reflected(self):

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))

        #build random tensors
        inf_x = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
        
        #import constant tensor
        r_1_t , _ , _ , _ = CONSTANT_TENSORS(inf_x,n_c,n_f)

        #build random tensors
        v_i_np = np.float32(np.random.rand(n_x,n_y,n_z,n_c))
        v_f_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))

        v_i_tf = tf.convert_to_tensor(v_i_np,dtype = data_type)
        v_f_tf = tf.convert_to_tensor(v_f_np,dtype = data_type)
        
        #calculte fields reflected
        f_r_tf = tf.einsum('ijkmn,ijkn->ijkm',r_1_t,v_i_tf) - np.float32(0.5)*v_f_tf
        f_r_np = REFLECTED_FIELD_NUMPY(inf_x,v_i_np,v_f_np,n_f)

        #convert tensorflow results back to numpy
        with tf.Session() as sess:

            f_r_tf = sess.run(f_r_tf)

            #compare tensorflow and with numpy results
            success = np.amax(f_r_tf-f_r_np) < 5.0e-7 and success

        self.assertEqual(success, True)

    def test_voltages_reflected(self):

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))

        #build random tensors
        inf_x = np.float32(np.random.rand(n_x,n_y,n_z,n_f))
        
        #import constant tensor
        _ , r , p , _ = CONSTANT_TENSORS(inf_x,n_c,n_f)

        #build random tensors
        v_i_np = np.float32(np.random.rand(n_x,n_y,n_z,n_c))
        f_np = np.float32(np.random.rand(n_x,n_y,n_z,n_f))

        v_i_tf = tf.convert_to_tensor(v_i_np)
        f_tf = tf.convert_to_tensor(f_np)
        
        #calculte voltages reflected
        v_r_tf = tf.einsum('ijkm,ijknm->ijkn',f_tf,r) - tf.einsum('ijkm,ijknm->ijkn',v_i_tf,p)
        v_r_np = REFLECTED_VOLTAGES_NUMPY(inf_x,v_i_np,f_np,n_c)

        #convert tensorflow results back to numpy
        with tf.Session() as sess:

            v_r_tf = sess.run(v_r_tf)

            #compare tensorflow and with numpy results
            success = np.array_equal(v_r_tf,v_r_np) and success

        self.assertEqual(success, True)

    #very time consuming test
    def test_multiple_lorentz(self):
        #test 

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_r = 1
        n_o = 2*n_r
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        del_t = 1.0e-12

        #build random tensors
        inf_x = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2))
        w_0 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))*10**12
        damp = 0.2*w_0
        del_x = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        x = np.float32(np.zeros((n_x,n_y,n_z,n_f//2,n_o)))
        f = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2))

        #convert state variable and field to tensors
        x_tf = tf.convert_to_tensor(x)
        f_tf = tf.convert_to_tensor(f)
        
        #produce operators
        sta_ope , tra_ope = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)

        #produce electrical dielectric accumulator and next state variable
        for _ in range(100):
            s_e_d , x_next = MULTIPLE_LORENTZ(f_tf,x_tf,sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])
            x_tf = x_next

        #pad tensors with zeros to include two more resonanc spots
        zero_tensor = np.float32(np.zeros((n_x,n_y,n_z,n_f//2,2)))
        w_0 = np.concatenate((w_0,zero_tensor),axis = 4)
        damp = np.concatenate((damp,zero_tensor),axis = 4)
        del_x = np.concatenate((del_x,zero_tensor),axis = 4)
        zero_tensor = np.float32(np.zeros((n_x,n_y,n_z,n_f//2,2*2)))
        x = np.concatenate((x,zero_tensor),axis = 4)

        #convert state variable and field to tensors
        x_tf = tf.convert_to_tensor(x)
        
        #produce operators
        sta_ope , tra_ope = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)

        #produce electrical dielectric accumulator and next state variable
        for _ in range(100):
            s_e_d_more_zeros , x_next_more_zeros = MULTIPLE_LORENTZ(f_tf,x_tf,sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3])
            x_tf = x_next_more_zeros
        
        with tf.Session() as sess:

            s_e_d_tf = sess.run(s_e_d)
            x_next_tf = sess.run(x_next)

            s_e_d_more_zeros_tf = sess.run(s_e_d_more_zeros)

            success = np.array_equal(np.shape(s_e_d_tf),np.array([n_x,n_y,n_z,n_f//2])) and np.array_equal(np.shape(x_next_tf),np.array([n_x,n_y,n_z,n_f/2,n_o])) and success

            print(s_e_d_tf[0,0,0,0])
            print(s_e_d_more_zeros_tf[0,0,0,0])
            

        self.assertEqual(success, True)

    def test_multiple_transmission_check(self):
        #test the MULTIPLE_TRANSMISSION for multiple resonances

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_r = 3
        n_o = 2*n_r
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        del_t = 1.0e-12

        #position and field arguments
        n_c = 12
        n_f = 6
        n_r = 3
        n_o = 2*n_r
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        del_t = 1.0e-12

        #build random tensors
        inf_x = np.float32(np.random.rand(n_x,n_y,n_z))
        w_0 = np.float32(np.random.rand(n_x,n_y,n_z,n_r))*10**12
        damp = 0.2*w_0
        del_x = np.float32(np.random.rand(n_x,n_y,n_z,n_r))
        x = 0*np.float32(np.random.rand(n_x,n_y,n_z,n_o,n_f//2))
        s_e_pre = 0*np.float32(np.random.rand(n_x,n_y,n_z,n_f//2))
        f_r = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2))

        #convert state variable and field to tensors
        x_tf = tf.convert_to_tensor(x)
        f_r_tf = tf.convert_to_tensor(f_r)
        
        #produce operators
        sta_ope , tra_ope = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)

        #calculate multiple transmission function
        f , s_e_pre , x_next = MULTIPLE_TRANSMISSION(f_r_tf,sta_ope,x_tf,tra_ope[0],tra_ope[1],s_e_pre)

        with tf.Session() as sess:

            f = sess.run(f)

            success = np.array_equal(np.shape(s_e_pre),np.array([n_x,n_y,n_z,n_f//2]))

        self.assertEqual(success, True)

    def test_single_input_lorentz_functions(self):
        success = True

        #field and simulation parameters
        n_c = 12
        n_f = 6
        n_r = 3
        n_o = 2*n_r
        del_t = 1.0e-12
        n_t = 50

        #build random tensors
        inf_x = np.float32(np.random.rand(1,1,1,1,))
        w_0 = np.float32(np.random.rand(1,1,1,n_r))*10**12
        damp = 0.2*w_0
        del_x = np.float32(np.random.rand(1,1,1,n_r))
        x = 0*np.float32(np.random.rand(n_o))

        #produce a linear combination of inputs
        alpha , beta = map(int, np.random.randint(low=1, high=50, size=2))
        f1 = tf.convert_to_tensor(np.float32(np.random.rand(n_t)))
        f2 = tf.convert_to_tensor(np.float32(np.random.rand(n_t)))
        f = tf.convert_to_tensor(alpha*f1+beta*f2)

        #produce operators
        sta_ope , tra_ope = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)
        A = sta_ope[0][0,0,0,:,:]
        B = sta_ope[1][0,0,0,:]
        C = sta_ope[2][0,0,0,:]
        D = sta_ope[3][0,0,0]

        #compute in tensorflow
        x = tf.zeros(n_o)
        for t in range(n_t):
            s_e_d1,x_next = MULTIPLE_LORENTZ_2(f1[t],x,A,B,C,D)
            x = x_next

        x = tf.zeros(n_o)
        for t in range(n_t):
            s_e_d2,x_next = MULTIPLE_LORENTZ_2(f2[t],x,A,B,C,D)
            x = x_next

        x = tf.zeros(n_o)
        for t in range(n_t):
            s_e_d,x_next = MULTIPLE_LORENTZ_2(f[t],x,A,B,C,D)
            x = x_next

        with tf.Session() as sess:

            #computer in numpy
            A_np = sess.run(A)
            B_np = sess.run(B)
            C_np = sess.run(C)
            D_np = sess.run(D)
            f1_np = sess.run(f1)
            f2_np = sess.run(f2)
            f_np = sess.run(f)

            print('Shape of A: ',np.shape(A_np))
            print('Shape of B: ',np.shape(B_np))
            print('Shape of C: ',np.shape(C_np))
            print('Shape of D: ',np.shape(D_np))

            x_np = np.zeros(n_o)
            for t in range(n_t):
                s_e_d1_np,x_next_np = MULTIPLE_LORENTZ_SINGLE_INPUT_NUMPY(f1_np[t],x_np,A_np,B_np,C_np,D_np)
                x_np = x_next_np

            x_np = np.zeros(n_o)
            for t in range(n_t):
                s_e_d2_np,x_next_np = MULTIPLE_LORENTZ_SINGLE_INPUT_NUMPY(f2_np[t],x_np,A_np,B_np,C_np,D_np)
                x_np = x_next_np

            x_np = np.zeros(n_o)
            for t in range(n_t):
                s_e_d_np,x_next_np = MULTIPLE_LORENTZ_SINGLE_INPUT_NUMPY(f_np[t],x_np,A_np,B_np,C_np,D_np)
                x_np = x_next_np

            #compare numpy and tensorflow results
            s_e_d = sess.run(s_e_d)
            s_e_d1 = sess.run(s_e_d1)
            s_e_d2 = sess.run(s_e_d2)
            x_next = sess.run(x_next)

            print('Shape of x_next: ',np.shape(x_next))
            print('Shape of s_e_d: ',np.shape(s_e_d))

            print('the tensorflow error is: ',s_e_d - alpha*s_e_d1 - beta*s_e_d2)
            print('the numpy error is: ',s_e_d_np - alpha*s_e_d1_np - beta*s_e_d2_np)

            print('the tensorflow accumulator is: ',s_e_d)
            print('the numpy accumulator is: ',s_e_d_np)

            print('the tensorflow accumulator 1 is: ',s_e_d1)
            print('the numpy accumulator 1 is: ',s_e_d1_np)

            print('the tensorflow accumulator 2 is: ',s_e_d2)
            print('the numpy accumulator 2 is: ',s_e_d2_np)

        self.assertEqual(success, True)
                
    def test_lorentz_functions_check(self):
        #test the MULTIPLE_LORENTZ and LORENTZ functions for a single resonance

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_r = 3
        n_o = 2*n_r
        n_x, n_y, n_z , alpha , beta = map(int, np.random.randint(low=1, high=50, size=5))
        del_t = 1.0e-12
        n_t = 30

        #build random tensors
        inf_x = np.float32(np.random.rand(n_x,n_y,n_z))
        w_0 = np.float32(np.random.rand(n_x,n_y,n_z,n_r))*10**12
        damp = 0.2*w_0
        del_x = np.float32(np.random.rand(n_x,n_y,n_z,n_r))
        x = 0*np.float32(np.random.rand(n_x,n_y,n_z,n_o,n_f//2))
        x_1_pre = 0*np.float32(np.random.rand(n_x,n_y,n_z,1))
        x_2_pre = 0*np.float32(np.random.rand(n_x,n_y,n_z,1))

        #produce a linear combination of inputs
        f1 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_t))
        f2 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_t))
        f = alpha*f1+beta*f2

        #convert state variable and field to tensors
        f_tf = tf.convert_to_tensor(alpha*f1+beta*f2)
        f1_tf = tf.convert_to_tensor(f1)
        f2_tf = tf.convert_to_tensor(f2)
        
        #produce operators
        sta_ope , tra_ope = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)

        #produce linearily combined electrical dielectric accumulator over three time steps
        s_e_d_tf,x_next_tf = LORENTZ_LINEARITY(f_tf,x,sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3],n_t)

        #produce electrical dielectric accumulator over three time steps
        s1_e_d_tf,x1_next_tf = LORENTZ_LINEARITY(f1_tf,x,sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3],n_t)

        #produce electrical dielectric accumulator over three time steps
        s2_e_d_tf,x2_next_tf = LORENTZ_LINEARITY(f2_tf,x,sta_ope[0],sta_ope[1],sta_ope[2],sta_ope[3],n_t)

        #calculate zeros and poles
        # A = sta_ope[0][0,0,0,:,:]
        # B = sta_ope[1][0,0,0,:]
        # C = sta_ope[2][0,0,0,:]
        # B = [[B[0]],[B[1]],[B[2]],[B[3]],[B[4]],[B[5]]]
        # C = [[C[0],C[1],C[2],C[3],C[4],C[5]]]
        # D = [[sta_ope[3][0,0,0]]]
        

        with tf.Session() as sess:

            a = sess.run(sta_ope[0])
            b = sess.run(sta_ope[1])
            c = sess.run(sta_ope[2])
            d = sess.run(sta_ope[3])

            #produce linearily combined electrical dielectric accumulator over three time steps
            s_e_d,x_next = LORENTZ_LINEARITY_NUMPY(f,x,a,b,c,d,n_t)

            #produce electrical dielectric accumulator over three time steps
            s1_e_d,x1_next = LORENTZ_LINEARITY_NUMPY(f1,x,a,b,c,d,n_t)

            #produce electrical dielectric accumulator over three time steps
            s2_e_d,x2_next = LORENTZ_LINEARITY_NUMPY(f2,x,a,b,c,d,n_t)

            s_e_d_tf = sess.run(s_e_d_tf)
            s1_e_d_tf = sess.run(s1_e_d_tf)
            s2_e_d_tf = sess.run(s2_e_d_tf)
            x_next = sess.run(x_next_tf)

            # A = sess.run(A)
            # B = sess.run(B)
            # C = sess.run(C)
            # D = sess.run(D)

            # print(A)
            # print(B)
            # print(C)
            # print(D)

            # z,p,k = signal.ss2zpk(A,B,C,D)
            # print('poles mag: ',np.abs(p))

            #s_d_comb = sess.run(s_d_comb)
            #s1_d = sess.run(s1_d)
            #s2_d = sess.run(s2_d)

            #test size of outputs
            success = np.array_equal(np.shape(s_e_d),np.array([n_x,n_y,n_z])) and np.array_equal(np.shape(x_next),np.array([n_x,n_y,n_z,n_o]))

            #test linearity of MULTIPLE LORENTZ state-space equation
            success = PERCENT_ERROR(s_e_d,alpha*s1_e_d + beta*s2_e_d) < 1.0e-4 and success
            print(PERCENT_ERROR(s_e_d,alpha*s1_e_d + beta*s2_e_d))
            print(PERCENT_ERROR(s_e_d_tf,alpha*s1_e_d_tf + beta*s2_e_d_tf))
            #print(PERCENT_ERROR(s_d_comb,s1_d + s2_d))


        self.assertEqual(success, True)

    def test_state_operators_2(self):
        #test the state operators to see if a single resonance and a three resonace (with only one non-zero) are equal

        success = True

        #position and field arguments
        n_f = 6
        n_r = 1
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        del_t = 1.0e-12

        #build random tensors
        w_0 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))*10**12
        damp = 0.2*w_0
        del_x = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))

        #secondary constants
        beta = np.sqrt(w_0**2 - damp**2)
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

        #pad primary constants
        zero_tensor = np.zeros((n_x,n_y,n_z,n_f//2,2))
        a_1 = np.concatenate((a_1,zero_tensor),axis = 4)
        a_2 = np.concatenate((a_2,zero_tensor),axis = 4)
        b_1 = np.concatenate((b_1,zero_tensor),axis = 4)
        b_2 = np.concatenate((b_2,zero_tensor),axis = 4)

        #calculate the lorentz transfer function
        tran_num_coeff_more_zeros , tran_den_coeff_more_zeros = LORENTZ_TRANSFER_FUNCTION(a_1,a_2,b_1,b_2)

        print(tran_num_coeff_more_zeros[0,0,0,0,:])
        print(tran_num_coeff[0,0,0,0,:])
        print(tran_den_coeff_more_zeros[0,0,0,0,:])
        print(tran_den_coeff[0,0,0,0,:])

        self.assertEqual(success, True)

    def test_state_operators_1(self):
        #test the state space operators to see if they correspond to the correct transfer function

        success = True

        #position and field arguments
        n_f = 6
        n_r = 3
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        del_t = 1.0e-12

        #build random tensors
        inf_x = np.float32(np.random.rand(n_x,n_y,n_z))
        w_0 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))*10**12
        damp = 0.2*w_0
        del_x = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        
        #produce operators
        sta_ope , _ = ELECTRIC_DISPERSION_OPERATORS(w_0,damp,del_x,del_t,inf_x)
        a = sta_ope[0]
        b = sta_ope[1]
        c = sta_ope[2]
        d = sta_ope[3]

        #secondary constants
        beta = np.sqrt(w_0**2 - damp**2)
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

        with tf.Session() as sess:

            d = sess.run(d)
            c = sess.run(c)
            b = sess.run(b)
            a = sess.run(a)

            B = b[0,0,0,0,:,np.newaxis]
            for i in range(len(b[0,0,0,0,:])):
                B[i,0] = b[0,0,0,0,i]
            A = a[0,0,0,0,:,:]
            C = [c[0,0,0,0,:]]
            D = d[0,0,0,0]

            numden = ss2tf(A,B,C,D)
            num = numden[0]
            den = numden[1]

            np.amax(num-tran_num_coeff[0,0,0,0]) < 5.0e-7 and np.amax(den-tran_den_coeff[0,0,0,0]) < 5.0e-7 and success

        self.assertEqual(success, True)

    def test_sum_poly_numpy(self):
        #test the sum_poly_numpy function used to test the sum_rational_poly function in layers

        sucess = True

        #position and field arguments
        n_o = 3
        n_r = 3

        #build random tensors
        num = np.float32(np.random.rand(n_r,n_o+1))
        den = np.float32(np.random.rand(n_r,n_o+1))

        x = np.linspace(-10,10,1000)
        y = np.zeros(1000)

        #solve for y in the simple way first
        y_simple = ( (num[0,0]*x**3 + num[0,1]*x**2 + num[0,2]*x**1 + num[0,3]*x**0) / (den[0,0]*x**3 + den[0,1]*x**2 + den[0,2]*x**1 + den[0,3]*x**0)
                    + (num[1,0]*x**3 + num[1,1]*x**2 + num[1,2]*x**1 + num[1,3]*x**0) / (den[1,0]*x**3 + den[1,1]*x**2 + den[1,2]*x**1 + den[1,3]*x**0)
                    + (num[2,0]*x**3 + num[2,1]*x**2 + num[2,2]*x**1 + num[2,3]*x**0) / (den[2,0]*x**3 + den[2,1]*x**2 + den[2,2]*x**1 + den[2,3]*x**0) )

        for i in range(1000):
            y[i] = SUM_POLY_NUMPY(num,den,x[i])

        success = np.array_equal(y,y_simple) and sucess

        self.assertEqual(success, True)

    def test_polynomial_function(self):
        #test rational polynomial summing function for the case of one polynomials and three polynomials

        success = True

        #position and field arguments
        n_o = 2
        n_r = 1

        #build random tensors
        num_coeff = np.float32(np.random.rand(n_r,n_o+1))
        den_coeff = np.float32(np.random.rand(n_r,n_o+1))

        #Sum the polynomial with zero
        numerator_results , denominator_results = SUM_RATIONAL_POLY(num_coeff,den_coeff)
        
        success = np.array_equal(numerator_results,num_coeff[0,:]) and np.array_equal(denominator_results,den_coeff[0,:]) and success

        #pad num_ceoff polynomials with zeros and compare results
        zero_tensor = np.zeros((2,n_o+1))
        num_coeff = np.concatenate((num_coeff,zero_tensor),axis = 0)
        den_coeff = np.concatenate((den_coeff,zero_tensor),axis = 0)

        #Sum again to see if the same results are given
        numerator_results , denominator_results = SUM_RATIONAL_POLY(num_coeff,den_coeff)

        success = np.array_equal(numerator_results,num_coeff[0,:]) and np.array_equal(denominator_results,den_coeff[0,:]) and success

        self.assertEqual(success, True)

    def test_transfer_function(self):
        #test the state operators of the Lorentz model for single and multiple Lorentz resonances

        success = True

        #position and field arguments
        n_c = 12
        n_f = 6
        n_r = 1
        n_o = 2*n_r + 1
        n_x, n_y, n_z = map(int, np.random.randint(low=1, high=50, size=3))
        del_t = 1.0e-12

        #build random tensors
        a_1 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        a_2 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        b_1 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        b_2 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))

        #calculate polynomial coefficients
        tran_num_coeff , tran_den_coeff = LORENTZ_TRANSFER_FUNCTION(a_1,a_2,b_1,b_2)

        a_1 = a_1[:,:,:,:,0]
        a_2 = a_2[:,:,:,:,0]
        b_1 = b_1[:,:,:,:,0]
        b_2 = b_2[:,:,:,:,0]

        tran_num_coeff_con = np.stack([b_1,b_2,b_1*0],axis = 4)
        tran_den_coeff_con = np.stack([a_1/a_1,-a_1,-a_2],axis = 4)

        #multiple Loretnz resonaces
        n_r = 3
        n_o = 2*n_r + 1

        #build random tensors
        a_1 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        a_2 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        b_1 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))
        b_2 = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2,n_r))

        #calculate polynomial coefficients
        tran_num_coeff , tran_den_coeff = LORENTZ_TRANSFER_FUNCTION(a_1,a_2,b_1,b_2)

        print(tran_num_coeff[0,0,0,0,:])
        print(tran_den_coeff[0,0,0,0,:])


        self.assertEqual(success, True)

    def test_minimize(self):

        success = True

        f_per = np.array([[1,2],[1,3]])

        f_per = tf.reduce_min(f_per,axis = -1)

        with tf.Session() as sess:

            f_per = sess.run(f_per)

            print(f_per)

        self.assertEqual(success, True)

    def test_gather(self):

        success = True

        for _ in range(10):

            n_m = 2
            n_r = 2

            n_x, n_y, n_z = map(int, np.random.randint(low=1, high=10, size=3))

            param_1 = np.float32(np.random.rand(n_x,n_y,n_z,n_m))
            param_2 = np.float32(np.random.rand(n_x,n_y,n_z,n_r,n_m))
            index = np.random.randint(0, high=n_m-1, size=[n_x,n_y,n_z])
            
            output_1 = param_1[:,:,:,0]
            output_2 = param_2[:,:,:,:,0]

            for x in range(n_x):
                for y in range(n_y):
                    for z in range(n_z):
                        output_1[x,y,z] = param_1[x,y,z,index[x,y,z]]
                        output_2[x,y,z,:] = param_2[x,y,z,:,index[x,y,z]]

            param_1_tf = tf.convert_to_tensor(param_1)
            param_2_tf = tf.convert_to_tensor(param_2)
            index_tf = tf.convert_to_tensor(index)

            index_1_tf = [[0,0,0,index_tf[0,0,0]]]
            index_2_tf = [[0,0,0,0,index_tf[0,0,0]]]

            for x in range(n_x):
                for y in range(n_y):
                    for z in range(n_z):

                        if x == 0 and y == 0 and z == 0:
                            index_1_tf = index_1_tf
                        else:
                            index_1_tf = tf.concat([index_1_tf,[[x,y,z,index_tf[x,y,z]]]],0)

                        for r in range(n_r):
                            if x == 0 and y == 0 and z == 0 and r == 0:
                                index_2_tf = index_2_tf
                            else:
                                index_2_tf = tf.concat([index_2_tf,[[x,y,z,r,index_tf[x,y,z]]]],0)

            output_1_tf = tf.reshape(tf.gather_nd(param_1_tf,index_1_tf,name=None),[n_x,n_y,n_z])
            output_2_tf = tf.reshape(tf.gather_nd(param_2_tf,index_2_tf,name=None),[n_x,n_y,n_z,n_r])

            print('tf.shape of output_1_tf:',tf.shape(output_1_tf))

            with tf.Session() as sess:

                output_1_tf = sess.run(output_1_tf)
                output_2_tf = sess.run(output_2_tf)

                print('np.shape of output_1_tf:',np.shape(output_1_tf))
                print('np.type of output_1_tf:',type(output_1_tf))
                print('np.type of output_1_tf:',type(output_1_tf[0,0,0]))

                success = np.array_equal(output_1_tf,output_1) and np.array_equal(output_2_tf,output_2) and success

        self.assertEqual(success, True)

    def test_non_linear(self):
        #test the non-linear function
        success = True

        n_f = 6
        n_x = 1
        n_y = 120
        n_z = 1
        
        x_nl = 1.0e-12*np.ones((n_x,n_y,n_z,n_f//2),dtype = data_type)
        t = 2.0*np.ones((n_x,n_y,n_z,n_f//2),dtype = data_type)
        u = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2),dtype = data_type)
        f_pre = np.float32(np.random.rand(n_x,n_y,n_z,n_f//2),dtype = data_type)

        x_nl = tf.convert_to_tensor(x_nl)
        t = tf.convert_to_tensor(t)
        u = tf.convert_to_tensor(u)
        f_pre = tf.convert_to_tensor(f_pre)

        #chi 2 non-linearity
        f = CHI_2_NON_LINEAR(x_nl,t,u,f_pre)

        with tf.Session() as sess:

            f = sess.run(f)
            print(f[0,0,0,:])

        self.assertEqual(success, True)

    def test_spectrum(self):

        success = True 

        #spatial/time/field numbers
        n_x = 1
        n_y = 100
        n_z = 1
        n_f = 6
        n_t = 100

        #time step
        del_t = 1.0e-12

        #frequency values
        freq_1 = 0.2/del_t
        freq_2 = 2*freq_1
        print('freq_1 (THz): ',10**-12*freq_1)

        #filed in space and time
        f = np.zeros((n_x,n_y,n_z,n_f,n_t),dtype = np.complex)
        for y in range(n_y):
            for t in range(n_t):
                if y == 24 or y == 74:
                    f[0,y,0,2,t] = np.cos(2*np.pi*freq_1*t*del_t)
                if y == 34 or y  == 64:
                    f[0,y,0,2,t] = np.cos(2*np.pi*freq_2*t*del_t)

        f = tf.convert_to_tensor(f)

        #calculate spectrum
        sp_1 , sp_2 = SPECTRUM_Z(f,del_t,n_t,freq_1,freq_2)


        with tf.Session() as sess:
            sp_1 = sess.run(sp_1)
            sp_2 = sess.run(sp_2)
            f = np.real(sess.run(f))

            #plot results
            plt.figure(1)
            plt.plot(np.abs(sp_1),label = 'res freq')
            plt.plot(np.abs(sp_2),label = 'double res freq')
            plt.xlabel('position')
            plt.title('spectrum')
            plt.ylabel('mag')
            plt.legend()
            plt.show()

            plt.figure(3)
            plt.imshow(f[0,:,0,2,:])
            plt.xlabel('time (ps)')
            plt.colorbar()
            plt.title('time signal over space')
            plt.ylabel('position')
            plt.show()

        self.assertEqual(success, True)

    def test_trapz(self):

        success = True

        n = 1000
        a = 0
        b = 3*np.pi

        x = np.linspace(a,b,n+1)
        f = np.sin(x)
        f_int = -np.cos(b) + np.cos(a)

        trapz = TRAPZ(f,a,b,n)

        print('f_int: ',f_int)

        with tf.Session() as sess:

            trapz = sess.run(trapz)
            print('trapz: ',trapz)

        self.assertEqual(success, True)

#runs all test cases in Test subclass
if __name__ == '__main__':
   unittest.main()