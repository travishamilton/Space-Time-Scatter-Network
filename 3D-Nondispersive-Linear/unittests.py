import unittest

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from weights import WEIGHT_CREATION

from layers import *
from weights import *
from files import *
from fields import SCATTER_2_ELECTRIC_NODES
from plots import PLOT_VIDEO
from spectrum import SPECTRUM

from forward_layers import SCATTER_TENSOR , TRANSFER_OPERATIONS , SCATTER_MATRIX
from forward_layers import PROPAGATE as PROPAGATE_NP

from forward_fields import POINT_SOURCE , TENSORIZE

from forward_plots import PLOT_TIME_SOURCE


data_type = np.float32

# ------------------- np test equations ---------------------#
def tensor_mul_sum_np(A,B):

    C = (A[...,None]*B).sum(3)

    return C

def tensor_mul_np(A,B):

   C = np.einsum('ijkm,ijkmn->ijkn',A,B)

   return C

def tensor_mul_tf(A,B):

   C = tf.einsum('ijkm,ijkmn->ijkn',A,B)

   return C

def spectrum_np(field_time_0,field_time_1,field_time_2,n_f):

    #polarization 0
	field_freq_0 = np.fft.fft(field_time_0, axis = -1)
	#polarization 1
	field_freq_1 = np.fft.fft(field_time_1,axis = -1)
	#polarization 2
	field_freq_2 = np.fft.fft(field_time_2,axis = -1)

	return field_freq_0 , field_freq_1 , field_freq_2

# ----------------------- unittest class ------------------------ #
data_type = tf.float32
class Test(unittest.TestCase):

    def test_tensor_mul(self):
        success = True
        for _ in range(10):
            # sample input
            n_c = 12
            n_x, n_y, n_z = map(int, np.random.randint(low=2, high=3, size=3))
            test_field = np.random.random([n_x, n_y, n_z, n_c])
            test_scatter = np.random.random([n_x, n_y, n_z, n_c, n_c])
            # evaluate the numpy version
            test_result = tensor_mul_np(test_field,test_scatter)
            # evaluate the tensorflow version
            with tf.Session() as sess:
                tf_field = tf.constant(test_field, dtype=data_type)
                tf_scatter = tf.constant(test_scatter, dtype=data_type)
                tf_result = tensor_mul_tf(tf_field,tf_scatter)
                tf_result = sess.run(tf_result)
                # check that outputs are similar
                success = success and np.allclose(test_result, tf_result)

        self.assertEqual(success, True)
    
    def test_weights(self):

        success = True

        for _ in range(10):
            
            n_x,n_y,n_z,n_w, initial_weight = map(int, np.random.randint(low=1, high=50, size=5))

            mask_start_x = np.random.randint(low=0, high=n_x)
            mask_start_y = np.random.randint(low=0, high=n_y)
            mask_start_z = np.random.randint(low=0, high=n_z)

            mask_end_x = np.random.randint(low=mask_start_x, high=n_x)
            mask_end_y = np.random.randint(low=mask_start_y, high=n_y)
            mask_end_z = np.random.randint(low=mask_start_z, high=n_z)
            
            mask_start = (mask_start_x,mask_start_y,mask_start_z)
            mask_end = (mask_end_x,mask_end_y,mask_end_z)

            weights , weights_train = WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z,n_w, initial_weight)

            with tf.Session() as sess:
                #initializies variables in graph
                sess.run( tf.global_variables_initializer() )

                weights = sess.run(weights)
                n_x_results , n_y_results , n_z_results , n_w_results = np.shape(weights)

            success = success and np.array_equal([n_x,n_y,n_z,n_w],[n_x_results,n_y_results,n_z_results,n_w_results])

        self.assertEqual(success, True)

    def test_fft3d(self):

        success = True
        for _ in range(1):

            #position and time arguments
            n_x, n_y, n_z, n_t = map(int, np.random.randint(low=1, high=50, size=4))
            #number of frequency points in each dimension and each polarization
            n_f_x,n_f_y,n_f_z = map(int, np.random.randint(low=1, high=50, size=3))
            n_f = (n_f_x,n_f_y,n_f_z)

            #produces random distribution on the interval [0,3.0) with type float32
            field_time_0 = np.float32(np.random.rand(n_x,n_y,n_z,n_t)*3)
            field_time_1 = np.float32(np.random.rand(n_x,n_y,n_z,n_t)*3)
            field_time_2 = np.float32(np.random.rand(n_x,n_y,n_z,n_t)*3)

            field_freq_0_np , field_freq_1_np , field_freq_2_np = spectrum_np(field_time_0,field_time_1,field_time_2,n_f)

            field_time_0 = tf.constant(field_time_0, dtype=data_type)
            field_time_1 = tf.constant(field_time_1, dtype=data_type)
            field_time_2 = tf.constant(field_time_2, dtype=data_type)

            field_freq_0 , field_freq_1 , field_freq_2 = SPECTRUM(field_time_0,field_time_1,field_time_2,n_f)

            with tf.Session() as sess:
                    # run tensorflow code
                    field_freq_0 = sess.run(field_freq_0)
                    field_freq_1 = sess.run(field_freq_1)
                    field_freq_2 = sess.run(field_freq_2)

                    _,_,_,n_f = np.shape(field_freq_0_np)

                    # check that outputs are similar
                    success = success and np.allclose(field_freq_0, field_freq_0_np[:,:,:,0:n_f//2 + 1])
                    success = success and np.allclose(field_freq_1, field_freq_1_np[:,:,:,0:n_f//2 + 1])
                    success = success and np.allclose(field_freq_2, field_freq_2_np[:,:,:,0:n_f//2 + 1])

        self.assertEqual(success, True)

#runs all test cases in Test subclass
if __name__ == '__main__':
   unittest.main()
