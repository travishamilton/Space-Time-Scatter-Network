import unittest

import numpy as np

import tensorflow as tf

from weights import WEIGHT_CREATION

from layers import *
from weights import *

from forward_model.layers import SCATTER_TENSOR

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

# ----------------------- unittest class ------------------------ #
data_type = tf.float32
class Test(unittest.TestCase):
    def test_1(self):
        # always passes
        self.assertEqual(True, True)

    def test_get_entry(self):
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
            
            layers, n_x,n_y,n_z,n_w, initial_weight = map(int, np.random.randint(low=1, high=50, size=6))

            mask_start_x = np.random.randint(low=0, high=n_x)
            mask_start_y = np.random.randint(low=0, high=n_y)
            mask_start_z = np.random.randint(low=0, high=n_z)

            mask_end_x = np.random.randint(low=mask_start_x, high=n_x)
            mask_end_y = np.random.randint(low=mask_start_y, high=n_y)
            mask_end_z = np.random.randint(low=mask_start_z, high=n_z)
            
            mask_start = (mask_start_x,mask_start_y,mask_start_z)
            mask_end = (mask_end_x,mask_end_y,mask_end_z)

            weights = WEIGHT_CREATION(mask_start, mask_end, layers, data_type, n_x, n_y, n_z,n_w, initial_weight)

            with tf.Session() as sess:
                #initializies variables in graph
                sess.run( tf.global_variables_initializer() )

                weights = sess.run(weights)
                layers_results , n_x_results , n_y_results , n_z_results , n_w_results = np.shape(weights)

            print('results :',layers_results , n_x_results , n_y_results , n_z_results)
            print('inputs :',layers , n_x , n_y , n_z)

            success = success and np.array_equal([layers,n_x,n_y,n_z,n_w],[layers_results,n_x_results,n_y_results,n_z_results,n_w_results])

        self.assertEqual(success, True)

    def test_scatter_tensor(self):

        success = True
        for _ in range(10):
            n_c = 12
            n_w = 1

            layers, n_x,n_y,n_z, initial_weight = map(int, np.random.randint(low=1, high=10, size=5))

            mesh = np.random.rand(n_x,n_y,n_z,3)*2.0 + 2.0          #[2.0,4.0)
            n = np.ones((n_x,n_y,n_z,1))

            mask_start_x = np.random.randint(low=0, high=n_x)
            mask_start_y = np.random.randint(low=0, high=n_y)
            mask_start_z = np.random.randint(low=0, high=n_z)

            mask_end_x = np.random.randint(low=mask_start_x, high=n_x)
            mask_end_y = np.random.randint(low=mask_start_y, high=n_y)
            mask_end_z = np.random.randint(low=mask_start_z, high=n_z)

            n[mask_start_x:mask_end_x,mask_start_y:mask_end_y,mask_start_z:mask_end_z,0] = initial_weight
            
            mask_start = (mask_start_x,mask_start_y,mask_start_z)
            mask_end = (mask_end_x,mask_end_y,mask_end_z)

            weight_tens = WEIGHT_CREATION(mask_start, mask_end, layers, data_type, n_x, n_y, n_z,n_w, initial_weight)

            a1,a21,a22,a3,a4,f1,f2,O = CONSTANT_TENSORS(mesh,n_c,layers)

            scatter_tens = SCATTER(weight_tens,a1,a21,a22,a3,a4,f1,f2,O,n_c,n_x,n_y,n_z,n_w,layers)

            scatter_tens_np = SCATTER_TENSOR(mesh,n,n_c)

            with tf.Session() as sess:
                #initializies variables in graph
                sess.run( tf.global_variables_initializer() )

                scatter_tens = sess.run(scatter_tens)

                success = success and np.array_equal(scatter_tens,scatter_tens_np)

#runs all test cases in Test subclass
if __name__ == '__main__':
   unittest.main()

