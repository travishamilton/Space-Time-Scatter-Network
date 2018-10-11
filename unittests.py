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

# ----------------------- unittest class ------------------------ #
data_type = tf.float32
class Test(unittest.TestCase):

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

    def test_scatter_tensor(self):

        success = True
        for _ in range(10):
            n_c = 12
            n_w = 1

            n_x,n_y,n_z, initial_weight = map(int, np.random.randint(low=1, high=10, size=4))
            #n_x = 10
            #n_y = 10
            #n_z = 1

            initial_weight = 1
            initial_n = 1/np.sqrt(initial_weight)

            mesh = np.random.rand(n_x,n_y,n_z,3)*2.0 + 2.0          #[2.0,4.0)
            mesh = np.ones((n_x,n_y,n_z,3))*2.0
            n = initial_n * np.ones((n_x,n_y,n_z,1))

            mask_start_x = np.random.randint(low=0, high=n_x)
            mask_start_y = np.random.randint(low=0, high=n_y)
            mask_start_z = np.random.randint(low=0, high=n_z)

            mask_end_x = np.random.randint(low=mask_start_x, high=n_x)
            mask_end_y = np.random.randint(low=mask_start_y, high=n_y)
            mask_end_z = np.random.randint(low=mask_start_z, high=n_z)

            n[mask_start_x-1:mask_end_x-1,mask_start_y-1:mask_end_y-1,mask_start_z-1:mask_end_z-1,0] = initial_n
            
            mask_start = (mask_start_x,mask_start_y,mask_start_z)
            mask_end = (mask_end_x,mask_end_y,mask_end_z)

            weight_tens , weight_trains_tens = WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z,n_w, initial_weight)

            a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,beta1_tens,beta2_tens = CONSTANT_TENSORS(mesh,n_c)

            scatter_tens = SCATTER(weight_tens,a1,a2,a3,a4,a5,a6,filter_d_tens,filter_b_tens,ones_tens,n_c,n_x,n_y,n_z,n_w,beta1_tens,beta2_tens)

            scatter_tens_np = SCATTER_TENSOR(mesh,n,n_c)
 
            with tf.Session() as sess:
                #initializies variables in graph
                sess.run( tf.global_variables_initializer() )

                scatter_tens = sess.run(scatter_tens)
                scatter_tens_np = np.array(scatter_tens_np)

                for i in range(n_x):
                    for j in range(n_y):
                        for k in range(n_z):
                            for c1 in range(n_c):
                                for c2 in range(n_c):
                                    scatter_tens[i,j,k,c1,c2] = float('%.6f'%(scatter_tens[i,j,k,c1,c2]))
                                    scatter_tens_np[i,j,k,c1,c2] = float('%.6f'%(scatter_tens_np[i,j,k,c1,c2]))

                #print('scatter_tens: ', np.round(scatter_tens[mask_start_x,mask_start_y,mask_start_z,:,:],decimals=2))
                #print('scatter_tens_np: ', np.round(scatter_tens_np[mask_start_x,mask_start_y,mask_start_z,:,:],decimals=2))


                # print('scatter tensor at the location of interest np;: ', scatter_tens_np[4,4,0,:,:])
                # print('scatter tensor at the location of interest n;: ', scatter_tens[4,4,0,:,:])
            success = success and np.array_equal(scatter_tens,scatter_tens_np)

        self.assertEqual(success, True)

    def test_propagation(self):

        n_c = 12
        n_w = 1

        file_address_fields = "C:/Users/travi/Documents/Northwestern/STSN/field_data/"
        file_address_mesh = "C:/Users/travi/Documents/Northwestern/STSN/mesh_data/"

        in_field , out_field , layers , mask_start , mask_end , n_x , n_y , n_z , mesh , file_id , ref_index = GET_DATA(file_address_fields , file_address_mesh)

        in_field = tf.convert_to_tensor(in_field,dtype = tf.float32)

        weight_tens , weight_trains_tens = WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z,n_w, initial_weight = 1)

        field_out_tens = PROPAGATE(in_field,mesh,n_c,weight_tens,layers,n_x,n_y,n_z,n_w)
        
        #np code
        location = (5,1,0)
        polarization = 2
        wavelength = 30
        full_width_half_maximum = 3*wavelength
        source_scatter_field_vector,source_time = POINT_SOURCE(location,mesh,n_c,layers,ref_index,polarization,wavelength,full_width_half_maximum)
        _,field_out_vector_np,_ = PROPAGATE_NP(mesh,ref_index,n_c,source_scatter_field_vector,layers)
        field_out_tens_np = np.zeros((n_x,n_y,n_z,n_c,layers))

        for t in range(layers):
            field_out_tens_np[:,:,:,:,t] = TENSORIZE(field_out_vector_np[:,t],n_x,n_y,n_z,n_c)

        #least sqaures between np and tensorflow output
        least_squares = tf.norm(field_out_tens-field_out_tens_np, ord=2,name='least_squre')**2 

        with tf.Session() as sess:
            #initializies variables in graph
            sess.run( tf.global_variables_initializer() )

            field_out_tens = sess.run(field_out_tens)

            PLOT_VIDEO(field_out_tens,ref_index,mesh)

            PLOT_VIDEO(field_out_tens_np,ref_index,mesh)

            plt.show()

            print('least squares: ',sess.run(least_squares) )


        success = True

        self.assertEqual(success, True)

#runs all test cases in Test subclass
if __name__ == '__main__':
   unittest.main()

