import tensorflow as tf
import numpy as np
from scipy.linalg import block_diag

data_type = np.float32

# ---------------------- Propagation Matrix ---------------------#
def propagate(x):
	sampN, featN = x.shape.as_list()	# get shape of the data, O(1) constant run-time
	
	out = np.zeros((featN,featN),dtype = data_type)

	for jj in range(featN//3):
	    if jj>=1:
	    	out[3*jj - 2,3*jj]=1
	    if jj<=featN//3-2:
	    	out[3*jj + 3,3*jj + 1]=1
	    out[3*jj+2,3*jj+2] = 1

	return out

# ---------------------- Scatter Matrix Generation ----------------------#
def scatter(w):

	N = w.shape.as_list()[0]	# extract number of weights, N

	A = [ [1,0,0] , [0,1,0] , [0,0,-2] ]				# initialize A matrix
	A2 = A
	B = [ [-1,0,2] , [0,-1,2] , [0,0,1] ]				# initialize B matrix
	B2 = B

	# diagonlize A and B N times
	for _ in range(N-1):
 		A = block_diag(A,A2)
 		B = block_diag(B,B2)
	A = tf.constant(A, dtype = tf.float32)

	U_left = [ [-1,-1] , [1,0] , [0,1] ]				# initialize U_left matrix
	U_left_2 = U_left
	U_right = [ [1] , [1] , [1] ]						# initialize U_right matrix
	U_right_2 = U_right

	for _ in range(N-1):
		U_left = block_diag(U_left,U_left_2)
		U_right = block_diag(U_right,U_right_2)
	
	U = np.concatenate((U_left,U_right),1)
	U_inv = np.linalg.inv(U)
	U = tf.constant(U,dtype = tf.float32)
	U_inv = tf.constant(U_inv,dtype = tf.float32)


	# create the weight matrix through the eigenvalue matrix V and eigenvector matrix U
	V = tf.zeros(shape = [2*N],dtype = tf.float32)
	V = tf.concat([V,tf.multiply(3.0,w)],axis = 0)
	V = tf.diag(V)

	with tf.name_scope('Weight_Matrix'):
		W = tf.matmul(tf.matmul(U,V,name='U_V_mul'),U_inv,name = 'U_V_Ui_mul')

	scatter = tf.matmul(W,A) + B	# create scatter matrix for 1 time unit

	return scatter

def trasmitTimeDep(x,W,N):
	# N is now a vector of indicating units of time per weight in w
	# W is now a 2-d matrix of weights, each row of W corresponding to the set of weights for each time unit
	# X is a 2-d matrix with shape (sample number,feature number)

	#Create propagation matrix
	with tf.name_scope('build_matrix'):
		P0 = propagate(x)
		
		for i in range(N):
			
			curr_w = W[i,:]
			curr_w = tf.reshape( curr_w, [-1] )
			S = scatter(curr_w)
			P0S = tf.matmul(P0,S)

			if i is 0:
				tmp = P0S
			else:
				tmp = tf.matmul(tmp,P0S)

	with tf.name_scope('calculate_scattering_propagation'):		
		out = tf.matmul(x,tf.transpose(tmp)) 

	return out

def transmit(x,W,N):
	# N is now a vector of indicating units of time per weight in w
	# W is now a 2-d matrix of weights, each row of W corresponding to the set of weights for each time unit

	#Create propagation matrix
	with tf.name_scope('build_submatrix'):
		P0 = tf.transpose(propagate(x))
		S = tf.transpose(scatter(W))
		SP0 = tf.matmul(S,P0)
		tmp = SP0

	with tf.name_scope('build_matrix'):
		for i in range(N-1):

			with tf.name_scope('scatter_translation_layer'):
				tmp = tf.matmul(tmp,SP0)

		out = tf.matmul(x,tmp)

	return out

