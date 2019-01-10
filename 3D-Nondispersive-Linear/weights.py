import tensorflow as tf
import numpy as np

data_type = tf.float32

def WEIGHT_CREATION(mask_start, mask_end, n_x, n_y, n_z):	
    #produces the weights to be trained within the masked region
    #mask_start: smallest coordinates of the masked region - tuple int, shape(3,)
    #mask_end: largest coordinates of the masked region - tuple int, shape(3,)
	#n_x: number of points in the first direction - int shape(1,)
	#n_y: number of points in the second direction - int shape(1,)
	#n_z: number of points in the third direction - int shape(1,)
	#
    #weights_train: tensorflow variable containing weights in masked region - tf.Variable, shape(n_x_masked,n_y_masked,n_z_masked)
	#weights: tensorflow variable containing weights in all regions -  shape(n_x,n_y,n_z)

	
	# ensure that start is not larger than end
	if mask_start[0] <= mask_end[0] and mask_start[1] <= mask_end[1] and mask_start[2] <= mask_end[2]:

		# create weights over the masked region
		weights_train = tf.Variable(tf.ones(shape = [mask_end[0] - mask_start[0] + 1 , mask_end[1] - mask_start[1] + 1 , mask_end[2] - mask_start[2] + 1 ] , dtype = data_type))
	
		#create weights over entire simulatoin region
		weights = weights_train
	
		# attach weights along the first axis
		if n_x > mask_end[0]+1:
			weights_tmp = tf.zeros(shape = [n_x - mask_end[0] - 1 , mask_end[1] - mask_start[1] + 1 , mask_end[2] - mask_start[2] + 1],dtype = data_type)
			weights = tf.concat([weights,weights_tmp],0)

		if mask_start[0] > 0:
			weights_tmp = tf.zeros(shape = [mask_start[0] , mask_end[1] - mask_start[1] + 1 , mask_end[2] - mask_start[2] + 1],dtype = data_type)
			weights = tf.concat([weights_tmp,weights],0)

		# attach weights along the second axis
		if n_y > mask_end[1]+1:
			weights_tmp = tf.zeros(shape = [n_x , n_y - mask_end[1] - 1 , mask_end[2] - mask_start[2] + 1],dtype = data_type)
			weights = tf.concat([weights,weights_tmp],1)

		if mask_start[1] > 0:
			weights_tmp = tf.zeros(shape = [n_x , mask_start[1] , mask_end[2] - mask_start[2] + 1],dtype = data_type)
			weights = tf.concat([weights_tmp,weights],1)

		# attach weights along the third axis
		if n_z > mask_end[2]+1:
			weights_tmp = tf.zeros(shape = [n_x , n_y , n_z - mask_end[2] - 1],dtype = data_type)
			weights = tf.concat([weights,weights_tmp],2)

		if mask_start[2] > 0:
			weights_tmp = tf.zeros(shape = [n_x , n_y , mask_start[2]],dtype = data_type)
			weights = tf.concat([weights_tmp,weights],2)

		return weights
	
	else:
		raise ValueError("Starting index must be smaller than or equal to ending index.")

def WEIGHT_CREATION_TEST(n_x, n_y, n_z):

	weights = tf.Variable(tf.zeros(shape = [n_x,n_y,n_z],dtype = data_type))

	return weights

def WEIGHT_INDEXING(weights,a_mat,b_mat,c_mat,d_mat):
	# weights: material index for each position - shape(n_x,n_y,n_z)
	# a_mat: constant tensor operating on x to update x for all materials - shape(n_x,n_y,n_z,n_s,n_s,n_m)
    # b_mat: constant tensor operating on f to update x for all materials - shape(n_x,n_y,n_z,n_s,n_m)
    # c_mat: constant tensor operating on x to update s_e_d for all materials - shape(n_x,n_y,n_z,n_s,n_m)
    # d_mat: constant tensor operating on f to update s_e_d for all materials - shape(n_x,n_y,n_z,n_m)
	#
	# a: constant tensor operating on x to update x for indexed materials - shape(n_x,n_y,n_z,n_s,n_s)
    # b: constant tensor operating on f to update x for indexed materials - shape(n_x,n_y,n_z,n_s)
    # c: constant tensor operating on x to update s_e_d for indexed materials - shape(n_x,n_y,n_z,n_s)
    # d: constant tensor operating on f to update s_e_d for indexed materials - shape(n_x,n_y,n_z)
	
	# get number of positional points, state space variables and materials.
	n_x,n_y,n_z,n_s,n_s,n_m = np.shape(a_mat)

	# index of each parameter
	index_a = [[0,0,0,0,0,weights[0,0,0]]]
	index_b = [[0,0,0,0,weights[0,0,0]]]
	index_c = [[0,0,0,0,weights[0,0,0]]]
	index_d = [[0,0,0,weights[0,0,0]]]

	# assign index for each constant tensor
	for x in range(n_x):
		for y in range(n_y):
			for z in range(n_z):
				if x == 0 and y == 0 and z == 0:
							index_d = index_d
				else:
					index_d = tf.concat([index_d,[[x,y,z,weights[x,y,z]]]],0)

				for s1 in range(n_s):
					if x == 0 and y == 0 and z == 0 and s1 == 0:
							index_b = index_b
							index_c = index_c
					else:
						index_b = tf.concat([index_b,[[x,y,z,s1,weights[x,y,z]]]],0)
						index_c = tf.concat([index_c,[[x,y,z,s1,weights[x,y,z]]]],0)

					for s2 in range(n_s):
						if x == 0 and y == 0 and z == 0 and s1 == 0 and s2 == 0:
							index_a = index_a
						else:
							index_a = tf.concat([index_a,[[x,y,z,s1,s2,weights[x,y,z]]]],0)

	# index constant tensor
	a = tf.reshape(tf.gather_nd(a_mat,index_a,name=None),[n_x,n_y,n_z,n_s,n_s])
	b = tf.reshape(tf.gather_nd(b_mat,index_b,name=None),[n_x,n_y,n_z,n_s])
	c = tf.reshape(tf.gather_nd(c_mat,index_c,name=None),[n_x,n_y,n_z,n_s])
	d = tf.reshape(tf.gather_nd(d_mat,index_d,name=None),[n_x,n_y,n_z])

	return a , b , c , d

