import tensorflow as tf
import numpy as np

def WEIGHT_CREATION(mask_start, mask_end, data_type, n_x, n_y, n_z , n_w, initial_weight):	
    #produces the weights to be trained within the masked region
    #mask_start: smallest coordinates of the masked region - tuple int, shape(3,)
    #mask_end: largest coordinates of the masked region - tuple int, shape(3,)
    #data_type: tensorflow data type used for tensors - tf.datatype
	#n_x: number of points in the first direction - int shape(1,)
	#n_y: number of points in the second direction - int shape(1,)
	#n_z: number of points in the third direction - int shape(1,)
	#n_w: number of weights per each node - int shape(1,)
	#initial_weight: sets the initial weight of the masked region - int shape(1,)
	#
    #weights_train: tensorflow variable containing weights in masked region - tf.Variable, shape(n_x_masked,n_y_masked,n_z_masked,n_w)
	#weights: tensorflow variable containing weights in all regions -  shape(n_x,n_y,n_z,n_w)

	
	# ensure that start is not larger than end
	if mask_start[0] <= mask_end[0] and mask_start[1] <= mask_end[1] and mask_start[2] <= mask_end[2]:

		# create weights over the masked region
		weights_train = tf.Variable(initial_weight*tf.ones(shape = [mask_end[0] - mask_start[0] + 1 , mask_end[1] - mask_start[1] + 1 , mask_end[2] - mask_start[2] + 1 , n_w] , dtype = data_type))
	
		#create weights over entire simulatoin region
		weights = weights_train
	
		# attach weights along the first axis
		if n_x > mask_end[0]+1:
			weights_tmp = tf.ones(shape = [n_x - mask_end[0] - 1 , mask_end[1] - mask_start[1] + 1 , mask_end[2] - mask_start[2] + 1 , n_w],dtype = data_type)
			weights = tf.concat([weights,weights_tmp],0)

		if mask_start[0] > 0:
			weights_tmp = tf.ones(shape = [mask_start[0] , mask_end[1] - mask_start[1] + 1 , mask_end[2] - mask_start[2] + 1 , n_w],dtype = data_type)
			weights = tf.concat([weights_tmp,weights],0)

		# attach weights along the second axis
		if n_y > mask_end[1]+1:
			weights_tmp = tf.ones(shape = [n_x , n_y - mask_end[1] - 1 , mask_end[2] - mask_start[2] + 1 , n_w],dtype = data_type)
			weights = tf.concat([weights,weights_tmp],1)

		if mask_start[1] > 0:
			weights_tmp = tf.ones(shape = [n_x , mask_start[1] , mask_end[2] - mask_start[2] + 1 , n_w],dtype = data_type)
			weights = tf.concat([weights_tmp,weights],1)

		# attach weights along the third axis
		if n_z > mask_end[2]+1:
			weights_tmp = tf.ones(shape = [n_x , n_y , n_z - mask_end[2] - 1 , n_w],dtype = data_type)
			weights = tf.concat([weights,weights_tmp],2)

		if mask_start[2] > 0:
			weights_tmp = tf.ones(shape = [n_x , n_y , mask_start[2] , n_w],dtype = data_type)
			weights = tf.concat([weights_tmp,weights],2)

		return weights , weights_train
	
	else:
		raise ValueError("Starting index must be smaller than or equal to ending index.")

