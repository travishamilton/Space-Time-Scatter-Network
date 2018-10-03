import numpy as np
import pandas as pd
import tensorflow as tf
import os

from layers import *
from weightGen import weightCreation, weightConcat_static
from costMask import costMask

import pickle

data_type = tf.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well

#------------------------ Read in Data ------------------------#
with tf.device('/cpu:0'):
	userScatter = input("Scatter-points: ")
	userTime = input("Time units: ")
	userPoints = input("Position units: ")
	start = input("Starting Weight Index of Mask: ")	# starting index (1 <--> wN)
	end = input("Ending Weight Index of Mask: ")		# ending index (start <--> wN)
	timeChanges = input("Number of time changes: ")

	address = "forward_model_1D/field_data/"
	file_id = "scatter" + userScatter + "_T" + userTime + "_N" + userPoints + "_start" + start + "_end" + end + "_tc" + timeChanges   #file id
	fileName = address + file_id

	X = np.transpose( np.genfromtxt(fileName + '_in.csv', delimiter = ',') )
	Y = np.transpose( np.genfromtxt(fileName + '_out.csv', delimiter = ',') )



#------------------------ Extract Number of Layers ------------------------#
	# Print Information
	print("This model contains:")
	print("\t- " + userTime + " time units")
	print("\t- " + str(int(userScatter)) + " expected non-zero weights\n")

	layers = int(userTime)	# number of scatter/prop. layers to navigate through

	sampN, featN = X.shape	# sampN: number of training samples, featN: features per sample 



#----------- Random Weight Generation For Material -----------#
	wN = featN//3	# number of transmission weights
	start = int(start)
	end = int(end)

	print("\t- " + str(wN) + " total weights")
	print("\t- " + str((end-start + 1)) + " trainable weights out of total weights (masked region)\n")

	# extract arrays for trainable & frozen weights and place on CPU
with tf.device('/cpu:0'):
	with tf.name_scope('weights_creation'):
		W_left, W_train, W_right = weightCreation(start, end, wN)

		#with tf.name_scope('weights')
		W_tens = weightConcat_static(W_left, W_train, W_right)



#--------------------------- Placeholder Instantiation --------------------------#
with tf.device('/cpu:0'):
	with tf.name_scope('input'):
		X_tens = tf.placeholder(dtype = data_type, shape = [sampN,featN])
		Y_tens = tf.placeholder(dtype = data_type, shape = [sampN,featN])

#--------------------------- Cost Function Definition --------------------------#
# compute least squares cost for each sample and then average out their costs
print("Building Cost Function (Least Squares) ... ... ...")

with tf.device('/GPU:0'):
	with tf.name_scope('cost_function'):
		Yhat_tens = transmit(X_tens, W_tens, layers) # prediction function

		with tf.name_scope('Masking'):
			Yhat_masked = costMask(Yhat_tens - Y_tens, start, end)	# masking region "we don't know" for the cost

# perform least squares by squaring l2-norm (normalizing the cost by the number of known points)
		
		least_squares = tf.norm(Yhat_masked, ord=2,name='least_squre')**2 	#
	print("Done!\n")



#--------------------------- Define Optimizer --------------------------#
print("Building Optimizer ... ... ...")
lr = 0.001

with tf.name_scope('train'):
	train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(least_squares, var_list = [W_train])
with tf.name_scope('clip'):
	clip_op = tf.assign(W_train, tf.clip_by_value(W_train, 0, 1))

print("Done!\n")



#--------------------------- Merge Summaries ---------------------------#
merged = tf.summary.merge_all()



#--------------------------- Training --------------------------#
epochs = 1
loss_tolerance = 1e-10
table = []
loss_value_pre = 0

# saves objects for every iteration
fileFolder = "results/" + file_id

# if the results folder does not exist for the current model, create it
if not os.path.exists(fileFolder):
		os.makedirs(fileFolder)


with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	sess.run( tf.global_variables_initializer() )
	
	print("Tensor X:")		# show info. for X
	print(X)
	print("")

	print("Tensor W: ") 	# show info. for W total
	print(W_tens.eval())
	print(W_tens.dtype)
	print("")

	# show only the trainable part of W
	print("Trainable part of W (weights " + str(start) + " through " + str(end) + "):")
	print(W_train.eval())
	print("")

	print("Tensor Y: ")		# show info. for Y
	print(Y)
	print("")

	print("--------- Starting Training ---------\n")
	for i in range(1, epochs+1):

		# run X and Y dynamically into the network per iteration
		_, loss_value = sess.run([train_op, least_squares], feed_dict = {X_tens: X, Y_tens: Y})
		
		# perform clipping 
		with tf.name_scope('clip'):
			sess.run(clip_op)

		currStatus = [i, loss_value]	# status of the network for the current epoch

		if i == 1:
			# add the weights to the status of the network for current epoch
			for w in W_tens.eval():
				currStatus.append(w)

		# print information for the user about loss and weights
		if i % 20 == 0:
			print("Epoch: " + str(i) + "\t\tLoss: " + str(loss_value))
			print(W_train.eval()**-1)

			# add the weights to the status of the network for current epoch
			for w in W_tens.eval():
				currStatus.append(w)

			fileName = "/epoch" + str(i) + "_lossAndWeights.p"
			pickle.dump( currStatus, open( fileFolder + fileName, "wb" ) )




		# break from training if loss tolerance is reached
		if loss_value <= loss_tolerance:
			endCondition = '_belowLossTolerance_epoch' + str(i)
			print(endCondition)
			break

		loss_value_pre = loss_value

	
	

	merged = tf.summary.merge_all()
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	train_writer = tf.summary.FileWriter('/tmp/test_test/' + '/train',sess.graph)
	
	test_writer = tf.summary.FileWriter('/tmp/test_test/' + '/test')
	test_writer.add_run_metadata(run_metadata,'testtesttest')
	tf.global_variables_initializer().run()


