import numpy as np
import tensorflow as tf
import os

from files import *
from weights import *

import pickle

data_type = tf.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well

# ----------------------- Simulation Constants ------------------#
n_c = 12    # number of field components per node
n_w = 1     # number of weights per node

#------------------------ Read in Data --------------------------#
with tf.name_scope('read_data'):
    file_address = "C:/Users/travi/Documents/Northwestern/STSN/forward_model/field_data/"
    in_field , out_field , layers , mask_start , mask_end , n_x , n_y , n_z = GET_FIELD_DATA(file_address)

    sample_n, feat_n = in_field.shape	# sampN: number of training samples, featN: features per sample

#------------------------ Create Weights ------------------------#
with tf.name_scope('create_weights'):
    weight_tens = WEIGHT_CREATION(mask_start, mask_end, layers, data_type, n_x, n_y, n_z , n_w)

#--------------------------- Placeholder Instantiation --------------------------#
with tf.name_scope('instantiate_placeholders'):
    in_field_tens = tf.placeholder(dtype = data_type, shape = [n_x,n_y,n_z])
    out_field_tens = tf.placeholder(dtype = data_type, shape = [n_x,n_y,n_z])

#--------------------------- Cost Function Definition --------------------------#
# compute least squares cost for each sample and then average out their costs
print("Building Cost Function (Least Squares) ... ... ...")

with tf.name_scope('cost_function'):
    
    pre_out_field_tens = PROPAGATE(in_field_tens,mesh,n_c,weights_tens,layers,n_x,n_y,n_z,n_w) # prediction function
    
    least_squares = tf.norm(pre_out_field_tens-out_field_tens, ord=2,name='least_squre')**2 	#

print("Done!\n")

#--------------------------- Define Optimizer --------------------------#
print("Building Optimizer ... ... ...")
lr = 0.01
with tf.name_scope('train'):
	train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(least_squares, var_list = [weights_train_tens])
with tf.name_scope('clip'):
	clip_op = tf.assign(weights_train_tens, tf.clip_by_value(weights_train_tens, 0, 1.0))
print("Done!\n")

#--------------------------- Merge Summaries ---------------------------#
merged = tf.summary.merge_all()

#--------------------------- Training --------------------------#
epochs = 10
loss_tolerance = 1e-10
table = []

# saves objects for every iteration
fileFolder = "results/" + file_id

# if the results folder does not exist for the current model, create it
if not os.path.exists(fileFolder):
		os.makedirs(fileFolder)


with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run( tf.global_variables_initializer() )

    print("Tensor in field:")		# show info. for in field
    print(in_field)
    print("")

    print("Tensor out field: ")		# show info. for out field
    print(out_field)
    print("")

    print("--------- Starting Training ---------\n")

    train_writer = tf.summary.FileWriter('/STSN/training summaries', sess.graph)
    tf.global_variables_initializer().run()
    
    for i in range(1, epochs+1):

        # run X and Y dynamically into the network per iteration and add runtime data to tensorboard summaries
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _,loss_value = sess.run([train_op, least_squares], feed_dict = {in_field_tens: in_field, out_field_tens: out_field},options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata,'epoch '+str(i))
        # perform clipping 
        #with tf.name_scope('clip'):
         #   sess.run(clip_op)

        print('Loss: ',loss_value)

        w = sess.run(weights_train_tens)
        print('weights: ' , np.squeeze(w))

        # break from training if loss tolerance is reached
        if loss_value <= loss_tolerance:
            endCondition = '_belowLossTolerance_epoch' + str(i)
            print(endCondition)
            break



merged = tf.summary.merge_all()
