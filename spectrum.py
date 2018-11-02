import numpy as np
import tensorflow as tf

def SPECTRUM(field_time_0,field_time_1,field_time_2,n_f):
	#spectrum produces the frequency domain spectrum of the electric field time signal using fft
	# field_time_#: real valued field tensor in time domain to be translated into the frequency domain - tf.float32 shape (n_x,n_y,n_z,n_t)
	#	n_t - number of time steps
	#	_# - polarization number
	# n_f: number of frequency points for each polarization - tf.int32 shape(3)
	# field_freq_#: complex valued (complex64) frequency domain conversion of field_time_# - shape (n_x,n_y,n_z,n_f)

	#rfft will perform fft over first dimensions of a tensor

	#polarization 0
	field_freq_0 = tf.spectral.rfft(field_time_0)
	#polarization 1
	field_freq_1 = tf.spectral.rfft(field_time_1)
	#polarization 2
	field_freq_2 = tf.spectral.rfft(field_time_2)

	return field_freq_0 , field_freq_1 , field_freq_2

def LINE_MONITOR(position,long_axis,field_freq_0,field_freq_1,field_freq_2):

	x = position[0]
	y = position[1]
	z = position[2]

	if long_axis == 0:

		monitor_0 = field_freq_0[:,y,z]
		monitor_1 = field_freq_1[:,y,z]
		monitor_2 = field_freq_2[:,y,z]
	
	elif long_axis == 1:

		monitor_0 = field_freq_0[x,:,z]
		monitor_1 = field_freq_1[x,:,z]
		monitor_2 = field_freq_2[x,:,z]

	elif long_axis == 2:

		monitor_0 = field_freq_0[x,y,:]
		monitor_1 = field_freq_1[x,y,:]
		monitor_2 = field_freq_2[x,y,:]



