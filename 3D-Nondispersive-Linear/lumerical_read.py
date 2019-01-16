
import numpy as np

address = 'lumerical_data/'

def readFields(n_f):
    # read i/o field data given by Lumerical and return a 1D array of field values
    
    # #get field data
    # file_id = 'Lumerical_Fields_1D_imag.csv' 	#file id
    # fileName_input = address + file_id						# name of file including location relative to current file
    
    # Et = np.genfromtxt(fileName_input, delimiter = ',')

    file_id = 'Lumerical_Fields_1D_real.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    Et = np.genfromtxt(fileName_input, delimiter = ',')

    #get time signal
    # file_id = 'Lumerical_Time_Signal_1D_imag.csv' 	#file id
    # fileName_input = address + file_id						# name of file including location relative to current file
    
    # time_signal = np.genfromtxt(fileName_input, delimiter = ',')

    file_id = 'Lumerical_Time_Signal_1D_real.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    time_signal = np.genfromtxt(fileName_input, delimiter = ',')
    
    #get Lorentz materail properties
    file_id = 'Lumerical_Lorentz_Perm_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    l_perm = np.genfromtxt(fileName_input, delimiter = ',')
    
    file_id = 'Lumerical_Lorentz_Res_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    w0 = np.genfromtxt(fileName_input, delimiter = ',')

    file_id = 'Lumerical_Lorentz_Line_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    l_lin = np.genfromtxt(fileName_input, delimiter = ',')

    file_id = 'Lumerical_Perm_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    perm = np.genfromtxt(fileName_input, delimiter = ',')

    file_id = 'Lumerical_Source_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    source_parameters = np.genfromtxt(fileName_input, delimiter = ',')
    polarization = source_parameters[0]
    location = [0,source_parameters[1].astype(int),0]
    del_l = source_parameters[2]
    del_t = source_parameters[3]

    #give space parameters
    file_id = 'Lumerical_Source_1D.csv' 	#file id
    fileName_input = address + file_id						# name of file including location relative to current file
    
    #source_parameters = np.genfromtxt(fileName_inpu
    n_x = 1
    n_z = 1
    n_y,n_t = np.shape(Et)

    return Et[np.newaxis,:,np.newaxis,:].astype(np.float32),time_signal.astype(np.float32),l_perm[np.newaxis,:,np.newaxis].astype(np.float32),w0[np.newaxis,:,np.newaxis,:].astype(np.float32),l_lin[np.newaxis,:,np.newaxis,:].astype(np.float32),perm[np.newaxis,:,np.newaxis,:].astype(np.float32),polarization.astype(int),location,del_l,del_t,n_x,n_y,n_z,n_t


    