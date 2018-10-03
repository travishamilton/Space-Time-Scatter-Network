import numpy as np
import pickle

def GET_FIELD_DATA(file_address_fields , file_address_mesh):
    #gets the input/output scatter field based on user input
    #file_address: location of field data
    #X: input scatter field - numpy.ndarray shape (n_x,n_y,n_z,n_c)
    #Y: output scatter field - numpy.ndarray shape (n_x,n_y,n_z,n_c)
    #layers: number of layers in model. equal to the number of time steps there are.
    #mask_start: smallest coordinates of a cube containing the masked region - tuple int - shape(3,)
    #mask_end: largest coordinates of a cube containing the masked region - tuple int - shape(3,)

    # time_steps = input("Number of time steps: ")
    # space_steps_x = input("Number of space steps in x: ")
    # space_steps_y = input("Number of space steps in y: ")
    # space_steps_z = input("Number of space steps in z: ")
    # scatter_type = input("Scatter distribuiton type: ")
    # mask_start_x = input("Start of mask in x: ")
    # mask_start_y = input("Start of mask in y: ")
    # mask_start_z = input("Start of mask in z: ")
    # mask_end_x = input("End of mask in x: ")
    # mask_end_y = input("End of mask in y: ")
    # mask_end_z = input("End of mask in z: ")
    # time_changes = input("Number of time changes: ")

    time_steps = '100'
    space_steps_x = '100'
    space_steps_y = '100'
    space_steps_z = '1'
    scatter_type = 'cylinder'
    mask_start_x = '45'
    mask_start_y = '45'
    mask_start_z = '1'
    mask_end_x = '45'
    mask_end_y = '45'
    mask_end_z = '1'
    time_changes = '0'

    # file id
    file_id = "timeSteps_" + time_steps + "_spaceSteps_" + space_steps_x + "_" + space_steps_y + "_" + space_steps_z + "_scatterType_" + scatter_type + "_maskStart_" + mask_start_x + "_" + mask_start_y + "_" + mask_start_z + "_maskEnd_" + mask_end_x + "_" + mask_end_y + "_" + mask_end_z + "_timeChanges_" + time_changes

    # file name for fields data
    file_name_fields = file_address_fields + file_id

    # file name for mesh data
    file_name_mesh = file_address_mesh + file_id

    #open pickle file
    pkl_file = open(file_name_fields + '_in.pkl' , 'rb')
    X = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(file_name_fields + '_out.pkl' , 'rb')
    Y = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(file_name_mesh + '.pkl' , 'rb')
    mesh = pickle.load(pkl_file)
    pkl_file.close()

    #number of layers in model
    layers = int(time_steps)
    #corner coordiantes of a cube making up the masked region
    mask_start = (int(mask_start_x),int(mask_start_y),int(mask_start_z))
    mask_end = (int(mask_end_x),int(mask_end_y),int(mask_end_z))
    #simulation space
    n_x = int(space_steps_x)
    n_y = int(space_steps_y)
    n_z = int(space_steps_z)

    return X , Y , layers , mask_start , mask_end , n_x , n_y , n_z , mesh
