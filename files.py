import numpy as np
import pickle

def GET_DATA(file_path , file_id):

    #open field in pickle file
    with open(file_path + file_id + '_in.pkl', 'rb') as f:
        fields_in = pickle.load(f)

    #open field out pickle file
    with open(file_path + file_id  + '_out.pkl', 'rb') as f:
        fields_out = pickle.load(f)
    
    #open mesh pickle file
    with open(file_path + file_id  + '_alpha.pkl' , 'rb') as f:
        mesh = pickle.load(f)

    #open refractive index pickle file
    with open(file_path + file_id  + '_ref_ind.pkl' , 'rb') as f:
        ref_index = pickle.load(f)

    return fields_in , fields_out , mesh , ref_index
