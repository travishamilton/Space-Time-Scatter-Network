import numpy as np
from forward_fields import ADMITTANCE , COMPONENT_2_INDEX , NORMALIZED_CAPACITANCE

def SCATTER_2_ELECTRIC_NODES(scatter_field_tensor,n_c,n,alpha):
    #produces the electric field values at each node given the scatter components

    #get spatial parameters
    n_i,n_j,n_k,_ = np.shape(n)

    V_0 = np.zeros((n_i,n_j,n_k),dtype = float)
    V_1 = np.zeros((n_i,n_j,n_k),dtype = float)
    V_2 = np.zeros((n_i,n_j,n_k),dtype = float)

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):

                normalized_capacitance = NORMALIZED_CAPACITANCE(alpha[i,j,k,:],n[i,j,k,:])

                for c in range(n_c):
                    
                    direction,polarization,_ = COMPONENT_2_INDEX(c)

                    if polarization == 0:
                        V_0[i,j,k] = normalized_capacitance[direction,polarization]*scatter_field_tensor[i,j,k,c] + V_0[i,j,k]
                    elif polarization == 1:
                        V_1[i,j,k] = normalized_capacitance[direction,polarization]*scatter_field_tensor[i,j,k,c] + V_1[i,j,k]
                    elif polarization == 2:
                        V_2[i,j,k] = normalized_capacitance[direction,polarization]*scatter_field_tensor[i,j,k,c] + V_2[i,j,k]


    #Electric fields time time step delta t polarized in the zeroth, first, and second directions
    
    E_0 = V_0#/alpha[:,:,:,0]
    E_1 = V_1#/alpha[:,:,:,1]
    E_2 = V_2#/alpha[:,:,:,2]



    
    return E_0,E_1,E_2
