import numpy as np

V_i_n = np.ones((3,3))
V_i_p = np.ones((3,3))
V_i_n[0,0] = 1

b = np.zeros((3,3),dtype = float)
d = np.ones((3,3),dtype = float)

C = np.zeros((3,3),dtype = float)

for i in range (3):
    for j in range(3):
        for k in range(3):
            C[i,k] = d[i,j]
            C[i,j] = 1-b[i,j]

V = np.zeros((3),dtype = float)
IZ = np.zeros((3),dtype = float)

V_r_n = np.zeros((3,3))
V_r_p = np.zeros((3,3))

for i in range(3):
    for k in range(3):
        for j in range(3):
            V[j] = C[i,j] * ( V_i_n[i,j] + V_i_p[i,j] ) + C[k,j] * ( V_i_n[k,j] + V_i_p[k,j] ) 
            IZ[k] = C[i,k] * ( V_i_p[i,j] - V_i_n[i,j] + V_i_n[j,i] - V_i_p[j,i] )

            V_r_n[i,j] = V[j] + IZ[k] - V_i_p[i,j]
            V_r_p[i,j] = V[j] - IZ[k] - V_i_n[i,j]

print(V_r_n)
print(V_r_p)

print(V_i_n)
print(V_i_p)