import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_x = 10
n_y = 10

mode_shape = np.array([1,2,3,4,5,4,3,2,1,0])
space_time_source = np.ones((n_x,n_y))
result = np.einsum('j,ij->ij',mode_shape,space_time_source)

print('result: ',result)