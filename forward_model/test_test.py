import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from layers import VECTORIZE , TENSOR_INDEX_TO_VECTOR_INDEX

n_i = 10
n_j = 10
n_k = 1
n_c = 12

A = np.zeros((n_i,n_j,n_k,n_c),dtype = np.float32)

tensor_index = (1,2,0,8)

A[tensor_index] = 1

B = VECTORIZE(A)

vector_index = TENSOR_INDEX_TO_VECTOR_INDEX(tensor_index,n_i,n_j,n_k,n_c)

print('vector index: ', vector_index)

vector_index_2 = np.flatnonzero(B)

print('vector index 2: ', vector_index_2)