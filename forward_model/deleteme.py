import numpy as np
import scipy.sparse as sp

m = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.shape(m))
c = np.array([0,1,2])
c.shape = (3,1)
print(c)
print(np.matmul(m , c))


a = sp.csr_matrix(m) @ c

print('a:',a)

for i in range(1):
    print('test')