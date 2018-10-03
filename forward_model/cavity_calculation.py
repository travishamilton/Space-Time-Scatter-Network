import numpy as np

m = 3
n = 0
p = 1 #direction of propagation

c = 3.0e8
dl = 7.112e-3/4
dt = 2.965e-12
alpha = dl/dt/c

a = 30*dl
b = 1
d = 30*dl

ref_index = 1

for m in range(10):
    for p in range(10):
        f = 0.5*(c/ref_index)*np.sqrt( (m/a)**2 + (n/b)**2 + (p/d)**2 ) * 10**-9

        print('The frequency in GHz is: ' + str(f))

print('The alpha is: ' + str(alpha))