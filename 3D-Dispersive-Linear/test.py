import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as p
from scipy import signal

#State-space operators 
A = np.array([[0, 1], [1, 2]])
A = np.array([[ 0.   ,      1. ,        0. ,        0. ,        0. ,        0. ,      ],
 [ 0.   ,      0. ,        1. ,        0.  ,       0.  ,       0.    ],
 [ 0.  ,       0. ,        0. ,        1.    ,     0.    ,     0.       ],
 [ 0.      ,   0.  ,       0. ,        0.  ,       1.  ,       0.       ],
 [ 0.     ,    0.   ,      0.   ,      0. ,        0.   ,      1.       ],
 [-0.5378974  ,2.981432 , -7.4844556 ,10.768508 , -9.31365 ,   4.5834928]])
B = np.array([[0], [1]])  # 2-dimensional column vector
B = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
C = np.array([[1, 2]])    # 2-dimensional row vector
C = np.array([[0.51050293, -2.1508305, 3.8449132, -3.5855603, 1.6916916, -0.30827755]])
D = np.array([1])
D = np.array([[-0.94907117]])
print(np.shape(A))
print(np.shape(B))
print(np.shape(C))
print(np.shape(D))

#get zeros and poles
z,p,k = signal.ss2zpk(A, B, C, D)
print('zeros: ',z)
print('zeros mag: ',np.abs(z))
print('poles mag: ',np.abs(p))

#number of time steps
n_t = 3000

#set input, initilize state variables to zero
alpha = 1.2223
beta = 3.334
u1 = np.float32(np.random.rand(n_t))
u2 = np.float32(np.random.rand(n_t))
u = u1*alpha + u2*beta
y1 = np.zeros(n_t)
y2 = np.zeros(n_t)
y = np.zeros(n_t)

#run state space system n_t times
x = np.array([[0], [0], [0], [0], [0], [0]])
for t in range(n_t):
    x_next = A@x + B*u1[t]*alpha
    y1[t] = C@x + D*u1[t]*alpha
    x = x_next

x = np.array([[0], [0], [0], [0], [0], [0]])
for t in range(n_t):
    x_next = A@x + B*u2[t]*beta
    y2[t] = C@x + D*u2[t]*beta
    x = x_next

x = np.array([[0], [0], [0], [0], [0], [0]])
for t in range(n_t):
    x_next = A@x + B*u[t]
    y[t] = C@x + D*u[t]
    x = x_next

plt.figure(1)
plt.plot(np.arange(0,n_t,1),y,label = 'output')
plt.plot(np.arange(0,n_t,1),u,label = 'input')
plt.legend()

plt.figure(2)
plt.plot(y - y1 - y2)

plt.show()