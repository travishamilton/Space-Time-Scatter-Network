import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from forward_fields import SCATTER_2_ELECTRIC_NODES

def TENSORIZE(field_vector,n_i,n_j,n_k,n_c):

    field_tensor = np.reshape(field_vector,(n_i,n_j,n_k,n_c))

    return field_tensor

def PLOT_BOUNDARIES(transfer_matrix,n_i,n_j,n_k,n_c,fig_num):

    boundary_tensor = TENSORIZE(transfer_matrix.diagonal(),n_i,n_j,n_k,n_c)

    for i in range(12):

        plt.figure(fig_num+i)

        plt.imshow(boundary_tensor[:,:,0,i])
        plt.title('boundaries')
        plt.colorbar()

def PLOT_TIME_SOURCE(time_source,n,alpha,fig_num):
#plots the source over time in terms of electric field components
#time_source - holds the sources scatter components at each time step - np.array, shape(n_t,n_c)
#n - refractive index at source location - float, shape(1,)
#alpha - alpha constant at source location for each axis - float, shape(3,)

    n_t,n_c = np.shape(time_source)

    electric0_time_source = np.zeros(n_t,dtype = float)
    electric1_time_source = np.zeros(n_t,dtype = float)
    electric2_time_source = np.zeros(n_t,dtype = float)

    for t in range(n_t):
        electric0_time_source_tmp , electric1_time_source_tmp , electric2_time_source_tmp = SCATTER_2_ELECTRIC_NODES(time_source[t,:],n_c,np.reshape(n,(1,1,1,1)),np.reshape(alpha,(1,1,1,3)))
        
        electric0_time_source[t] = np.squeeze(electric0_time_source_tmp)
        electric1_time_source[t] = np.squeeze(electric1_time_source_tmp)
        electric2_time_source[t] = np.squeeze(electric2_time_source_tmp)

    plt.figure(fig_num)
    plt.plot(np.arange(0,n_t,1),electric0_time_source ,'o',np.arange(0,n_t,1), electric1_time_source,'-*', np.arange(0,n_t,1), electric2_time_source)
    plt.legend(('E0', 'E1', 'E2'))



def PLOT_VIDEO(scatter_field_vector_time,n_c,n,alpha):

    fig = plt.figure()

    ims = []

    #get number of time steps
    _,n_t = np.shape(scatter_field_vector_time)

    n_x,n_y,n_z,_ = np.shape(alpha)

    for t in range(n_t):

        E0_tmp,E1_tmp,E2_tmp = SCATTER_2_ELECTRIC_NODES(np.squeeze(scatter_field_vector_time[:,t]),n_c,n,alpha)

        im = plt.imshow(E2_tmp[:,:,0], animated=True,vmin = -2, vmax = 2)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save("foward_movie.mp4")

    plt.show()

def PLOT_RESULTS(E0,E1,E2,n,fig_nums):
    #plot the last time step electric field and refractive index distribuiton.

    plt.figure(fig_nums[0])
    plt.imshow(E2[:,:,0])
    plt.title('E2')
    plt.colorbar()

    plt.figure(fig_nums[1])
    plt.imshow(E1[:,:,0])
    plt.title('E1')
    plt.colorbar()

    plt.figure(fig_nums[2])
    plt.imshow(E0[:,:,0])
    plt.title('E0')
    plt.colorbar()

    plt.figure(fig_nums[3])
    plt.imshow(n[:,:,0,0])
    plt.title('Refractive Index')
    plt.colorbar()

    plt.figure(1010)
    plt.plot(E2[5,:,0])
    plt.title('E2 - Cut')


    