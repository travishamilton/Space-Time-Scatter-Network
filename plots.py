import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from fields import SCATTER_2_ELECTRIC_NODES

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
        electric0_time_source_tmp , electric1_time_source_tmp , electric2_time_source_tmp = SCATTER_2_ELECTRIC_NODES(time_source[np.newaxis,np.newaxis,np.newaxis,t,:],n_c,np.reshape(n,(1,1,1,1)),np.reshape(alpha,(1,1,1,3)))
        
        electric0_time_source[t] = np.squeeze(electric0_time_source_tmp)
        electric1_time_source[t] = np.squeeze(electric1_time_source_tmp)
        electric2_time_source[t] = np.squeeze(electric2_time_source_tmp)

    plt.figure(fig_num)
    plt.plot(np.arange(0,n_t,1),electric0_time_source ,'o',np.arange(0,n_t,1), electric1_time_source,'-*', np.arange(0,n_t,1), electric2_time_source)
    plt.legend(('E0', 'E1', 'E2'))

def PLOT_RESULTS(out_field,n,alpha,fig_nums):
    #plot the last time step electric field and refractive index distribuiton.
    # out_field: field at final time step over all space - np.array float, shape(n_x,n_y,n_z,n_c)
    # n: refractive index distribution over all space - np.array float, shape(n_x,n_y,n_z,n_w)
    # alpha: mesh constants for all space - np.array float, shape(n_x,n_y,n_z,3)
    # fig_nums: list of figure number to be used in plots - np.array int, shape(4,)

    _,_,_,n_c = np.shape(out_field)

    #get field polarizations
    E0,E1,E2 = SCATTER_2_ELECTRIC_NODES(out_field,n_c,n,alpha)

    #plot E2
    plt.figure(fig_nums[0])
    plt.imshow(E2[:,:,0])
    plt.title('E2')
    plt.colorbar()

    #plot E1
    plt.figure(fig_nums[1])
    plt.imshow(E1[:,:,0])
    plt.title('E1')
    plt.colorbar()

    #plot E0
    plt.figure(fig_nums[2])
    plt.imshow(E0[:,:,0])
    plt.title('E0')
    plt.colorbar()

    #plot refractive index
    plt.figure(fig_nums[3])
    plt.imshow(n[:,:,0,0])
    plt.title('Refractive Index')
    plt.colorbar()

def PLOT_RESULTS_1D(out_field,n,alpha,fig_nums):
    #plot the last time step electric field and refractive index distribuiton assuming spatail steps are equal to one in all directions except one
    # out_field: field at final time step over all space - np.array float, shape(n_x,n_y,n_z,n_c)
    # n: refractive index distribution over all space - np.array float, shape(n_x,n_y,n_z,n_w)
    # alpha: mesh constants for all space - np.array float, shape(n_x,n_y,n_z,3)
    # fig_nums: list of figure number to be used in plots - np.array int, shape(4,)
    # axis: axis that will be selected to plot over - int, shape(1,)

    _,_,_,n_c = np.shape(out_field)

    #get field polarizations
    E0,E1,E2 = SCATTER_2_ELECTRIC_NODES(out_field,n_c,n,alpha)

    #plot E2
    plt.figure(fig_nums[0])
    plt.plot(np.squeeze(E2))
    plt.title('E2')

    #plot E1
    plt.figure(fig_nums[1])
    plt.plot(np.squeeze(E1))
    plt.title('E1')

    #plot E0
    plt.figure(fig_nums[2])
    plt.plot(np.squeeze(E0))
    plt.title('E0')

    #plot refractive index
    plt.figure(fig_nums[3])
    plt.plot(np.squeeze(n))
    plt.title('Refractive Index')

def PLOT_VIDEO(scatter_field_vector_time,n_c,n,alpha,fig_num):

    fig = plt.figure(fig_num)

    ims = []

    #get number of time steps
    _,_,_,_,n_t = np.shape(scatter_field_vector_time)

    n_x,n_y,n_z,_ = np.shape(alpha)

    for t in range(n_t):

        E0_tmp,E1_tmp,E2_tmp = SCATTER_2_ELECTRIC_NODES(scatter_field_vector_time[:,:,:,:,t],n_c,n,alpha)

        im = plt.imshow(np.squeeze(E2_tmp),vmin = -2,vmax = 2,animated=True)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    #ani.save("foward_movie.mp4")
    plt.show()

def PLOT_VIDEO_1D(scatter_field_vector_time,n_c,n,alpha,fig_num):

    fig = plt.figure(fig_num)

    ims = []

    #get number of time steps
    _,_,_,_,n_t = np.shape(scatter_field_vector_time)

    n_x,n_y,n_z,_ = np.shape(alpha)

    for t in range(n_t):

        E0_tmp,E1_tmp,E2_tmp = SCATTER_2_ELECTRIC_NODES(scatter_field_vector_time[:,:,:,:,t],n_c,n,alpha)

        im, = plt.plot(np.squeeze(E2_tmp), 'b',animated=True)
        plt.ylim(-2,2)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    #ani.save("foward_movie.mp4")
    plt.show()

def PLOT_BOUNDARY_1D(boundary,fig_num):
    #plots boundary tensor assuming 1d simulation
    #boundary: boundary tensor - np.array data_type, shape (n_x,n_y,n_z,n_c)
    #fig_num: figure number - int, shape(1,)

    plt.figure(fig_num)
    plt.imshow(np.squeeze(boundary))
    plt.colorbar()
    plt.title('boundary')

    
