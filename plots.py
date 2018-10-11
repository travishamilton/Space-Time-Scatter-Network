import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

from fields import SCATTER_2_ELECTRIC_NODES

def PLOT_VIDEO(scatter_field_tensor_time,n,alpha):

    fig = plt.figure()

    ims = []

    #get number of time steps
    _,_,_,n_c,n_t = np.shape(scatter_field_tensor_time)

    for t in range(n_t):

        E0_tmp,E1_tmp,E2_tmp = SCATTER_2_ELECTRIC_NODES(scatter_field_tensor_time[:,:,:,:,t],n_c,n,alpha)

        im = plt.imshow(E2_tmp[:,:,0], animated=True,vmin = -2, vmax = 2)
        #im = plt.imshow(scatter_field_tensor_time[:,:,0,4,t], animated=True,vmin = -2, vmax = 2)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    plt.show()

    ani.save("movie.mp4")

    plt.plot(E2_tmp[:,30,0])

    plt.show()