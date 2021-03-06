import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

#from fields import SCATTER_2_ELECTRIC_NODES

def PLOT_TIME_SOURCE(v_f,time_source,current_density,del_l,fig_num,location,del_t):
    #plots the source over time in terms of electric field components
    #time_source - holds the sources scatter components at each time step - np.array, shape(n_t,n_f)
    #del_l - space step in all three directions - float

    n_t,n_f = np.shape(time_source)
    eta_0 = 119.9169832*np.pi

    plt.figure(fig_num[0])
    plt.plot(np.arange(n_t)*del_t*1.0e15,-v_f[location[0],location[1],location[2],2,:]/del_l)
    plt.ylabel('V/m')
    plt.xlabel('fs')
    plt.title('free source fields at source location - z component')

    plt.figure(fig_num[1])
    plt.plot(np.arange(n_t)*del_t*1.0e15,time_source[:,2])
    plt.ylabel('V')
    plt.xlabel('fs')
    plt.title('free voltage source - z component')

    plt.figure(fig_num[2])
    plt.plot(np.arange(n_t)*del_t*1.0e15,current_density)
    plt.ylabel('A/m^2')
    plt.xlabel('fs')
    plt.title('current density')

def PLOT_TIME_SOURCE_LUMERICAL(v_f,fig_num,location):
    #plots the source over time in terms of electric field components
    #time_source - holds the sources scatter components at each time step - np.array, shape(n_t,n_f)

    plt.figure(fig_num)
    plt.plot(v_f[location[0],location[1],location[2],2])
    plt.title('free source fields - z component')


def PLOT_RESULTS_2(f_final,del_l,fig_num):

    plt.figure(fig_num)
    plt.plot(np.squeeze(-f_final/del_l), 'b')
    plt.ylabel('V/m')
    plt.title('final field value - z polarized')

def PLOT_VIDEO_1D(f_time,del_l,fig_num):

    fig = plt.figure(fig_num)

    ims = []

    #get number of time steps
    n_x,n_y,n_z,n_t = np.shape(f_time)

    for t in range(n_t):

        im, = plt.plot(np.squeeze(-f_time[:,:,:,t]/del_l), 'b',animated=True)
        plt.title('Lorentz Dielectric - Wrapping Boundary Conditions')
        plt.ylabel('V/m')

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    plt.show()

    ani.save("foward_movie.mp4")

def PLOT_COMPARISON_VIDEO_1D(f_time,f_lumerical_time,del_l,fig_num):

    fig = plt.figure(fig_num)

    ims = []

    #get number of time steps
    _,_,_,n_t = np.shape(f_time)

    for t in range(n_t):

        im1,im2 = plt.plot(np.squeeze(-f_time[:,:,:,t]/del_l), 'b',np.squeeze(-f_lumerical_time[:,:,:,t]/del_l),'r',animated=True)
        plt.title('Lorentz Dielectric - Wrapping Boundary Conditions')
        #plt.ylim(-2,2)

        ims.append([im1,im2])

    ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True,
                                    repeat_delay=1000)

    ani.save("foward_movie.mp4")
    plt.show()

def PLOT_VIDEO_2D(f_time,del_l,fig_num):

    fig = plt.figure(fig_num)

    ims = []

    #get number of time steps
    n_x,n_y,n_z,n_t = np.shape(f_time)

    for t in range(n_t):

        im, = plt.imshow(np.squeeze(-f_time[:,:,:,t]/del_l),animated=True)
        plt.colorbar

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

def PLOT_DISPERSION_PARAMETERS_1D(inf_x,w_0,damp,del_x,fig_num):
    #plot the dispersion parameters assuming number of field components equals 3
    # inf_x: high frequency susceptibility tensor - np.constant, shape(n_1)
    # w_0: Lorentz resonance frequency tensor (rad/s) - np.constant, shape(n_1,n_r)
    # damp: Lorentz damping frequency tensor (rad/s) - np.constant, shape(n_1,n_r)
    # del_x: change in susceptibility tensor (unitless) - np.constant, shape(n_1,n_r)
    # fig_num: numbers for figures - list, shape(2,)

    #get size of first spatial axis, number of electric field components and number of resonances
    n_1,n_r = np.shape(w_0)

    #susceptibility
    plt.figure(fig_num[0])
    plt.plot(inf_x,label = 'infinite susceptibility')

    #over all resonances
    for r in range(n_r):
        
        plt.plot(del_x[:,r],label = 'change in susceptibility: ' + str(r))
        plt.ylabel('Susceptibility')
        plt.xlabel('Simulation points')
        plt.title("Susceptibility for each Resonance")

    plt.legend()

    #resonant frequency
    plt.figure(fig_num[1])

    #over all resonances
    for r in range(n_r):
        
        plt.plot(w_0[:,r]*10**-12,label = 'resonance frequency: ' + str(r))
        plt.plot(damp[:,r]*10**-12,label = 'damping frequency: ' + str(r))
        plt.ylabel('THz')
        plt.xlabel('Simulation point')
        plt.title("frequency")
    
    plt.legend()

def PLOT_SPECTRUM_Z(fig_num,f_time,location,del_t):
    # fig_num: the figure number
    # f_time: the field over all time - shape(n_x,n_y,n_z,6,n_t)
    # location: the location of the point monitor - shape(3,)

    #get number of time steps
    _,_,_,_,n_t = np.shape(f_time)

    plt.figure(fig_num)

    #determine location
    x = location[0]
    y = location[1]
    z = location[2]

    time_signal = f_time[x,y,z,2,:]

    t = np.arange(0,del_t*n_t,del_t)
    sp = np.fft.fft(time_signal)
    freq = np.fft.fftfreq(t.shape[-1])
    l = len(freq)
    plt.plot(10**-12*freq[0:l//2-1]/del_t, np.abs(sp[0:l//2-1]))
    plt.xlabel('THz')
    plt.ylabel('Spectrum')
    plt.show()
    

    




    
