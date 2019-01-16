import tkinter
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pickle as pkl
import tensorflow as tf
import matplotlib

from files import *
from weights import *
from layers import *
from parameters import REFRACTIVE_INDEX , ALPHA
from main import *

matplotlib.use('TkAgg')

os.system('cls' if os.name == 'nt' else 'clear')

data_type = tf.float32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

tf.reset_default_graph()							#reset tensorflow

np.random.seed(7)		# seeding the random number generator to reproduce identical results

class GuiThing(tkinter.Tk):
	#inputs:

    def __init__(self):
        tkinter.Tk.__init__(self)
        
        #add a label
        tkinter.Label(self, text="STSN").grid(column=0,row=0, columnspan=2,rowspan=1)
        
        #add a button and an input field, grouped in a subframe
        subframe = tkinter.Frame(self,bd=2,relief='groove',padx=5,pady=5)
        subframe.grid(row=1,column=0,rowspan=1,columnspan=2)
        self.button1 = tkinter.Button(subframe,text="run fwd 1D",command=self.Press_Forward_1D).grid(column=0,row=21)
        self.button2 = tkinter.Button(subframe,text="run fwd",command=self.Press_Forward).grid(column=1,row=21)
        self.button3 = tkinter.Button(subframe,text="train 1D",command=self.Press_Train_1D).grid(column=2,row=21)
        #create entry fields and organize in grid
        self.field1 = tkinter.Entry(subframe)
        self.field2 = tkinter.Entry(subframe)
        self.field3 = tkinter.Entry(subframe)
        self.field4 = tkinter.Entry(subframe)
        self.field5 = tkinter.Entry(subframe)
        self.field6 = tkinter.Entry(subframe)
        self.field7 = tkinter.Entry(subframe)
        self.field8 = tkinter.Entry(subframe)
        self.field9 = tkinter.Entry(subframe)
        self.field10 = tkinter.Entry(subframe)
        self.field11 = tkinter.Entry(subframe)
        self.field12 = tkinter.Entry(subframe)
        self.field13 = tkinter.Entry(subframe)
        self.field14 = tkinter.Entry(subframe)
        self.field15 = tkinter.Entry(subframe)
        self.field16 = tkinter.Entry(subframe)
        self.field17 = tkinter.Entry(subframe)
        self.field18 = tkinter.Entry(subframe)
        self.field19 = tkinter.Entry(subframe)
        self.field20 = tkinter.Entry(subframe)
        self.field21 = tkinter.Entry(subframe)
        self.field1.grid(column=1,row=0)
        self.field2.grid(column=1,row=1)
        self.field3.grid(column=1,row=2)
        self.field4.grid(column=1,row=3)
        self.field5.grid(column=1,row=4)
        self.field6.grid(column=1,row=5)
        self.field7.grid(column=1,row=6)
        self.field8.grid(column=1,row=7)
        self.field9.grid(column=1,row=8)
        self.field10.grid(column=1,row=9)
        self.field11.grid(column=1,row=10)
        self.field12.grid(column=1,row=11)
        self.field13.grid(column=1,row=12)
        self.field14.grid(column=1,row=13)
        self.field15.grid(column=1,row=14)
        self.field16.grid(column=1,row=15)
        self.field17.grid(column=1,row=16)
        self.field18.grid(column=1,row=17)
        self.field19.grid(column=1,row=18)
        self.field20.grid(column=1,row=19)
        self.field21.grid(column=1,row=20)

        #create labels
        tkinter.Label(subframe, text="n_x:").grid(column=0,row=0,sticky="E")
        tkinter.Label(subframe, text="n_y:").grid(column=0,row=1,sticky="E")
        tkinter.Label(subframe, text="n_z:").grid(column=0,row=2,sticky="E")
        tkinter.Label(subframe, text="n_t:").grid(column=0,row=3,sticky="E")
        tkinter.Label(subframe, text="time changes:").grid(column=0,row=4,sticky="E")
        tkinter.Label(subframe, text="scatter type:").grid(column=0,row=5,sticky="E")
        tkinter.Label(subframe, text="mask: [[").grid(column=0,row=6,sticky="E")
        tkinter.Label(subframe, text="],").grid(column=2,row=6,sticky="W")
        tkinter.Label(subframe, text="[").grid(column=0,row=7,sticky="E")
        tkinter.Label(subframe, text="]]").grid(column=2,row=7,sticky="W")
        tkinter.Label(subframe, text="weights per node:").grid(column=0,row=8,sticky="E")
        tkinter.Label(subframe, text="initial_weight:").grid(column=0,row=9,sticky="E")
        tkinter.Label(subframe, text="location: [").grid(column=0,row=10,sticky="E")
        tkinter.Label(subframe, text="]").grid(column=2,row=11,sticky="W")
        tkinter.Label(subframe, text="polarization:").grid(column=0,row=11,sticky="E")
        tkinter.Label(subframe, text="wavelength:").grid(column=0,row=12,sticky="E")
        tkinter.Label(subframe, text="injection axis:").grid(column=0,row=13,sticky="E")
        tkinter.Label(subframe, text="injection direction:").grid(column=0,row=14,sticky="E")
        tkinter.Label(subframe, text="FWHM:").grid(column=0,row=15,sticky="E")
        tkinter.Label(subframe, text="mode FWHM:").grid(column=0,row=16,sticky="E")
        tkinter.Label(subframe, text="n_m:").grid(column=0,row=17,sticky="E")
        tkinter.Label(subframe, text="mode center: [").grid(column=0,row=18,sticky="E")
        tkinter.Label(subframe, text="]").grid(column=2,row=18,sticky="W")
        tkinter.Label(subframe, text="mode axis:").grid(column=0,row=19,sticky="E")
        tkinter.Label(subframe, text="source type:").grid(column=0,row=20,sticky="E")

        
        #add a "message board", by itself in another subframe
        #subframe = tkinter.Frame(self,bd=2,relief='groove',padx=5,pady=5)
        #subframe.grid(column=5,row=0,columnspan=2,rowspan=10)
        #self.messagetext = tkinter.StringVar(self,"Message Board")
        #tkinter.Button(subframe, width=40, height=12,textvariable = self.messagetext,
                      #command = self.ClearMessages, relief='flat',anchor='n',
                      #wraplength=330, justify='left').grid(column=0,row=0,columnspan=2,rowspan=10)

    def INTAKE(self):
        n_i = self.field1.get()
        n_j = self.field2.get()
        n_k = self.field3.get()
        time_steps = self.field4.get()
        tc = self.field5.get()
        scatter_type = self.field6.get()
        rawstr7 = self.field7.get()
        rawstr8 = self.field8.get()
        rawstr9 = self.field9.get()
        rawstr10 = self.field10.get()
        rawstr11 = self.field11.get()
        rawstr12 = self.field12.get()
        rawstr13 = self.field13.get()
        rawstr14 = self.field14.get()
        rawstr15 = self.field15.get()
        rawstr16 = self.field16.get()
        rawstr17 = self.field17.get()
        rawstr18 = self.field18.get()
        rawstr19 = self.field19.get()
        rawstr20 = self.field20.get()
        source_type = self.field21.get()

        n_x = int(n_i.strip(" "))
        n_y = int(n_j.strip(" "))
        n_z = int(n_k.strip(" "))
        n_t = int(time_steps.strip(" "))
        time_changes = int(tc.strip(" "))
        mask_row_1 = np.fromstring(rawstr7,dtype=int,count=-1,sep=',')
        mask_row_2 = np.fromstring(rawstr8,dtype=int,count=-1,sep=',')
        mask = np.array([mask_row_1,mask_row_2])
        n_w = int(rawstr9.strip(" "))
        initial_weight = float(rawstr10.strip(" "))
        location = np.fromstring(rawstr11,dtype=int,count=-1,sep=',')
        polarization = int(rawstr12.strip(" "))
        wavelength = int(rawstr13.strip(" "))
        injection_axis = int(rawstr14.strip(" "))
        injection_direction = int(rawstr15.strip(" "))
        fwhm = int(rawstr16.strip(" "))
        fwhm_mode = int(rawstr17.strip(" "))
        n_m = float(rawstr18.strip(" "))
        center_mode = np.fromstring(rawstr19,dtype=float,count=-1,sep=',')
        mode_axis = int(rawstr20.strip(" "))

        return n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,n_w,initial_weight,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type

    #Meant to be called when Button1 is pressed
    def Press_Forward_1D(self):
        n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,n_w,initial_weight,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type = self.INTAKE()
        n_c = 12

        FORWARD_1D(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type)
     
    def Press_Forward(self):
        n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,n_w,initial_weight,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type = self.INTAKE()
        n_c = 12    # number of field components per node
        FORWARD(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type)

    def Press_Train_1D(self):
        n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,n_w,initial_weight,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type = self.INTAKE()
        n_c = 12
        INVERSE_1D(n_c,n_w,initial_weight,n_x,n_y,n_z,n_t,time_changes,scatter_type,mask,location,polarization,wavelength,injection_axis,injection_direction,fwhm,fwhm_mode,n_m,center_mode,mode_axis,source_type)

    	   #self.messagetext.set("parameters: \n\nn_x = %i\nn_y = %i\nn_z = %i\nn_t = %i\ntime changes = %i\nscatter type = %s\nmask = %0.3f\ntotal n value = %0.3f\n\n(click to clear)" %(n_x,n_y,n_z,n_t,tc,scatter_type,mask,n_sum))

    #Meant to be called when the button that is the message board is pressed
    #def ClearMessages(self):
        #self.messagetext.set("Message Board")
        
        
####----THE THING WHICH ACTUALLY RUNS THE GUI----####
master = GuiThing()
master.mainloop() #window doesn't appear until this line is run
