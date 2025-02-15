# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:53:59 2022

@author: Fearg
"""
import pandas as pd
import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt

Q = 0.7071

        

def Filter(Sig,Fs):
    if(Fs == 256):
        coeffA = np.asarray([ 0.41383847, -0.65971228,  0.58308188, -0.33109553])
        coeffB = np.asarray([0.00000000e+00, -1.79472566e-07, -1.08208428e-06, -2.32059491e-06,
               -4.04897556e-06, -5.77211064e-06, -6.62584334e-06, -5.74110432e-06,
               -2.45741536e-06,  3.47719435e-06,  1.17585948e-05,  2.14645230e-05,
                3.11000836e-05,  3.87570586e-05,  4.23752464e-05,  4.00761824e-05,
                3.05246026e-05,  1.32633556e-05, -1.10349779e-05, -4.04507584e-05,
               -7.18714120e-05, -1.01233473e-04, -1.23943748e-04, -1.35444092e-04,
               -1.31859642e-04, -1.10651425e-04, -7.11840434e-05, -1.51199420e-05,
                5.34355392e-05,  1.28090068e-04,  2.00639538e-04,  2.61841722e-04,
                3.02426847e-04,  3.14246554e-04,  2.91434319e-04,  2.31434960e-04,
                1.35761436e-04,  1.03551706e-05, -1.34539130e-04, -2.85024316e-04,
               -4.24709155e-04, -5.36294420e-04, -6.03448870e-04, -6.12789951e-04,
               -5.55753797e-04, -4.30130787e-04, -2.41059309e-04, -1.31153678e-06,
                2.69231765e-04,  5.44940427e-04,  7.96562024e-04,  9.93961954e-04,
                1.10924212e-03,  1.11994373e-03,  1.01200624e-03,  7.82153684e-04,
                4.39412459e-04,  5.53185699e-06, -4.85827101e-04, -9.91153741e-04,
               -1.46029574e-03, -1.84058667e-03, -2.08156066e-03, -2.13984297e-03,
               -1.98376254e-03, -1.59723278e-03, -9.82490946e-04, -1.61373232e-04,
                8.25075995e-04,  1.91871552e-03,  3.04803630e-03,  4.13325099e-03,
                5.09235108e-03,  5.84763568e-03,  6.33215573e-03,  6.49550769e-03,
                6.30845292e-03,  5.76593098e-03,  4.88816865e-03,  3.71974974e-03,
                2.32668959e-03,  7.91734519e-04, -7.91734519e-04, -2.32668959e-03,
               -3.71974974e-03, -4.88816865e-03, -5.76593098e-03, -6.30845292e-03,
               -6.49550769e-03, -6.33215573e-03, -5.84763568e-03, -5.09235108e-03,
               -4.13325099e-03, -3.04803630e-03, -1.91871552e-03, -8.25075995e-04,
                1.61373232e-04,  9.82490946e-04,  1.59723278e-03,  1.98376254e-03,
                2.13984297e-03,  2.08156066e-03,  1.84058667e-03,  1.46029574e-03,
                9.91153741e-04,  4.85827101e-04, -5.53185699e-06, -4.39412459e-04,
               -7.82153684e-04, -1.01200624e-03, -1.11994373e-03, -1.10924212e-03,
               -9.93961954e-04, -7.96562024e-04, -5.44940427e-04, -2.69231765e-04,
                1.31153678e-06,  2.41059309e-04,  4.30130787e-04,  5.55753797e-04,
                6.12789951e-04,  6.03448870e-04,  5.36294420e-04,  4.24709155e-04,
                2.85024316e-04,  1.34539130e-04, -1.03551706e-05, -1.35761436e-04,
               -2.31434960e-04, -2.91434319e-04, -3.14246554e-04, -3.02426847e-04,
               -2.61841722e-04, -2.00639538e-04, -1.28090068e-04, -5.34355392e-05,
                1.51199420e-05,  7.11840434e-05,  1.10651425e-04,  1.31859642e-04,
                1.35444092e-04,  1.23943748e-04,  1.01233473e-04,  7.18714120e-05,
                4.04507584e-05,  1.10349779e-05, -1.32633556e-05, -3.05246026e-05,
               -4.00761824e-05, -4.23752464e-05, -3.87570586e-05, -3.11000836e-05,
               -2.14645230e-05, -1.17585948e-05, -3.47719435e-06,  2.45741536e-06,
                5.74110432e-06,  6.62584334e-06,  5.77211064e-06,  4.04897556e-06,
                2.32059491e-06,  1.08208428e-06,  1.79472566e-07,  0.00000000e+00])
        
    elif(Fs == 200):
        coeffA = np.asarray([ 0.55248619, -0.54380776,  0.44751381, -0.44048428])
        coeffB = np.asarray([ 0.00000000e+00,  3.03809232e-07, -1.53790305e-06, -6.47049501e-06,
               -1.50431059e-05, -2.51369860e-05, -3.10225231e-05, -2.63743466e-05,
               -6.57966038e-06,  2.90482037e-05,  7.58333322e-05,  1.23634342e-04,
                1.58346147e-04,  1.64888042e-04,  1.31151385e-04,  5.20502834e-05,
               -6.73444378e-05, -2.10545919e-04, -3.50925408e-04, -4.55682890e-04,
               -4.92073922e-04, -4.34864574e-04, -2.73546955e-04, -1.76719332e-05,
                3.01185618e-04,  6.31813296e-04,  9.10305630e-04,  1.07084667e-03,
                1.05871097e-03,  8.43177342e-04,  4.27742498e-04, -1.44800546e-04,
               -7.94171609e-04, -1.41194040e-03, -1.87766011e-03, -2.07966995e-03,
               -1.93728317e-03, -1.42040511e-03, -5.62615434e-04,  5.35530879e-04,
                1.71483698e-03,  2.77740824e-03,  3.51611480e-03,  3.74957209e-03,
                3.35724416e-03,  2.30868410e-03,  6.81311385e-04, -1.33749771e-03,
               -3.46601958e-03, -5.36077095e-03, -6.66251653e-03, -7.04966566e-03,
               -6.29135435e-03, -4.29196081e-03, -1.11954011e-03,  2.98736040e-03,
                7.63707751e-03,  1.23254416e-02,  1.64950973e-02,  1.96059775e-02,
                2.12073658e-02,  2.10013478e-02,  1.88883398e-02,  1.49876951e-02,
                9.62980932e-03,  3.32017044e-03, -3.32017044e-03, -9.62980932e-03,
               -1.49876951e-02, -1.88883398e-02, -2.10013478e-02, -2.12073658e-02,
               -1.96059775e-02, -1.64950973e-02, -1.23254416e-02, -7.63707751e-03,
               -2.98736040e-03,  1.11954011e-03,  4.29196081e-03,  6.29135435e-03,
                7.04966566e-03,  6.66251653e-03,  5.36077095e-03,  3.46601958e-03,
                1.33749771e-03, -6.81311385e-04, -2.30868410e-03, -3.35724416e-03,
               -3.74957209e-03, -3.51611480e-03, -2.77740824e-03, -1.71483698e-03,
               -5.35530879e-04,  5.62615434e-04,  1.42040511e-03,  1.93728317e-03,
                2.07966995e-03,  1.87766011e-03,  1.41194040e-03,  7.94171609e-04,
                1.44800546e-04, -4.27742498e-04, -8.43177342e-04, -1.05871097e-03,
               -1.07084667e-03, -9.10305630e-04, -6.31813296e-04, -3.01185618e-04,
                1.76719332e-05,  2.73546955e-04,  4.34864574e-04,  4.92073922e-04,
                4.55682890e-04,  3.50925408e-04,  2.10545919e-04,  6.73444378e-05,
               -5.20502834e-05, -1.31151385e-04, -1.64888042e-04, -1.58346147e-04,
               -1.23634342e-04, -7.58333322e-05, -2.90482037e-05,  6.57966038e-06,
                2.63743466e-05,  3.10225231e-05,  2.51369860e-05,  1.50431059e-05,
                6.47049501e-06,  1.53790305e-06, -3.03809232e-07,  0.00000000e+00])
    
    return signal.lfilter(coeffB, coeffA, Sig,axis = -1)
        
def fast_resample(xf,fs_out,fs_in):
	#start_time = time.time()
	if fs_in!=fs_out:    	
        
		ti = np.arange(0,(len(xf))/fs_in,1/fs_in)
		to = np.arange(0,(len(xf))/fs_in,1/fs_out)
        
		if(len(ti) != len(xf)):
			ti = ti[0:len(xf)]
            
		y = np.interp(to, ti, xf)  

	else:
		y = xf
	#print("---%s seconds for resample---"%(time.time()-start_time))
	return y

# Turn Electrode Potentials into Bipolar Montage
def make_montage(df):
    if 'O1' in df.columns:
        data = {'time':df['time'],'F4-C4':df['F4'] - df['C4'],'C4-O2':df['C4'] - df['O2'],'F3-C3':df['F3'] - df['C3'],'C3-O1':df['C3'] - df['O1'],'C4-T4':df['C4'] - df['T4'], 'C4-Cz':df['C4'] - df['Cz'],'Cz-C3':df['Cz'] - df['C3'],'C3-T3':df['C3'] - df['T3']}
    else:
        data = {'time':df['time'],'F4-C4':df['F4'] - df['C4'],'C4-P4':df['C4'] - df['P4'],'F3-C3':df['F3'] - df['C3'],'C3-P3':df['C3'] - df['P3'],'C4-T4':df['C4'] - df['T4'], 'C4-Cz':df['C4'] - df['Cz'],'Cz-C3':df['Cz'] - df['C3'],'C3-T3':df['C3'] - df['T3']}

    tf = pd.DataFrame(data)
    
    return tf
    
def filt_resample(df):
    # Find sampling frequency
    if(df.size/256 == 3600):
        f_in = 256
    else:
        f_in = 200
    
    # 0.5 to 12.8 Hz bandpass with 50 Hz notch
    df = Filter(df, f_in)
        
    tf = fast_resample(df,32,f_in)
    
    return tf
    
# Wrapper Function for all preproccessing stages
def get_data(baby): # Baby = CSV File Name
    df = pd.read_csv(baby)

    df = df.rename(columns={'CZ':'Cz'})
    if 'FP1' in df.columns and 'F3' not in df.columns:
         df = df.rename(columns={'FP1':'F3'})
    if 'FP2' in df.columns and 'F4' not in df.columns:
         df = df.rename(columns={'FP2':'F4'})
    
    if 'O1' in df.columns:
        labels = ['F4-C4','C4-O2','F3-C3','C3-O1','C4-T4','C4-Cz','Cz-C3','C3-T3']
    else:
        labels = ['F4-C4','C4-P4','F3-C3','C3-P3','C4-T4','C4-Cz','Cz-C3','C3-T3']
    
    # Make Bi-polar Montage
    df = make_montage(df)
    
    rsamp = []

    # Downsample
    for label in labels:
         rsamp.append(filt_resample(df[label]))
    rsamp = np.array(rsamp)
         
    tf = pd.DataFrame(rsamp.T, columns = labels)
    return tf
    
