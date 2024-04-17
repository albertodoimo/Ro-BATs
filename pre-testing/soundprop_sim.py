# -*- coding: utf-8 -*-
"""
Simulating sound propagation to plan experiments and post-hoc checks
====================================================================
Simulating sound propagation is an extremely quick and effective way
to check if everything is as expected - before and after you've collected your
experimental data. 

Pyroomacoustics to the rescue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we'll use the ```pyroomacoustics``` package to simulate two scenarios
    1. A freefield simulation with a microphone array and multiple sources all around
    2. A room simulation with 1st order reflections activated and the same sources source

Run this module interactively in your favourite IDE step by step. Another alternative
is to export this .py file into a .ipynb file and run it as a Jupyter Notebook. 

Caveats
~~~~~~~
The reason pyroomacoustics and other ray-tracing packages are so fast and convenient
is that they use ray-tracing. Ray-tracing works well when the simulated volume is
much bigger than the wavelengths of the sounds being handled. If this condition
doesn't hold true - it is better to try other numerical methods like finite-difference
modelling etc. 


Author: Thejasvi Beleyur
License: MIT License
"""
import matplotlib.pyplot as plt
import numpy as np 
import pyroomacoustics as pra
import scipy.signal as signal 
from scipy.spatial.transform import Rotation
import soundfile as sf
np.random.seed(78464)

# Make a small room simulation with the tristar array (60 cm)
# Let's first define the room dimensions in metres

room_dims = [5,5,2] # x,y,z size metres
R = 0.6
micxyz = np.zeros((4,3))
micxyz[1,:] = [R*np.cos(60), 0, R*np.sin(30)]
micxyz[2,:] = [-R*np.cos(60), 0, R*np.sin(30)]
micxyz[3,:] = [0,0,R]

# Add some constants just so the coordinates remain with the room-dimensions (x,y,z >=0)
micxyz[:,0] += 1
micxyz[:,1] += 2  # move the array to a bit into the room.
micxyz[:,2] += 1.2 # raise the height of the array





#%%
# Create a free-field simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will specifically use the AnechoicRoom class - as this implements a situation
# with no reflections 

fs = 192000 # note the very high sampling rate of 192 kHz - because we need to 
# simulate ultrasound
freefield = pra.AnechoicRoom(  
             fs=fs)
freefield.add_microphone_array(micxyz.T)

# generate a flight trajectory

t = np.linspace(0,2048/fs,2048)

source_sound = signal.chirp(t,90000,t[-1],5000,'log')
source_sound *= signal.windows.tukey(source_sound.size, alpha=0.95)

n_sources = 20
t = np.linspace(0,2,n_sources)
x,y,z = 2*np.cos(2*np.pi*0.2*t), 2*np.sin(2*np.pi*0.2*t), np.tile(0.3,t.size)
source_points = np.column_stack((x,y,z))
source_points[:,1] = np.abs(source_points[:,1])
source_points[:,0] += 2

for i,each in enumerate(source_points):
    freefield.add_source(each, signal=source_sound, delay=i*0.1)

# run  simulation 
freefield.simulate()

#%%
# Visualise the mic array + known source points
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure()
a0 = plt.subplot(111, projection='3d')
a0.plot(micxyz[:,0],micxyz[:,1],micxyz[:,2],'r*')
a0.plot(source_points[:,0], source_points[:,1], source_points[:,2],'g*')
a0.set_xlim(0,room_dims[0])
a0.set_ylim(0,room_dims[1])
a0.set_zlim(0,room_dims[2])

#%%
# Visualise the output audio
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# pyroomacoustics audio data is formatted as a (Mchannels, Nsamples) np.array
# while most audio packages deal with audio as (Msamples, Nchannels). Just make 
# sure you know this!
#
# You can access the simulated audio in ```freefield.mic_array.signals```.

plt.figure()
plt.specgram(freefield.mic_array.signals[1,:],Fs=fs)

# Write the multichannel audio after transposition - to a file. 
sf.write('freefile_trista60cm.wav', freefield.mic_array.signals.T, samplerate=fs)
# store the simulated source points and microphone xyz data. 
np.savetxt('micxyz.csv', micxyz,delimiter=',')
np.savetxt('sources.csv', source_points,delimiter=',')


#%% 
# Implementing 1st order reflections 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's recreate the same sources emitting the same sounds - but this time with 
# 1st order reflections activated.


#%%
# Room geometry
# ~~~~~~~~~~~~~ 
# A pra.ShoeBox room refers to a cuboidal room geometry. Pyroomacoustics 
# allows you to also simulate arbitrary geometries in 2/3D - along with 
# also defining a volume using an already available mesh. pra.ShoeBox is 
# just the fastest and easiest example to show :)
# https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html


# increase the max_order to 2,3,4 etc and notice the effect it has on audio 
# - also notice the run-time difference!!
roomsim = pra.ShoeBox(  
             room_dims,fs=fs, max_order=1)
roomsim.add_microphone_array(micxyz.T)

# generate random points at various distances 

for i,each in enumerate(source_points):
    roomsim.add_source(each, signal=source_sound, delay=i*0.1)


# Compute the impule response ofthe room - watch for this additional step 
roomsim.compute_rir()
# run  simulation 
roomsim.simulate() 

# save file 
sf.write('roomsim_trista60cm.wav', roomsim.mic_array.signals.T, samplerate=fs)

#%%
# Visualise the same sources but with reverberation included

plt.figure()
plt.specgram(roomsim.mic_array.signals[1,:],Fs=fs)

#%% 
# Compare the effect of including reflectiosn in the simulations 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure()
a0 = plt.subplot(211)
plt.specgram(freefield.mic_array.signals[1,:int(fs*0.25)], Fs=fs)
a0.set_title('Free-field simulation')
a1 = plt.subplot(212, sharex=a0, sharey=a0)
plt.specgram(roomsim.mic_array.signals[1,:int(fs*0.25)], Fs=fs)
a1.set_title('With 1st order reflections active')
a1.set_xlabel('Time (s)', fontsize=12)
