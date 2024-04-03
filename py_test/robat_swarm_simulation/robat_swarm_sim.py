# -*- coding: utf-8 -*-
"""
Robot DOA simulations
=====================
Here I'll test the efficacy of various DOA algorithms for our proposed
Ro-BAT microphone array

Created on Tue Jan 16 18:22:36 2024

@author: theja
"""
import pyroomacoustics as pra
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import scipy.signal as signal 
import soundfile as sf
import pandas as pd
import sounddevice as sd

#%%
# Robot is a 'hockey-puck' style thing with the speaker placed in the 
# centre and microphones placed on the periphery.

radius = 0.1 # m
mic_radius = radius + 2e-2
num_mics = 5 # placed from +90 to -90 deg
angular_posns = np.linspace(270, 90, num_mics)
mic_posns = np.column_stack((mic_radius*np.cos(np.radians(angular_posns)),
                             mic_radius*np.sin(np.radians(angular_posns))))


#%%
robo_centers = np.array([[0,1,1],
                         [1,0,1],
                         [0,0,1],
                         [1,1,1]]) # m
all_cylinders = []
sources = []
speaker_height = 2e-2
speaker_posns = []

for robo_center in robo_centers:
    robo_cylinder = pv.Cylinder(robo_center, direction=(0,0,1),
                                radius=radius, height=0.1)
    bounds = robo_cylinder.bounds
    speaker_xyz = [ np.mean(bounds[:2]), np.mean(bounds[2:4]) , bounds[-1] + speaker_height]
    speaker_posns.append(speaker_xyz)
    robo_cylinder.compute_normals(inplace=True, flip_normals=True)
    all_cylinders.append(robo_cylinder)
    sources.append(speaker_xyz)

room = pv.Box((-2,2,-2,2,0,2)) # m
room.extract_geometry().triangulate().save(f'room.stl')


source_idx = 0 # index of source robot
mic_oneset = mic_posns + robo_centers[source_idx][:2]
mic_oneset = np.column_stack((mic_oneset, np.tile(sources[source_idx][2], mic_oneset.shape[0])))

pd.DataFrame(mic_oneset, columns=['x','y','z']).to_csv('robot_sim_micxyz.csv')
pd.DataFrame(np.array(speaker_posns), columns=['x','y','z']).to_csv('robot_sources.csv')

#%% ROOM VISUALIZATION

scene = pv.Plotter() #create plotter instance
scene.add_mesh(room, opacity=0.2)
for i,every in enumerate(all_cylinders):
    scene.add_mesh(every)
    if i==source_idx:
        sphere_col = 'blue' 
    else:
        sphere_col = 'red'  
    scene.add_mesh(pv.Sphere(0.03, center=sources[i]), color=sphere_col) 
    every.extract_geometry().triangulate().save(f'robot_{i}.stl')

for i,each in enumerate(mic_oneset):
    scene.add_mesh(pv.Sphere(0.02, center=each), color='green')
    scene.add_mesh(pv.Text3D(str(i), center=each+np.array([0,0,2e-2]), width=0.02, height=0.02))

scene.show(cpos='xy')

# TODO:
# Add microphones in the scene, along with an emitter in the 'front' of each 
# hockey-puck robot.


#%% 
# Load material properties for sound absorption and scattering
material = pra.Material(energy_absorption=0.2, scattering=0.1) 

# Load mesh from STL file and get number of triangles
the_mesh = mesh.Mesh.from_file('room.stl')
ntriang, nvec, npts = the_mesh.vectors.shape

# Scale down mesh to get realistic room size 
size_reduc_factor = 1  

# Create PyRoomAcoustics wall for each triangle in mesh
# Apply material properties to all the surfaces
walls = []
for w in range(ntriang):
    walls.append(
        pra.wall_factory(
            the_mesh.vectors[w].T / size_reduc_factor, 
            material.energy_absorption["coeffs"],
            material.scattering["coeffs"],
        )
    )
    
# Add source sound emitter geometry later

for i in range(len(all_cylinders)):
    sphere_meshes = mesh.Mesh.from_file(f'robot_{i}.stl')
    ntriang, nvec, npts = sphere_meshes.vectors.shape
    
    for w in range(ntriang):
        walls.append(
            pra.wall_factory(
                sphere_meshes.vectors[w].T / size_reduc_factor,
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )


scenario_name = 'speaker_mics_and_cylinders'

#%%
# Load the room and sources

fs = int(192e3)
print('...initialising room...')
room = pra.Room(
            walls,
            fs=fs,
            max_order=1,
            ray_tracing=False,
            air_absorption=True,
        )

# call creation
call_durn = 1
t_call = np.linspace(0, call_durn, int(fs*call_durn))
chirp = signal.chirp(t_call, f0=6e3, f1=1e3, t1=t_call[-1])
chirp *= signal.windows.tukey(chirp.size, alpha=0.1)
chirp *= 0.75
print('...room initialised...')

# add sources to the room:
# The code is looping through all source indices except the one 
# already added (source_idx) and adding the other sources to the room
room.add_source(np.float64(sources[source_idx]), signal=chirp)
other_sources = set(range(len(sources))) - set([source_idx])
for i, othersource in enumerate(other_sources):
    print(i)
    room.add_source(np.float64(sources[othersource ]), signal=chirp, delay=5e-3*(i+1))

# now add all the other 'bat-calls' 

room.add_microphone_array(np.float64(mic_oneset.T))

room.image_source_model()
#room.ray_tracing()
room.compute_rir()
plt.figure()
room.plot_rir()
plt.suptitle('RIRs')

plt.savefig(f'rir.png')
room.simulate()
# show the room

audio = (room.mic_array.signals/20).T
sf.write('multi_robot_roomsim.wav', audio, samplerate=fs)

#%%

plt.figure()

plt.title('Spectrogram')
chnum = 4
f, t, sxx = signal.spectrogram(audio[:,chnum], fs=fs, nperseg=256, noverlap=200)
plt.specgram(audio[:,chnum], Fs=fs, cmap='viridis')
plt.ylim(0,10e3)
call_times = t<= call_durn
winsize = 96
rms_profile = np.sqrt(signal.convolve( audio[:,chnum]**2, np.ones(winsize), 'same')/winsize)
plt.savefig(f'spectrogram.png')


plt.figure()

a0 = plt.subplot(311)
plt.title('specgram')
plt.imshow(20*np.log10(sxx), origin='lower', aspect='auto', extent=[t[0], t[-1], 0, f[-1]])
plt.ylim(0,10e3)

a1 = plt.subplot(312, sharex=a0)
plt.title('rms power')
# The selected code is plotting the rms profile over time
plt.plot(np.linspace(0, rms_profile.size/fs, rms_profile.size), rms_profile)
plt.ylim(0,2)

plt.subplot(313, sharex=a0)
plt.title('rir')
log_rir = np.sqrt(np.abs(room.rir[chnum][0]))
plt.plot(np.linspace(0, log_rir.size/fs, log_rir.size), log_rir)
plt.savefig(f'{scenario_name}.png')


#%% CREATE AUDIBLE SOUNDS
# 
# # Downsample or convert frequency if necessary
# # For example, you can use signal.resample to downsample
# audible_signal = signal.resample(audio, len(audio)//2)
# 
# # Normalize the signal
# audible_signal /= np.max(np.abs(audible_signal))
# 
# # Play the audible sound
# sf.write('multi_robot_roomsim_audible.wav', audible_signal, samplerate=fs//4)
# # sd.play(audible_signal, fs//2)  # Assuming half the sample rate for audible sound
# # sd.wait()
# 
#%%
 
audiotype = 'audiofile'
audiofile_path = './resampled_ultra_audio.wav'

fs = sf.info(audiofile_path).samplerate
loaded_durn = 15 # seconds
audio_prova, fs = sf.read(audiofile_path, stop=int(fs*loaded_durn))

# Downsample or convert frequency if necessary
# For example, you can use signal.resample to downsample
audible_signal = signal.resample(audio_prova, len(audio_prova)//2)

# Normalize the signal
audible_signal /= np.max(np.abs(audible_signal))

# Play the audible sound
# sd.play(audible_signal, fs//2)  # Assuming half the sample rate for audible sound
# sd.wait()

sf.write('prova3.wav', audible_signal, samplerate=fs//8)

# %%