import numpy as np
import matplotlib.pyplot as plt

sample_rate = 192000
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitter signal
t = np.arange(N)/sample_rate # time vector
f_tone = 4000
tx = np.exp(2j * np.pi * f_tone * t)


d = 0.5 # half wavelength spacing
Nr = 4  # number of elements in the array
theta_degrees = 20 # direction of arrival (feel free to change this, it's arbitrary)
theta = theta_degrees / 180 * np.pi # convert to radians
a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # array factor
print(a) # note that it's 3 elements long, it's complex, and the first element is 1+0j

a = a.reshape(-1,1)
print("a dimension =",a.shape) # 3x1
tx = tx.reshape(-1,1)
print("tx dimension =",tx.shape) # 10000x1

# matrix multiply
r = a @ tx.T  # dont get too caught up by the transpose, the important thing is we're multiplying the array factor by the tx signal
print("r dimension =", r.shape) # 3x10000.  r is now going to be a 2D array, 1D is time and 1D is the spatial dimension

fig1 = plt.figure()
plt.plot(np.asarray(r[0,:]).squeeze().real[0:50]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
plt.plot(np.asarray(r[1,:]).squeeze().real[0:50])
plt.plot(np.asarray(r[2,:]).squeeze().real[0:50])
plt.plot(np.asarray(r[3,:]).squeeze().real[0:50])
plt.show() 

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N) 
r = r + 0.5*n # r and n are both 3x10000
print("r =",r)
print("r dimension =", r.shape)



# DAS = DElAY AND SUM BEAMFORMER
w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Conventional, aka delay-and-sum, beamformer
print("w =",w)
print("w shape =", w.shape )
r_1 = w.conj().T @ r # example of applying the weights to the received signal (i.e., perform the beamforming)
print("r shape =",r.shape)

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
   w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
   r_weighted = w.conj().T @ r # apply our weights. remember r is 3x10000
   results.append(10*np.log10(np.var(r_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
results -= np.max(results) # normalize

# print angle that gave us the max value
print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

fig2 = plt.figure() 
plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()
plt.show()  

# POLAR PLOT 

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetagrids(np.arange(0, 360, 15)) # number of grid lines
plt.show()  



# MUSIC = MUltiple SIGNal Classification
num_expected_signals = 1 # Try changing this! must be < Nr = number of elements in the array

# part that doesn't change with theta_i
R = r @ r.conj().T # Calc covariance matrix, it's Nr x Nr
w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
v = v[:, eig_val_order] # sort eigenvectors using this order
# We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64)
for i in range(Nr - num_expected_signals):
   V[:, i] = v[:, i]

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # -180 to +180 degrees
results = []
for theta_i in theta_scan:
    a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # array factor
    a = a.reshape(-1,1)
    metric = 1 / (a.conj().T @ V @ V.conj().T @ a) # The main MUSIC equation
    metric = np.abs(metric.squeeze()) # take magnitude
    metric = 10*np.log10(metric) # convert to dB
    results.append(metric)

results /= np.max(results) # normalize

# print angle that gave us the max value
print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()
plt.show()  

# POLAR PLOT 
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetagrids(np.arange(0, 360, 15)) # number of grid lines
plt.show()  

