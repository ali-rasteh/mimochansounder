# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:57:08 2024

@author: sdran
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from deconv import Deconv

# Parameters
fc = 6e9   # Carrier frequency
fsamp = 1e9 # Sample freq in Hz
nantrx=2    # Number of antennas per array
nfft=1024   # Number of FFT points
nrx = 1     # Number of RX antennas
ntx = 1     # Number of TX antennas
nframe_avg = 1 # number of frames for averaging
file_version = 1  # Set to 1 for the latest version

# Get the file path for the calibration data
fn = 'chamber.npz'
calib_path = os.path.join(os.getcwd(), 'data')
calib_path = os.path.join(calib_path, fn)

# Get the file path for the calibration data
fn = 'wall_reflection.npz'
data_path = os.path.join(os.getcwd(), 'data')
data_path = os.path.join(data_path, fn)

# Create a deconvolution object
dec = Deconv(fc=fc, fsamp=fsamp, nfft=nfft,
             ntx=ntx, nrx=nrx, file_version=file_version)

h_est = dec.load_data(calib_path)
hpow = 10*np.log10(np.abs(h_est)**2)

# Set the system response
dec.set_system_resp_data(path=calib_path,
                         nframe_avg=nframe_avg)

# Compute the channel response
dec.load_chan_data(data_path)
dec.compute_chan_resp()


# Perform the sparse estimation
dec.sparse_est(npaths=10, nframe_avg=nframe_avg,
               drange=[-32,64], ndly=5000, cv=True)
    
    
# Plot the raw response
dly = np.arange(nfft) 
dly = dly - nfft*(dly > 0.9*nfft)
dly = dly / fsamp
chan_pow = 20*np.log10(np.abs(dec.chan_td_tr))

# Roll and shift the response
rots = 32  # Rotation 
yshift = np.percentile(chan_pow, 25)
ntaps = 128
chan_powr = np.roll(chan_pow, rots) - yshift
dlyr = np.roll(dly, rots)
plt.plot(dlyr[:ntaps]*1e9, chan_powr[:ntaps])
plt.grid()

# Compute the axes
ymax = np.max(chan_powr)+5
ymin = -10

# Plot the locations of the detected peaks
scale = np.mean(np.abs(dec.grxfd))**2
peaks  = 10*np.log10(np.abs(dec.coeffs_est)**2 * scale  )-yshift
plt.stem(dec.dly_est*1e9, peaks, 'r-', basefmt='', bottom=ymin)
plt.ylim([ymin, ymax])
plt.xlabel('Relative delay [ns]')
plt.ylabel('SNR [dB]')

plt.savefig('peaks.png')
