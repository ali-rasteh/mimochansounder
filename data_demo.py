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

# Get the file path for the calibration data
fn = 'siso_trx_cabled.npz'
calib_path = os.path.join(os.getcwd(), 'data')
calib_path = os.path.join(calib_path, fn)

# Get the file path for the calibration data
fn = 'siso_trx_OTA.npz'
data_path = os.path.join(os.getcwd(), 'data')
data_path = os.path.join(data_path, fn)

# Create a deconvolution object
dec = Deconv(fc=fc, fsamp=fsamp, nfft=nfft,
             ntx=ntx, nrx=nrx)

# Set the system response
dec.set_system_resp_data(path=calib_path)

# Compute the channel response
dec.load_chan_data(data_path)
dec.compute_chan_resp()


# Perform the sparse estimation
dec.sparse_est(npaths=3, nframe_avg=1, drange=[-6,32], 
               ndly=5000)


# Plot the raw response
dly = np.arange(nfft) 
dly = dly - nfft*(dly > nfft/2)
dly = dly / fsamp
chan_pow = 20*np.log10(np.abs(dec.chan_td_avg))

# Roll the response and shift the response
rots = 32
yshift = np.percentile(chan_pow, 25)
chan_powr = np.roll(chan_pow, rots) - yshift
dlyr = np.roll(dly, rots)
plt.plot(dlyr[:128]*1e9, chan_powr[:128])
plt.grid()

# Compute the axes
ymax = np.max(chan_powr)+5
ymin = -10

# Plot the locations of the detected peaks
scale = np.mean(np.abs(dec.grxfd))**2
peaks  = 10*np.log10(np.abs(dec.coeffs_est)**2 * scale  )-yshift
plt.stem(dec.dly_est*1e9, peaks, 'r-', basefmt='', bottom=ymin)
plt.ylim([ymin, ymax])
plt.xlabel('Delay [ns]')
plt.ylabel('SNR [dB]')

plt.savefig('peaks.png')