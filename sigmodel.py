import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
import matplotlib.pyplot as plt

class RoomModel(object):
    def __init__(self, xlim=np.array([-10,10]), ylim=np.array([-3,10])):

        # Define the walls of the room
        # Each row is a wall with the first two columns being the start points
        # of the wall and the last two columns being the end point 
        self.walls = np.array([[xlim[0],ylim[0],xlim[0],ylim[1]],
                               [xlim[0],ylim[1],xlim[1],ylim[1]],
                                 [xlim[1],ylim[1],xlim[1],ylim[0]] ] )
        
    def draw_walls(self, ax=None):
        """
        Draw the walls of the room.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to draw the walls on.  If None, the current axis is used.
        """
        if ax is None:
            ax = plt.gca()
        for i in range(self.walls.shape[0]):
            ax.plot(self.walls[i,[0,2]], self.walls[i,[1,3]], 'b-', linewidth=2)

    def find_reflection(self, x):
        """
        Find all the reflections of a point x in the room.
        """
        nwall = self.walls.shape[0]
        xref = np.zeros((nwall,2))
        for i in range(nwall):
            xref[i,:] = self.reflect_point(x, self.walls[i,:])

        return xref

    def reflect_point(self, x, wall):
        """
        Reflect a point x off a wall.
        """
        x1 = wall[0:2]
        x2 = wall[2:4]
        v = x2 - x1
        v = v/np.linalg.norm(v)
        n = np.array([-v[1], v[0]])
        d = np.dot(x - x1, n)
        xref = x - 2*d*n
        return xref

class Sim(object):
    """
    Describes the configuration of the simulation.

    Parameters
    ----------
    fc : float
        Carrier frequency.
    fsamp : float
        Sampling frequency.
    bw : float
        Sounding bandwidth relative to the sample rate.
    nfft : int
        Number of FFT points.
    snr : float
        Signal to noise ratio in dB.
    nantrx : int
        Number of antennas at the RX.
    antsep : nparray of floats
        Separation between antennas in wavelengths.
    rxlocsep : nparray of floats
        Maximum separation between RX locations in meters
    sepdir : np.array of shape (p,)
        Direction of separation between antennas and locations
    nrxloc : int
        Number of RX locations.  The RX locations are placed on a line
        from the origin the rxlocsep.  At each rx location, the RX antennas are
        separated by antsep.  So, the total number of RX measurements is
        nmeas = nrxloc*len(antsep).
    npath : int
        Number of paths from the TX which is assumed to be the 
        same for all RX locations.
    lossavg : float
        Average loss of the paths in dB.
    tx : (npath, p) array or None
        List of the TX locations.  If None, the TX locations are randomly
        generated in the region.
    npath_est : int
        Number of paths to estimate.
    stop_thresh : float
        Relative peak size at which the search stops
    region : np.array of shape (2,p) or None
        Region in which the TX locations are generated.  If None, the region
        is set to include all the TX image points.
    """
    def __init__(self, fc=6e9, fsamp=1e9, bw=0.8, nfft=1024, snr=20, nantrx=2,
                antsep=None, rxlocsep=None, sepdir=None, npath=4,
                lossavg=10, tx=None, npath_est=4, stop_thresh=0.03, region=None):
        # Parameters
        self.fc = fc  # Carrier frequency
        self.c = 3e8  # Speed of light
        self.lam = self.c/self.fc  # Wavelength
        self.fsamp = fsamp  # Sampling frequency
        self.bw = bw    # Sounding bandwidth relative to the sampling rate
        self.nfft = nfft  # Number of FFT points
        self.snr = snr  # Signal to noise ratio in dB
        self.nantrx = nantrx # Number of antennas at the RX
        if antsep is None:
            antsep=np.array([0.5,1,2])
        self.antsep = antsep
        if rxlocsep is None:
            rxlocsep = np.array([0,1])  
        if sepdir is None:
            sepdir=np.array([1,0])
        self.sepdir = sepdir
        self.rxlocsep = rxlocsep
        self.npath = npath
        self.sepdir = sepdir
        self.nrxlocsep = len(self.rxlocsep)
        self.nantsep = len(self.antsep)
        self.nmeas = self.nrxlocsep * self.nantsep
        self.lossavg = lossavg
        self.tx = tx
        self.npath_est = npath_est
        self.stop_thresh = stop_thresh
        self.region = region


        # Generate the TX positions and search region
        self.gen_tx_pos()

        # Compute the RX locations and antenna positions
        self.compute_rx_pos()

        # Compute the channel frequency response
        self.compute_freq_resp()

        # Create the test points for the TX locations
        self.create_tx_test_points()

        # Initialize the path estimates
        self.path_est_init()

        self.locate_tx()
       
    def gen_tx_pos(self):
        """
        Generate random TX image positions and reflection losses
        from each TX
        """
        # Generate the TX locations
        if self.tx is None:
            # Region for the transmitter
            if self.region is None:
                self.region = np.array([[-10,10], [0,30]] )  
            self.p = self.region.shape[1]  # Dimension of region

            wid = self.region[:,1] - self.region[:,0]
            self.tx = np.random.uniform(0,1, (self.npath, self.p))*wid[None,:]\
                    + self.region[None,:,0]
            
        else:

            # Set region to include all the TX image points
            self.p = self.tx.shape[1]
            if self.region is None:
                xmin = np.min(self.tx, axis=0)
                xmax = np.max(self.tx, axis=0)
                self.region = np.array([[xmin[0]-5, xmax[0]+5], [0, xmax[1]+5]])
            self.npath = self.tx.shape[0]

        # Sort the TX locations in distance from the origin
        dist = np.linalg.norm(self.tx, axis=1)
        idx = np.argsort(dist)
        self.tx = self.tx[idx,:]

        # Generate the reflection losses in dB
        # The first path has zero loss since it is LOS
        self.loss = np.random.exponential(self.lossavg, (self.npath,))
        self.loss[0] = 0



    def compute_rx_pos(self):
        """
        Compute the locations of the RX antennas across all measurements.

        rxantpos[i,m,:] is the location of the i-th RX antenna in the m-th
        measurement
        """
        # Generate the RX locations
        self.rxloc = self.rxlocsep[:,None]*self.sepdir[None,:]

        # Generate the RX antenna positions
        self.rxantpos = np.zeros((self.nantrx, self.nmeas, self.p))
        for k in range(self.nrxlocsep):
            for i in range(self.nantsep):
                m = k*self.nantsep + i

                # Linear distance of the RX antennas from the origin
                t = self.rxlocsep[k] + self.antsep[i]*np.arange(self.nantrx)*self.lam

                # Position of the RX antennas
                self.rxantpos[:,m,:] = t[:,None]*self.sepdir[None,:]
            
        

    def compute_freq_resp(self):
        """
        Computes the frequency response from all TX locations at all measurements


        chan[:,i,m,itx] is the channel frequency response from 
        TX location itx, measurement m to RX antenna i.

        distref[m] is the reference distance for measurement m.  This is the time
        at which the measurement was taken and changes from measurement to measurement
        since the measurements are not synchronized.

        distrel[i,m,itx] is the relative distance from TX location itx to RX antenna i
        """
 

        # Compute the frequency response of the channel sounder
        nbw = int(self.bw*self.nfft/2)
        self.gresp = np.zeros(self.nfft, dtype=np.complex64)
        self.gresp[0:nbw+1] = 1
        self.gresp[self.nfft-nbw:] = 1

        # Frequency of each FFT bin
        self.freq = self.fc + (np.arange(self.nfft)/self.nfft-0.5)*self.fsamp
        
        # Compute the distance from each TX location to each RX antenna
        #  dist[i,m,itx] is the distance from TX location itx to RX antenna i
        #  in measurement m
        self.dist = self.rxantpos[:,:,None,:] - self.tx[None,None,:,:]
        self.dist = np.sum(self.dist**2, axis=3)**0.5

        # Compute the free space path loss using Friis' formula
        #  pathloss[i,m,itx] is the path loss from TX location itx to RX antenna i
        #  in measurement m
        self.pathloss = -20*np.log10(self.lam/(4*np.pi*self.dist))   

        # Add the reflection losses
        self.pathloss += self.loss[None,None,:]

        # Compute the channel coefficients from the path loss values
        self.coeffs = 10**(-self.pathloss/20)
        
        # Add random phase to the channel coefficients
        phase = np.exp(1j*2*np.pi*
                  np.random.uniform(0,1,(self.nmeas,self.npath)))
        self.coeffs = self.coeffs*phase[None,:,:]

        # Find the reference distance for each measurement from the first path
        dsamp = self.c/self.fsamp  # Distance traveled in one sample period
        dnom = 10*dsamp
        self.distref = np.mean(self.dist[:,:,0], axis=0) - dnom + np.random.uniform(-0.5,0.5,self.nmeas)*dsamp
        self.distrel = self.dist - self.distref[None,:,None]

        # Compute the frequency-domain channel response
        self.chan = self.mpath_chan(self.distrel, self.coeffs)

        # Find the power from the strongest path
        chan_pow = np.mean(np.abs(self.chan[:,:,0])**2)
        wvar = chan_pow*10**(-self.snr/10)

        # Add noise to the channel
        self.chan_fd = self.chan + np.random.normal(0, np.sqrt(wvar/2), self.chan.shape) + \
            1j*np.random.normal(0, np.sqrt(wvar/2), self.chan.shape)

        # Compute the time-domain channel response
        self.chan_td = np.fft.ifft(self.chan_fd, axis=0)

    def plot_chan_td(self, indsamp=None, indrx=0, indmeas=0):
        """
        Plots the channel response in the time domain.

        Parameters
        ----------
        indsamp : slice
            The range of samples to plot.
        indrx : int or slice
            The indices of the RX antenna to plot.
        indmeas : int or slice
            The indices of the measurement to plot.
        """

        # Set default sample range to the entire range
        if indsamp is None:
            indsamp = slice(0,self.nfft)

        # Compute the expected path locations in samples
        self.path_samp = self.distrel / self.c * self.fsamp
        self.path_pow = 20*np.log10(np.abs(self.coeffs))
        self.chan_pow =  20*np.log10(np.abs(self.chan_td))

        # Find the max value of y for the plot
        samp_ind = np.arange(self.nfft)
        chan_pow1 = self.chan_pow[indsamp,indrx,indmeas]
        path_samp1 = self.path_samp[indrx,indmeas,:]
        path_pow1 = self.path_pow[indrx,indmeas,:]
        ymax = np.max([np.max(chan_pow1), np.max(path_pow1)]) + 5

        # Find the 25% percentile value of of the chan_pow1 to estimate the noise
        ymin = np.percentile(chan_pow1.flatten(), 25)-5
       
        # Plot the channel power
        plt.plot(samp_ind[indsamp], chan_pow1)
        plt.stem(path_samp1.flatten(), path_pow1.flatten(), 
                 'r-', bottom=ymin, basefmt=' ')
        plt.grid()
        plt.ylim([ymin, ymax])


    def mpath_chan(self, dist, coeffs=None, basis_fn=False):
        """
        Creates the channel frequency response for a given set of delays
        and coefficients.  If the `basis_fn` is True, then the coefficients
        are ignored and the method returns the basis functions.

        Parameters
        ----------
        coeffs : np.array of shape (d1,d2,...,dk,npath)
            The coefficients of the path.
        dist : np.array of shape (d1,d2,...,dk,npath) 
            The relative distances of the paths

        Returns
        -------
        h : np.array of shape (nfft,d1,...,dk) 
            The channel frequency response.
        """
        #t = np.arange(self.nfft)
        
        # Reshape the distance to add a dimension at the beginning
        dist1 = dist.reshape((1,) + dist.shape)
        if basis_fn:
            coeffs1 = 1
        else:
            coeffs1 = coeffs.reshape((1,) + dist.shape)

        # Reshape the frequency to add dimensions at the end
        ndims = len(dist.shape)
        freq1 = self.freq.reshape(self.freq.shape + (1,)*ndims)
        gresp1 = self.gresp.reshape(self.gresp.shape + (1,)*ndims)
        
        # Compute the phase
        phase = 2*np.pi*freq1*dist1 / self.c
        h = gresp1*coeffs1*np.exp(-1j*phase)

        # If we are not computing the basis functions, we sum over the 
        # final dimension corresponding to the paths
        if not basis_fn:
            h = np.sum(h, axis=-1)

        return h


        
    def plot_tx_rx(self):
        """
        Plot the TX and RX locations.
        """
        # Plot the RX locations as red circles
        plt.plot(self.rxloc[:,0], self.rxloc[:,1], 'ro')

        # Plot the TX locations as blue circles
        plt.plot(self.tx[:,0], self.tx[:,1], 'bo')

        #plt.grid(True)
        #plt.show()

    def create_tx_test_points(self):
        """
        Create a set of test points over the region using meshgrid.
        """

        # Create a set of points x uniformly distributed in the area
        # using meshgrid
        npoints = 100
        x1 = np.linspace(self.region[0,0],self.region[0,1],npoints)
        x2 = np.linspace(self.region[1,0],self.region[1,1],npoints)
        X1, X2 = np.meshgrid(x1,x2)
        X = np.zeros((npoints**2,2))
        X[:,0] = X1.flatten()
        X[:,1] = X2.flatten()
        self.xtest0 = X1
        self.xtest1 = X2
        self.Xtest = X
        self.npoints = npoints

        # Find the distances from each TX test location to each RX antenna
        dist = self.rxantpos[:,:,None,:] - X[None,None,:,:]
        self.dtest = np.sqrt(np.sum(dist**2, axis=3))

    def path_est_init(self):
        """
        Initializes the path estimate data structures
        """

        # For each measurement, find the peak location of the channel response
        # and circularly rotate the response to that location, so the peaks
        # all align to sample 0
        self.chan_td_rot = np.zeros((self.nfft, self.nantrx, self.nmeas), dtype=np.complex64)
        self.chan_fd_rot = np.zeros((self.nfft, self.nantrx, self.nmeas), dtype=np.complex64)
        f = np.arange(self.nfft)[:,None]
        for i in range(self.nmeas):
            chan_pow = np.sum(np.abs(self.chan_td[:,:,i])**2, axis=1)
            im = np.argmax(chan_pow)
            self.chan_td_rot[:,:,i] = np.roll(self.chan_td[:,:,i], -im, axis=0)
            self.chan_fd_rot[:,:,i] = self.chan_fd[:,:,i]*np.exp(1j*2*np.pi*im*f/self.nfft)

        # Estimated reference distance for each measurement.
        # This is a measure of the point at which the measurement was made
        self.distref_est = np.zeros(self.nmeas)

        # Estimate TX locations, absolute distance, and coefficients for each path
        self.tx_est = np.zeros((self.npath_est, self.p))
        self.dist_est = np.zeros((self.nantrx, self.nmeas, self.npath_est))
        self.coeffs_est = np.zeros((self.nmeas, self.npath_est), dtype=np.complex64)

    def locate_tx(self):
        """
        Iteratively locates the TX image points
        """
        

        # Initialize the correlation estimates for each iteration
        ntest = self.dtest.shape[-1]
        self.rho = np.zeros((ntest, self.npath_est))

        # Compute the phase difference
        dexp = np.exp(2*np.pi*1j/self.lam*self.dtest)

        for ipath in range(self.npath_est):
        
            if ipath == 0:

                # For each measurement find the peak of the channel response
                # and take the response at that sample frequency
                aresp = self.chan_td_rot[0,:,:]

                # Compute the correlation
                self.rho[:,ipath] = np.sum( np.abs(np.sum(aresp[:,:,None]*dexp, axis=0))**2, axis=0 )

            else:
                                   
                aresp = np.zeros((self.nantrx, self.nmeas, ntest), dtype=np.complex64)
                for i in range(ntest):
                    for m in range(self.nmeas):
                
                        # Get the array response from the appropriate sample offset
                        isamp = self.samp_offset[m,i]
                        aresp[:,m,i] = self.resid[isamp,:,m]

                # Compute the correlation
                self.rho[:,ipath] = np.sum( np.abs(np.sum(aresp*dexp, axis=0))**2, axis=0 )


            # Find the location that maximizes the correlation
            im = np.argmax(self.rho[:,ipath])

            # Check stopping condition
            if ipath == 0:
                rho_max = np.max(self.rho[:,0])
                done = False
            else:
                done = np.max(self.rho[:,ipath]) < self.stop_thresh*rho_max

            if done:               
                break
            else:
                self.npath_det = ipath+1

            self.tx_est[ipath,:] = self.Xtest[np.argmax(self.rho[:,ipath]),:]

            # Set the disances for the path
            self.dist_est[:,:,ipath] = self.dtest[:,:,im] 

            # For the first path, we estimate the reference distance
            # and the sample locations of the other paths
            if ipath == 0:   
                self.set_ref_distances()

            # Fit the coefficients and find the residual
            self.fit_coeffs(npath_fit=ipath+1)

    def set_ref_distances(self):
        """
        Set the reference distances for each measurement.

        Computes

        samp_offset[m,i] = sample offset for test location i in measurement m
        """

        # Set the reference distance
        self.distref_est = np.mean(self.dist_est[:,:,0], axis=0)

        # Compute the sample offset for all other distances
        dtest_mean = np.mean(self.dtest, axis=0)
        self.samp_offset = (dtest_mean - self.distref_est[:,None])/self.c*self.fsamp
        self.samp_offset = np.round(self.samp_offset).astype(int)
        self.samp_offset = np.maximum(0, self.samp_offset)
        self.samp_offset = np.minimum(self.nfft-1, self.samp_offset)





    def fit_coeffs(self, npath_fit=1):
        
        # Compute the relative distances of the paths
        dist_rel = self.dist_est[:,:,:npath_fit] - self.distref_est[None,:,None]

        # Compute the basis functions
        self.basis = self.mpath_chan(dist_rel, basis_fn=True)

        # Stack the basis functions for the antennas 
        self.basis1 = self.basis.reshape((self.nfft*self.nantrx, self.nmeas, npath_fit), order='F')

        # Stack the channel responses for the antennas
        chan_fd_rot1 = self.chan_fd_rot.reshape((self.nfft*self.nantrx, self.nmeas), order='F')

        # On each measurement, find the least squares estimate of the coefficients
        self.coeffs_est = np.zeros((self.nmeas, npath_fit), dtype=np.complex64)
        for i in range(self.nmeas):
            self.coeffs_est[i,:] = np.linalg.lstsq(self.basis1[:,i,:], chan_fd_rot1[:,i], rcond=None)[0]

        # Compute channel from the sum of the estimated paths
        self.chan_fd_paths = np.sum(self.coeffs_est[None,:,:]*self.basis1, axis=2)
        self.chan_fd_paths = self.chan_fd_paths.reshape((self.nfft, self.nantrx, self.nmeas), order='F')

        # Compute the time-domain channel response
        self.chan_td_paths = np.fft.ifft(self.chan_fd_paths, axis=0)

        # Compute the residual 
        self.resid = self.chan_td_rot - self.chan_td_paths
        #self.resid = np.sum(self.coeffs_est[:,None,:]*self.basis1, axis=2) - chan_fd_rot1
        #self.resid = np.sum(np.abs(np.sum(self.coeffs_est[:,None,:]*self.basis, axis=2) - chan_fd_rot1)**2, axis=0)



   