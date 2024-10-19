# deconv.py:  methods for deconvolving an channel response
import numpy as np
import os

class Deconv(object):
    def __init__(self, nfft=1024, fsamp=1e9, fc=6e9, nrx=None, ntx=None):
        """
        Parameters
        ----------

        nfft : int
            Number of FFT points.
        fsamp : float
            Sample rate in Hz
        fc : float
            Carrier frequency in Hz
        nrx : int
            Number of RX antennas.  
        ntx : int
            Number of TX antennas
        dly_test : ndarray
            Delays to test in seconds
        """
        self.nfft = nfft
        self.fsamp = fsamp
        self.fc = fc
        self.nrx = nrx
        self.ntx = ntx

        # Set the default system response
        self.grxfd = None
        self.gtxfd = None

        # Frequency of each FFT bin
        self.freq = (np.arange(self.nfft)/self.nfft)*self.fsamp + self.fc - self.fsamp/2

        
            
    def load_data(self, path):
        """
        Load TX and RX data from a file

        Returns
        -------
        txtd : ndarray (nfft,ntx)
            Time-domain TX data
        rxtd : ndarray (nfft,nrx,nframe)
            Time-domain RX data
        """

        # Check if the file exists
        if not os.path.exists(path):
            raise ValueError('File does not exist')

        # Load the file
        data = np.load(path)
        txtd = data['txtd']
        rxtd = data['rxtd']

        # Rearrange the indices so the order (nfft,nrx,nframe)
        rxtd = np.transpose(rxtd, (2,1,0))
        txtd = np.transpose(txtd, (2,1,0))
        txtd = txtd[:,:,0]

        # Reduce the number of antennas if needed
        if (rxtd.shape[1] > self.nrx):
            rxtd = rxtd[:,:self.nrx,:]
        if (txtd.shape[1] > self.ntx):
            txtd = txtd[:,:self.ntx]

  
        return txtd, rxtd

    def set_system_resp_data(self, path):
        """
        Set the system response from a file.  The MIMO response will be 
        two diagonal transfer functions:

        grx[:,irx] = RX frequency response for RX antenna irx
        gtx[:,itx] = TX frequency response for TX antenna itx

        Parameters
        ----------
        path : str
            Path to the file
        """
        txtd, rxtd = self.load_data(path)

        # Compute the TX freq domain
        txfd = np.fft.fft(txtd, axis=0)  # (nfft,ntx)
        rxfd = np.fft.fft(rxtd, axis=0)  # (nfft,nrx,nframe)

        # Noise level assumed for the MMSE 
        snr = 40
        wvar = np.mean(np.abs(rxfd)**2)*(10**(-snr/10))

        # Compute the RX calibration response.
        # This only works for ntx=1.
        # Also, we do not average over multiple frames
        S = np.conj(txfd[:,0])/(np.abs(txfd[:,0])**2 + wvar)
        self.grxfd = S[:,None]*rxfd[:,:,0]
        self.gtxfd = np.ones((self.nfft, self.ntx), dtype=np.complex64)

        # Compute the time-domain response
        self.grxtd = np.fft.ifft(self.grxfd, axis=0)
        self.gtxtd = np.fft.ifft(self.gtxfd, axis=0)

    def load_chan_data(self, path):
        """
        Load the channel data

        Parameters
        ----------
        path : str
            Path to the file
        """
        txtd, rxtd = self.load_data(path)

        # Save the response
        self.txtd = txtd
        self.rxtd = rxtd

    def compute_chan_resp(self):
        """
        Compute the channel response

        """
        # Compute the TX freq domain
        txfd = np.fft.fft(self.txtd, axis=0)  # (nfft,ntx)
        rxfd = np.fft.fft(self.rxtd, axis=0)  # (nfft,nrx,nframe)

        # Noise level assumed for the MMSE 
        snr = 40
        wvar = np.mean(np.abs(rxfd)**2)*(10**(-snr/10))

        # Initialize the MIMO response
        nframe = self.rxtd.shape[2]
        self.chan_fd = np.zeros((self.nfft, self.nrx, self.ntx, nframe),
                                dtype=np.complex64)

        # Compute the MIMO response via the Wiener filter
        for itx in range(self.ntx):
            for irx in range(self.nrx):
                g = txfd[:,itx]*self.gtxfd[:,itx]*self.grxfd[:,irx]
                S = np.conj(g)/(np.abs(g)**2 + wvar)
                self.chan_fd[:,irx,itx,:] = S[:,None]*rxfd[:,irx,:]

        # Compute the time-domain response
        self.chan_td = np.fft.ifft(self.chan_fd, axis=0)

    def set_system_resp(self, grxfd):
        """
        Set the system response

        Parameters
        ----------
        g : ndarray (nfft,)
            System response

        """
        self.g = g

    def set_system_rect_resp(self, nsc, dc_block=True):
        """
        Set the system response to a rectangular window

        Parameters
        ----------
        nsc : int
            Number of used subcarriers
        dc_block: bool 
            If True, block the DC subcarrier
        """
        nsc1 = nsc // 2
        self.g = np.zeros(self.nfft, dtype=np.complex64)
        self.g[:nsc1] = 1
        self.g[-nsc1:] = 1
        if dc_block:
            self.g[nsc1] = 0


    def create_chan_resp(self, dly, coeffs=None, basis=False, irx=0, itx=0):
        """
        Create the channel response

        Parameters
        ----------
        dly : ndarray (npaths,)
            Delays in seconds
        coeffs : ndarray (npaths,)
            Complex coefficients
        basis : bool
            If True, return the basis functions

        Returns
        -------
        h : ndarray (nfft,)
            Channel response
        """
        B = self.grxfd[:,irx,None]*self.gtxfd[:,itx,None]*np.exp(-2*np.pi*1j*self.freq[:,None] * dly[None,:])
        if basis:
            return B
        else:
            h = B.dot(coeffs)
            return h

    def sparse_est(self, npaths=1, nframe_avg=1, ndly=10000, drange=[-6,20]):
        """
        Estimate the channel response using sparse deconvolution

        Parameters
        ----------
        hfd : ndarray (nfft,)
            Frequency-domain channel response
        npaths : int
            Number of paths to estimate
        nframe_avg :int
            Number of frames to average
       
        """

        # Compute the initial channel esitmate via averaging
        # over the frames
        self.chan_fd_avg = np.mean(self.chan_fd[:,0,0,:nframe_avg], axis=1)
        self.chan_td_avg = np.fft.ifft(self.chan_fd_avg, axis=0)

        # Set the delays to test around the peak
        idx = np.argmax(np.abs(self.chan_td_avg))
        self.dly_test = (idx + np.linspace(drange[0],drange[1],ndly))/self.fsamp

        # Create the basis vectors
        self.B = self.create_chan_resp(self.dly_test, basis=True)

        # Use OMP to find the sparse solution
        self.coeff_est = np.zeros(npaths)
        
        resid = self.chan_fd_avg
        indices = []
        self.mse = np.zeros(npaths)
        for i in range(npaths):
            # Compute the correlation
            cor = np.abs(self.B.conj().T.dot(resid))

            # Add the highest correlation to the list
            idx = np.argmax(cor)
            indices.append(idx)

            # Use least squares to estimate the coefficients
            self.coeffs_est = np.linalg.lstsq(self.B[:,indices], self.chan_fd_avg, rcond=None)[0]

            # Compute the resulting sparse channel
            self.chan_fd_sparse = self.B[:,indices].dot(self.coeffs_est)
            
            resid = self.chan_fd_avg - self.chan_fd_sparse

            self.mse[i] = np.mean(np.abs(resid)**2)/np.mean(np.abs(self.chan_fd_avg)**2)

        self.dly_est = self.dly_test[indices]
        self.chan_td_sparse = np.fft.ifft(self.chan_fd_sparse, axis=0) 

        