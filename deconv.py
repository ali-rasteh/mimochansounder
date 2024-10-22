# deconv.py:  methods for deconvolving an channel response
import numpy as np
import os

class Deconv(object):
    def __init__(self, nfft=1024, fsamp=1e9, fc=6e9, nrx=1, ntx=1, 
                 file_version=1):
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
        file_version : int 
            File version type.  Current 0 or 1
        """
        self.nfft = nfft
        self.fsamp = fsamp
        self.fc = fc
        self.nrx = nrx
        self.ntx = ntx
        self.file_version = file_version

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
        
        if self.file_version  == 0:
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
        else:
            h_est = data['h_est_full']
            h_est = np.transpose(h_est, (3,1,2,0))  # (nfft,nrx,ntx,nframe)
            h_est = h_est[:,:self.nrx, :self.ntx, :]
            return h_est

    def set_system_resp_data(self, path, nframe_avg=1):
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
        if self.file_version == 0:
            txtd, rxtd = self.load_data(path)
    
            # Compute the TX freq domain
            txfd = np.fft.fft(txtd, axis=0)  # (nfft,ntx)
            rxfd = np.fft.fft(rxtd, axis=0)  # (nfft,nrx,nframe)
    
            # Average over the different frames
            rxfd = np.mean(rxfd[:,:,:nframe_avg], axis=2)
    
            # Noise level assumed for the MMSE 
            snr = 40
            wvar = np.mean(np.abs(rxfd)**2)*(10**(-snr/10))
    
            # Compute the RX calibration response.
            # This only works for ntx=1.
            # Also, we do not average over multiple frames
            S = np.conj(txfd[:,0])/(np.abs(txfd[:,0])**2 + wvar)
            self.grxfd = S[:,None]*rxfd
            self.gtxfd = np.ones((self.nfft, self.ntx), dtype=np.complex64)
    
            # Compute the time-domain response
            self.grxtd = np.fft.ifft(self.grxfd, axis=0)
            self.gtxtd = np.fft.ifft(self.gtxfd, axis=0)
            
        else:
            h_est = self.load_data(path)
            self.grxtd = np.mean(h_est[:,:,0,:nframe_avg], axis=2)
            self.grxfd = np.fft.fft(self.grxtd, axis=0)
            self.gtxfd = np.ones((self.nfft, self.ntx), dtype=np.complex64)

    def load_chan_data(self, path):
        """
        Load the channel data

        Parameters
        ----------
        path : str
            Path to the file
        """
        if self.file_version == 0:
            txtd, rxtd = self.load_data(path)
    
            # Save the response
            self.txtd = txtd
            self.rxtd = rxtd
        else:
            self.chan_td = self.load_data(path)
            

    def compute_chan_resp(self):
        """
        Compute the channel response

        """
        if self.file_version == 0:
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
        else:
            self.chan_fd = np.fft.fft(self.chan_td, axis=0)

    def set_system_resp(self, grxfd):
        """
        Set the system response

        Parameters
        ----------
        g : ndarray (nfft,)
            System response

        """
        self.g = grxfd

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

    def sparse_est(self, npaths=1, nframe_avg=1, ndly=10000, drange=[-6,20],
                   cv=True):
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
        cv : bool
            If True, use cross-validation to estimate the number of paths
        """

        # Number of paths stops when test error exceeds training error
        # by 1+cv_tol
        cv_tol = 0.1

        # Compute the channel estimates for training and test
        # by averaging over the different frames
        nframe = self.chan_fd.shape[3]
        if cv:
            if (nframe < 2*nframe_avg):
                raise ValueError('Not enough frames for cross-validation')
            Itr = np.arange(0,nframe_avg)*2
            Its = Itr + 1
            self.chan_fd_tr = np.mean(self.chan_fd[:,0,0,Itr], axis=1)
            self.chan_fd_ts = np.mean(self.chan_fd[:,0,0,Its], axis=1)

            # For the FA probability, we set the threhold to the energy
            # of the max on nfft random basis functions.  The energy
            # on each basis function is exponential with mean 1/nfft.
            # So, the maximum energy is exponential with mean 1/nfft* (\sum_k 1/k)
            t = np.arange(1,self.nfft)
            cv_dec = (1 - 2*np.sum(1/t)/self.nfft)
        else:
            if (nframe < nframe_avg):
                raise ValueError('Not enough f  rames for averaging')
            self.chan_fd_tr = np.mean(self.chan_fd[:,0,0,:nframe_avg],axis=1)
        self.chan_td_tr = np.fft.ifft(self.chan_fd_tr, axis=0)
        
        # Roll training and test so that the peak is at index = 0
        idx = np.argmax(np.abs(self.chan_td_tr))
        self.chan_td_tr = np.roll(self.chan_td_tr, -idx)
        self.chan_fd_tr *= np.exp(2*np.pi*1j*idx*np.arange(self.nfft)/self.nfft)
        if cv:
            self.chan_fd_ts *= np.exp(2*np.pi*1j*idx*np.arange(self.nfft)/self.nfft)

        # Set the delays to test around the peak
        idx = np.argmax(np.abs(self.chan_td_tr))
        self.dly_test = (idx + np.linspace(drange[0],drange[1],ndly))  
        self.dly_test = self.dly_test - (self.dly_test > self.nfft/2)*self.nfft
        self.dly_test /= self.fsamp

        # Create the basis vectors
        self.B = self.create_chan_resp(self.dly_test, basis=True)

        # Use OMP to find the sparse solution
        self.coeff_est = np.zeros(npaths)
        
        resid = self.chan_fd_tr
        indices = []
        indices1 = []
        self.mse_tr = np.zeros(npaths)
        self.mse_ts = np.zeros(npaths)

        npaths_est = 0
        for i in range(npaths):
            
            # Compute the correlation
            cor = np.abs(self.B.conj().T.dot(resid))

            # Add the highest correlation to the list
            idx = np.argmax(cor)
            indices1.append(idx)

            # Use least squares to estimate the coefficients
            self.coeffs_est = np.linalg.lstsq(self.B[:,indices1], self.chan_fd_tr, rcond=None)[0]

            # Compute the resulting sparse channel
            self.chan_fd_sparse = self.B[:,indices1].dot(self.coeffs_est)

            # Compute the current residual 
            resid = self.chan_fd_tr - self.chan_fd_sparse
            
            # Compute the MSE on the training data
            self.mse_tr[i] = np.mean(np.abs(resid)**2)/np.mean(np.abs(self.chan_fd_tr)**2)

            # Compute the MSE on the test data if CV is used
            if cv:
                resid_ts = self.chan_fd_ts - self.chan_fd_sparse
                self.mse_ts[i] = np.mean(np.abs(resid_ts)**2)/np.mean(np.abs(self.chan_fd_ts)**2)

                # Check if path is valid
                if (i > 0):
                    if (self.mse_ts[i] > cv_dec*self.mse_ts[i-1]):
                        break
                if (self.mse_ts[i] > (1+cv_tol)*self.mse_tr[i]):
                    break

            # Updated the number of paths
            npaths_est = i+1
            indices.append(idx)

        self.dly_est = self.dly_test[indices]

        # Use least squares to estimate the coefficients
        self.coeffs_est = np.linalg.lstsq(self.B[:,indices], self.chan_fd_tr, rcond=None)[0]

        # Compute the resulting sparse channel
        self.chan_fd_sparse = self.B[:,indices].dot(self.coeffs_est)
        self.chan_td_sparse = np.fft.ifft(self.chan_fd_sparse, axis=0) 

        