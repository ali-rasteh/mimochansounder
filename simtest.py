# simtest.py:  Basic simulation of estimating the location of a transmitter in a room
import numpy as np
import matplotlib.pyplot as plt
from sigmodel import Sim, RoomModel


# Parameters
fc = 12e9  # Carrier frequency   
fsamp = 1e9 # Sample freq in Hz
nantrx=2  # Number of antennas per array
antsep=np.array([0.5,1,2,4]) # Separation between antennas in wavelengths
rxlocsep = np.array([0,1])    # Separation between RX locations
npath=4   # Number of paths
npath_est = 5  # Maximum number of paths to estimate
plot_type = 'init_est' 

# Construct the room
rm = RoomModel()


# Place a source
xsrc = np.array([6,4])

# Find the reflections
xref = rm.find_reflection(xsrc)

# Create all the transmitters
xtx =  np.vstack((xsrc, xref))


# Create the simulation object and run the simulation
sim = Sim(fc=fc, fsamp=fsamp, nantrx=nantrx,
          rxlocsep=rxlocsep, antsep=antsep, npath=npath,
          tx=xtx,npath_est=npath_est)

plt.plot(np.arange(len(sim.chan_td[:100,0,0])), 20*np.log10(np.abs(sim.chan_td[:100,0,0])))
plt.show()

if (plot_type == 'init_est') or (plot_type == 'iter_est'):
    """
    Plots the heatmaps of the estimated TX locations in each iteration
    """
    if plot_type == 'init_est':
        nplots = 1
    else:
        nplots = sim.npath_det

    fig, ax = plt.subplots(1, nplots)
    if nplots == 1:
        ax = [ax]
    for i in range(nplots):
      
        # Plot the estimated heatmap
        rho1 = sim.rho[:,i].reshape(sim.npoints,sim.npoints)
        ax[i].contourf(sim.xtest0, sim.xtest1, rho1)

        # Draw the room walls
        rm.draw_walls(ax[i])
            
        
        # Plot the TX locations as blue circles
        ax[i].plot(sim.tx[:,0], sim.tx[:,1], 'ro')
        ax[i].plot(sim.tx_est[i,0], sim.tx_est[i,1], 'gx')
        ax[i].set_xlim(sim.region[0])
        ax[i].set_ylim(sim.region[1])
        if (nplots > 1):
            ax[i].set_title(f'Iter {i}')
        if (i > 0):
            ax[i].set_yticks([])
        
        
    plt.tight_layout()





