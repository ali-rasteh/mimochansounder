# Near-Field Channel Modeling

## Overview
This python package simualtes parameter extraction for
near-field channel modeling.  We use the reflection
model of [HYR+23] where each path of the channel is 
modeled by the image point of the transmitter.
Hence, the channel modeling problem is to extract
the locations of the image points and the path gains
from each image point.


## Features
- **Channel Modeling**: Simulates near-field channel modeling using reflection models.
- **Parameter Extraction**: Extracts locations of image points and path gains.
- **Sparse Estimation**: Performs sparse estimation for channel response.
- **Simulation Tools**: Includes facilities for creating and running simulations (e.g., `Sim`, `RoomModel`).

## Installation
Clone the repository and install the required dependencies.
```sh
git clone https://github.com/ali-rasteh/mimochansounder.git
cd mimochansounder
pip install -r requirements.txt
```

## Usage
### Example
An example of how to use the package can be found in `data_demo.py`:
```python
from deconv import Deconv

# Parameters
fc = 6e9  # Carrier frequency
fsamp = 1e9  # Sample frequency
nfft = 1024  # Number of FFT points

# Create a Deconv object
dec = Deconv(fc=fc, fsamp=fsamp, nfft=nfft)

# Load and process data
calib_path = 'data/chamber.npz'
data_path = 'data/wall_reflection.npz'
dec.set_system_resp_data(calib_path)
dec.load_chan_data(data_path)
dec.compute_chan_resp()
```

### Simulation
Run the simulation script `simtest.py` to estimate the location of a transmitter in a room:
```sh
python simtest.py
```

## References

* [HYR+23] Hu, Y., Yin, M., Rangan, S., & Mezzavilla, M.
[Parametrization and Estimation of High-Rank Line-of-Sight
MIMO Channels With Reflected 
Paths](https://ieeexplore.ieee.org/abstract/document/10247221),
IEEE Transactions on Wireless Communications, 2023.


## License
This project is licensed under the MIT License.
```
