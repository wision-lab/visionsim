"""
This is a temporary module which contains all the base classes for the active spc emulator support.
The classes will be modified/moved to different sub-packages as required. 

General NOTE: When implementing all the functions ensure that they support differentiable pipelines. 

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import cast


#######################################
# Classes related to scene properties #
#######################################

class LightSource:
    """Base class for any type of light source, active, passive, constant, time varying etc.    
    Any type of light source must inherit this class"""
    def __init__(self):
        pass

class ConstantSource(LightSource):
    """
    Base class for any light source which has constant light intensity over time."""
    def __init__(self):
        pass

class DynamicSource(LightSource):
    """
    Base class for any time varying light source. Can model flicker or AC light sources?
    """
    def __init__(self):
        super().__init__()

# Encode scene geometry #
#########################


class RGBDStream:
    """Base class to load an RGBD stream from existing datasets."""
    def __init__(self, frame_exposure: int):
        """_summary_

        Args:
            frame_exposure (int): Set the per frame exposure time. Or we can use fps.
        
        ..note:: Can add 
            - Undistortion method? One option is to assume that the RGBD images are already undistorted?
            - dictionary to pass all the camera parameters?
        """
        pass

    def batch_frames(self,batch_size: int):
        """_summary_

        Args:
            batch_size (int): Combines multiple RGBD frames to enable SPC histograms generated combining multiple histograms
        """
        pass


########################
# Single Photon Camera #
########################


class Detector:
    """Base class to set SPC detector properties

    Include optical characteristics:
    - Use From RGBD stream or custom calibration params (focal length, optical center, distortion params etc).
    - Band pass filter/ optical attenuation
    - Sensor FOV per pixel:
        Includes:
        * Output resolution
        * FoV for each pixel (This feature will allow lazy evaluation of high-res transients)
        * Optical effects like vignetting effect
        * Can also add effect of dead pixels consider real hardware
    - Pixel properties:
        * Pixel dimensions
        * Pitch/Area
        * Photon detection efficiency
        * Detector dead time/ Sensor dead time Td
        * Fill factor 
        * dark current
    """
    def __init__(self):
        pass

class ActiveSource(LightSource):
    """Base class for active illumination source used by the SPAD camera
    
    Includes:
        * pulse wavelength
        * Laser power
        * Total photons per cycle
        * Time period/ rep-rate
        * Time jitter
    Modeling choice:
        * Assume a gaussian pulse with known sigma
        * Use a custom IRF (Allows better hardware emulation)
    """
    def __init__(self):
        super().__init__()

class Histogrammer:
    """Base class for all types of histogrammers.

    Includes:
        * Synchronization details (Synchronous, Free running, Asynchronous with deterministic shift)
        * Will act as the main code that controls the interaction between the ActiveSource and Detector instances.
        * Takes a dictionary or param file that contains all the properties of the Active source, 
            detector instances, TDC resolution etc.
        * Timing circuit details:
            - TDC resolution
            - Timing electronics dead time Te
    """

    def __init__(self):
        pass

    def generate_transient(self, fov_masks: npt.NDArray[np.floating]):
        """Takes input FOV mask to generate intermediate transients for desired pixels based on fov masks.

        ..note:: Design some uniform method of passing RGB, Depth, Scene albedo, Scene Reflectance etc. 

        This class interacts with the RGBDStream class, to get RGB, and depth data, and illumination sources.

        Args:
            fov_masks (npt.NDArray[np.floating]): 3D or 4D mask that determines the which pixels of which 
                    frame number are used to compute the transient and the SPC data. 3D mask for a single RGB-D 
                    frame for desired K pixels and a 4D mask to include the temporal aspect if we want to use multiple 
                    RGB-D frames to generate the transient and histograms (simulate motion artifacts).
        """
        pass

    def simulate_photons(self, tr: npt.NDArray[np.floating]):
        """Simulate photon timestamps based on the transient (tr)

        Args:
            tr (npt.NDArray[np.floating]): _description_
        """
        pass


class BaseEWH:
    """Base class for SPCs using equi-width histogrammers. 

    Includes:
        * Number of EWH bins
        * Shifted time gates (different max range than active illumination)?
        * Precision/ bit depth?
    """