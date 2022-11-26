from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Image():
    """ Image pixel values """
    data : np.ndarray
    """ Feature descriptor vector """
    features : np.ndarray
    