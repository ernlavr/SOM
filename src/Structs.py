from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Image():
    data : np.ndarray
    features : np.ndarray
    def __sub__(self, other):
        return np.linalg.norm(self.features - other.features)
    