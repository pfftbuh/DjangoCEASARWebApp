import numpy as np
from collections import deque

class GazeTrackerState:
    def __init__(self):
        # Landmark indices
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 130]
        self.right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 359]
        self.face_indices = [234, 454, 10, 152, 1, 19, 24, 110, 237, 130, 243, 112, 26, 389, 356, 454]
        self.padding = 5
        
        # Distance coefficients
        self.y_dist = [240, 132, 350, 560, 200]
        self.cm_dist = [25, 50, 15, 20, 30]
        self.dist_coff = np.polyfit(self.y_dist, self.cm_dist, deg=2)
        
        # Face landmarks
        self.KEY_FACE_LANDMARKS = {
            "left": 234,
            "right": 454,
            "top": 10,
            "bottom": 152,
            "front": 1
        }
        