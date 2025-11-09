
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# TOPIC 1

class SimulationBox:
    """
    We created this class to represent a 3D box where our Monte Carlo simulation happens.
    Like a container in 3D space where we will throw random points.
    
    This class stores information about:
        x_min, x_max: The left and right walls of the box (in Angstrom units)
        y_min, y_max: The front and back walls of the box (in Angstrom units)
        z_min, z_max: The bottom and top walls of the box (in Angstrom units)
        volume: How much space is inside the box (in Angstrom^3)
    
    Note: Angstrom (Å) is a unit for measuring very small distances(perfect for atoms and molecules).
    1 Angstrom = 0.0000000001 meters.
    """
    
    def __init__(self, x_range: Tuple[float, float], 
                 y_range: Tuple[float, float], 
                 z_range: Tuple[float, float]):
        """
        Initialize simulation box with given ranges.
        
        Parameters:
            x_range: (min, max) for x dimension in Angstrom
            y_range: (min, max) for y dimension in Angstrom
            z_range: (min, max) for z dimension in Angstrom
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        
        self.x_size = self.x_max - self.x_min
        self.y_size = self.y_max - self.y_min
        self.z_size = self.z_max - self.z_min
        
        self.volume = self.x_size * self.y_size * self.z_size
    
    def __repr__(self):
        return (f"SimulationBox(x=[{self.x_min:.2f}, {self.x_max:.2f}], "
                f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}], "
                f"Volume={self.volume:.2f} Å³)")