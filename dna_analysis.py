
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
        We created this initialization function to set up our simulation box.
        When we create a new box, and then we tell it how big it should be in each direction.
        
        Parameters:
            x_range: A pair of numbers (min, max) telling us the box size in x direction
            y_range: A pair of numbers (min, max) telling us the box size in y direction
            z_range: A pair of numbers (min, max) telling us the box size in z direction
        
        """
        # We store the minimum and maximum positions for each direction
        # This tells us where the walls of our box are located
        self.x_min, self.x_max = x_range # Unpack the x boundaries
        self.y_min, self.y_max = y_range # Unpack the y boundaries
        self.z_min, self.z_max = z_range # Unpack the z boundaries
        
         # We calculate how wide the box is in each direction
        # This is simply: maximum position minus minimum position
        self.x_size = self.x_max - self.x_min # Width of box in x direction
        self.y_size = self.y_max - self.y_min # Width of box in y direction
        self.z_size = self.z_max - self.z_min # Width of box in z direction
        
        # We calculate the total volume of the box
        # Volume of a rectangular box = length × width × height
        # We multiply the size in all three directions
        self.volume = self.x_size * self.y_size * self.z_size
    
    def __repr__(self):
        """
        We created this function to show information about our box in a readable way.
        When we print the box object, Python will use this function to display it nicely.
        
        This is called a "string representation" - it converts our box data into text.
        """
        # We use f-strings (the f before the quotes) to insert our values into the text
        # The :.2f means "show this number with 2 digits after the decimal point"
        # Å³ is the symbol for cubic Angstroms (volume unit)
        return (f"SimulationBox(x=[{self.x_min:.2f}, {self.x_max:.2f}], "
                f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}], "
                f"Volume={self.volume:.2f} Å³)")